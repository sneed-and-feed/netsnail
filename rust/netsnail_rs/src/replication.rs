//! Layer 4 & Layer 5: Minkowski Reachability PVS (`R_j(dt) = v_max * (0.5 * RTT_i + tau_margin)`),
//! Anonymous Portal-Diffracted `AcousticImpulse` Serializer, and `< 1200 B` Baseline Delta
//! Bitpacker with Strict Anti-DMA Observational Non-Interference.

use crate::arena::{
    hash_arena_canonical, AcousticImpulse, NetTwinArena, PlayerTickState, QuantizedEntityState,
    TierACylinder, NT_DT_SEC, NT_MAX_ACOUSTIC_EMITTERS, NT_MAX_PLAYERS, NT_MTU_CEILING_BYTES,
    NT_STANCE_ACTIVE, NT_STANCE_CROUCH,
};
use crate::jitter_morse::circular_tick_age;
use crate::kinematics::{quantize_player_state, SPEED_SPRINT_MPS};
use crate::map_bvh::{compute_anonymous_acoustic_arrival, is_reachability_ball_visible};
use crate::math::canon_f32;

pub const DATAGRAM_MAGIC_U16: u16 = 0x4E53; // "NS"
pub const DATAGRAM_HEADER_BYTES: usize = 16;
pub const DELTA_RECORD_BYTES: usize = 12;   // Within 9-14 B contract
pub const FULL_RECORD_BYTES: usize = 14;    // Within 9-14 B contract
pub const ACOUSTIC_RECORD_BYTES: usize = 12;
pub const TAU_MARGIN_SEC: f32 = 0.016f32;   // 2-tick jitter safety margin (~15.6 ms)

#[inline]
fn crc16_ccitt_bytes(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[inline]
fn compare_acoustic_canonical(a: &AcousticImpulse, b: &AcousticImpulse) -> core::cmp::Ordering {
    a.arrival_tick
        .cmp(&b.arrival_tick)
        .then_with(|| a.azimuth_q12.cmp(&b.azimuth_q12))
        .then_with(|| a.elevation_q12.cmp(&b.elevation_q12))
        .then_with(|| a.band_energy_q8.cmp(&b.band_energy_q8))
}

#[inline]
fn sort_acoustics_canonical(slice: &mut [AcousticImpulse]) {
    let n = slice.len();
    for i in 1..n {
        let mut j = i;
        while j > 0
            && compare_acoustic_canonical(&slice[j], &slice[j - 1]) == core::cmp::Ordering::Less
        {
            slice.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Packs an observer-specific UDP datagram (`< 1200 B` MTU) into `out_mtu_buf`.
///
/// **Anti-DMA Observational Non-Interference Invariant:**
/// Actors strictly outside observer `i`'s Minkowski Reachability PVS cone `R_j(dt)` contribute
/// **zero** entity bytes and never influence any header field, ordering, or checksum.
/// Returns the exact number of packed bytes (`16..=1200`), or `< 0` on invalid buffer/arguments.
pub fn pack_client_datagram_impl(
    arena: &mut NetTwinArena,
    observer_id: u32,
    baseline_ack_tick: u32,
    out_mtu_buf: &mut [u8],
) -> i32 {
    let obs = observer_id as usize;
    if obs >= NT_MAX_PLAYERS || out_mtu_buf.len() < DATAGRAM_HEADER_BYTES {
        return -1;
    }

    let max_cap = out_mtu_buf.len().min(NT_MTU_CEILING_BYTES);
    let force_scalar = arena.telemetry_and_header.simd_mode_override != 0;
    let server_tick = arena.telemetry_and_header.server_tick;
    let cur_slot = (server_tick as usize) & 63;
    let active_mask = arena.telemetry_and_header.active_player_mask;

    // Observer eye pose & Minkowski reachability radius R_j(dt) = v_max * (0.5 * RTT_i + tau_margin)
    let obs_state = arena.tier_b_poses_plates[cur_slot][obs];
    let obs_eye_z = if (obs_state.stance_flags & NT_STANCE_CROUCH) != 0 {
        0.95f32
    } else {
        1.55f32
    };
    let obs_eye = [
        obs_state.pos[0],
        obs_state.pos[1],
        obs_state.pos[2] + obs_eye_z,
    ];
    let rtt_obs_ticks = arena.client_jitter_gates[obs].est_rtt_ticks;
    let half_rtt_sec = 0.5f32 * (rtt_obs_ticks as f32) * NT_DT_SEC;
    let reach_radius_m = canon_f32(SPEED_SPRINT_MPS * (half_rtt_sec + TAU_MARGIN_SEC));

    // Validate baseline_ack_tick against the 64-tick circular ring window
    let has_valid_baseline = match circular_tick_age(server_tick, baseline_ack_tick) {
        Some(age) => age > 0 && age < 64,
        None => false,
    };
    let base_slot = (baseline_ack_tick as usize) & 63;
    let prev_obs_pvs_mask = if has_valid_baseline {
        arena.client_baseline_snapshots.observer_pvs_ring[obs][base_slot]
    } else {
        0u64
    };

    // Reserve 16-byte header at out_mtu_buf[0..16]
    let mut cursor = DATAGRAM_HEADER_BYTES;
    let mut vis_mask: u64 = 0;
    let mut vis_count: u8 = 0;

    for j in 0..NT_MAX_PLAYERS {
        if (active_mask & (1u64 << j)) == 0 {
            continue;
        }

        // Ego observer is always included; every other actor j != obs MUST pass Minkowski ReachBall PVS
        let is_visible = if j == obs {
            true
        } else {
            let target_pos = arena.tier_b_poses_plates[cur_slot][j].pos;
            is_reachability_ball_visible(
                &arena.static_voxel_bvh,
                &obs_eye,
                &target_pos,
                reach_radius_m,
                force_scalar,
            )
        };

        // CRITICAL NON-INTERFERENCE GATE: never read or touch actor j if outside R_j!
        if !is_visible {
            continue;
        }

        // Refresh quantized ring entry for visible actor j
        let cur_q = quantize_player_state(&arena.tier_b_poses_plates[cur_slot][j]);
        arena.client_baseline_snapshots.quantized_ring[cur_slot][j] = cur_q;

        let can_delta = if has_valid_baseline && ((prev_obs_pvs_mask >> j) & 1) != 0 {
            let bq = arena.client_baseline_snapshots.quantized_ring[base_slot][j];
            if bq.valid != 0 {
                let dx = (cur_q.qx_2mm as i32) - (bq.qx_2mm as i32);
                let dy = (cur_q.qy_2mm as i32) - (bq.qy_2mm as i32);
                let dz = (cur_q.qz_2mm as i32) - (bq.qz_2mm as i32);
                if dx.abs() <= 32767 && dy.abs() <= 32767 && dz.abs() <= 32767 {
                    Some((dx as i16, dy as i16, dz as i16))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((dx, dy, dz)) = can_delta {
            if cursor + DELTA_RECORD_BYTES > max_cap {
                break;
            }
            out_mtu_buf[cursor] = (j as u8) | 0x80; // bit7 = 1 -> 12-byte baseline delta
            out_mtu_buf[cursor + 1..cursor + 3].copy_from_slice(&dx.to_le_bytes());
            out_mtu_buf[cursor + 3..cursor + 5].copy_from_slice(&dy.to_le_bytes());
            out_mtu_buf[cursor + 5..cursor + 7].copy_from_slice(&dz.to_le_bytes());
            out_mtu_buf[cursor + 7] = cur_q.qvx_16 as u8;
            out_mtu_buf[cursor + 8] = cur_q.qvy_16 as u8;
            out_mtu_buf[cursor + 9] = cur_q.yaw_q8 as u8;
            out_mtu_buf[cursor + 10] = cur_q.pitch_q8 as u8;
            out_mtu_buf[cursor + 11] = (cur_q.stance_u16 & 0xFF) as u8;
            cursor += DELTA_RECORD_BYTES;
        } else {
            if cursor + FULL_RECORD_BYTES > max_cap {
                break;
            }
            out_mtu_buf[cursor] = j as u8; // bit7 = 0 -> 14-byte baseline-entry state (0 pop-in!)
            out_mtu_buf[cursor + 1..cursor + 3].copy_from_slice(&cur_q.qx_2mm.to_le_bytes());
            out_mtu_buf[cursor + 3..cursor + 5].copy_from_slice(&cur_q.qy_2mm.to_le_bytes());
            out_mtu_buf[cursor + 5..cursor + 7].copy_from_slice(&cur_q.qz_2mm.to_le_bytes());
            out_mtu_buf[cursor + 7] = cur_q.qvx_16 as u8;
            out_mtu_buf[cursor + 8] = cur_q.qvy_16 as u8;
            out_mtu_buf[cursor + 9] = cur_q.qvz_16 as u8;
            out_mtu_buf[cursor + 10] = cur_q.yaw_q8 as u8;
            out_mtu_buf[cursor + 11] = cur_q.pitch_q8 as u8;
            out_mtu_buf[cursor + 12] = cur_q.lean_q8 as u8;
            out_mtu_buf[cursor + 13] = (cur_q.stance_u16 & 0xFF) as u8;
            cursor += FULL_RECORD_BYTES;
        }

        vis_mask |= 1u64 << j;
        vis_count = vis_count.saturating_add(1);
    }

    // Record observer's PVS bitmask at cur_slot so future ticks can delta-encode safely
    arena.client_baseline_snapshots.observer_pvs_ring[obs][cur_slot] = vis_mask;

    // Collect anonymous portal-diffracted AcousticImpulses emitted on server_tick
    let mut pending_acoustics = [AcousticImpulse::default(); 16];
    let mut num_acoustics = 0usize;

    let emitter_count = arena.projectiles_acoustic_slab.emitter_count as usize;
    let emitter_head = arena.projectiles_acoustic_slab.emitter_head as usize;
    for k in 0..emitter_count {
        let idx = (emitter_head + NT_MAX_ACOUSTIC_EMITTERS - 1 - k) & (NT_MAX_ACOUSTIC_EMITTERS - 1);
        let ev = &arena.projectiles_acoustic_slab.emitters[idx];
        if ev.active == 0 || ev.emit_tick != server_tick {
            if ev.emit_tick != server_tick && circular_tick_age(server_tick, ev.emit_tick).unwrap_or(99) > 0 {
                break;
            }
            continue;
        }
        if ev.source_player_id == observer_id {
            continue;
        }
        if let Some((az_q12, el_q12, bands)) = compute_anonymous_acoustic_arrival(
            &arena.static_voxel_bvh,
            &obs_eye,
            obs_state.yaw_q15,
            &ev.pos,
            ev.max_radius_m,
            &ev.band_energy_q8,
            force_scalar,
        ) {
            if num_acoustics < pending_acoustics.len() {
                pending_acoustics[num_acoustics] = AcousticImpulse {
                    arrival_tick: server_tick,
                    azimuth_q12: az_q12,
                    elevation_q12: el_q12,
                    band_energy_q8: bands,
                };
                num_acoustics += 1;
            }
        }
    }

    // Sort canonically so impulse order leaks zero player_id slot ordering
    sort_acoustics_canonical(&mut pending_acoustics[..num_acoustics]);

    let mut serialized_acoustics: u8 = 0;
    for imp in &pending_acoustics[..num_acoustics] {
        if cursor + ACOUSTIC_RECORD_BYTES > max_cap {
            break;
        }
        out_mtu_buf[cursor..cursor + 4].copy_from_slice(&imp.arrival_tick.to_le_bytes());
        out_mtu_buf[cursor + 4..cursor + 6].copy_from_slice(&imp.azimuth_q12.to_le_bytes());
        out_mtu_buf[cursor + 6..cursor + 8].copy_from_slice(&imp.elevation_q12.to_le_bytes());
        out_mtu_buf[cursor + 8..cursor + 12].copy_from_slice(&imp.band_energy_q8);
        cursor += ACOUSTIC_RECORD_BYTES;
        serialized_acoustics += 1;
    }

    // Write 16-byte header (checksum covers payload [16..cursor] ONLY — zero occluded arena hash leak!)
    let hdr_flags: u8 = if has_valid_baseline { 1 } else { 0 };
    out_mtu_buf[0..2].copy_from_slice(&DATAGRAM_MAGIC_U16.to_le_bytes());
    out_mtu_buf[2] = observer_id as u8;
    out_mtu_buf[3] = hdr_flags;
    out_mtu_buf[4..8].copy_from_slice(&server_tick.to_le_bytes());
    out_mtu_buf[8..12].copy_from_slice(&baseline_ack_tick.to_le_bytes());
    out_mtu_buf[12] = vis_count;
    out_mtu_buf[13] = serialized_acoustics;
    out_mtu_buf[14..16].copy_from_slice(&0u16.to_le_bytes());

    let crc = crc16_ccitt_bytes(&out_mtu_buf[0..cursor]);
    out_mtu_buf[14..16].copy_from_slice(&crc.to_le_bytes());

    // Zero all trailing bytes in caller buffer up to max_cap for zero residual memory leakage
    if cursor < max_cap {
        out_mtu_buf[cursor..max_cap].fill(0);
    }

    arena.telemetry_and_header.last_datagram_bytes = cursor as u32;
    arena.telemetry_and_header.last_pvs_visible_count = vis_count as u32;
    arena.telemetry_and_header.last_acoustic_impulses = serialized_acoustics as u32;

    cursor as i32
}

#[inline]
fn reconstruct_state_from_quantized(q: &QuantizedEntityState) -> (PlayerTickState, TierACylinder) {
    let x = canon_f32((q.qx_2mm as f32) * 0.002f32);
    let y = canon_f32((q.qy_2mm as f32) * 0.002f32);
    let z = canon_f32((q.qz_2mm as f32) * 0.002f32);
    let vx = canon_f32((q.qvx_16 as f32) * (1.0f32 / 16.0f32));
    let vy = canon_f32((q.qvy_16 as f32) * (1.0f32 / 16.0f32));
    let vz = canon_f32((q.qvz_16 as f32) * (1.0f32 / 16.0f32));

    let stance = q.stance_u16 | NT_STANCE_ACTIVE;
    let height_mm: u16 = if (stance & NT_STANCE_CROUCH) != 0 {
        1150
    } else {
        1800
    };

    let pstate = PlayerTickState {
        pos: [x, y, z],
        vel: [vx, vy, vz],
        yaw_q15: (q.yaw_q8 as i16) << 8,
        pitch_q15: (q.pitch_q8 as i16) << 8,
        lean_q15: (q.lean_q8 as i16) << 8,
        stance_flags: stance,
        cayley_gen_q15: [0; 3],
        stamina_q15: 32767,
        health_q8: 25600,
        team_id: 0,
        weapon_id: 0,
        plate_intact_mask: 0,
        plate_material_bits: 0,
        plate_hp_pools_q8: [0; 4],
    };

    let cyl = TierACylinder {
        x,
        y,
        z_base: z,
        rad_height: TierACylinder::pack_rad_height(350, height_mm),
    };
    (pstate, cyl)
}

/// Unpacks an incoming UDP datagram into `client_arena`, **zeroing (`0x00`)** all out-of-PVS
/// enemy slots in client RAM so a physical PCIe DMA card extracts **zero bits** on occluded actors.
/// Returns unpacked visible actor count (`>= 0`), or `< 0` on invalid/corrupted datagram.
pub fn unpack_client_datagram_impl(
    client_arena: &mut NetTwinArena,
    in_mtu_buf: &[u8],
    mut out_acoustics: Option<&mut [AcousticImpulse]>,
) -> (i32, u32) {
    if in_mtu_buf.len() < DATAGRAM_HEADER_BYTES {
        return (-1, 0);
    }

    let magic = u16::from_le_bytes([in_mtu_buf[0], in_mtu_buf[1]]);
    if magic != DATAGRAM_MAGIC_U16 {
        return (-2, 0);
    }

    let obs_id = in_mtu_buf[2] as usize;
    if obs_id >= NT_MAX_PLAYERS {
        return (-3, 0);
    }
    let _hdr_flags = in_mtu_buf[3];
    let server_tick = u32::from_le_bytes([in_mtu_buf[4], in_mtu_buf[5], in_mtu_buf[6], in_mtu_buf[7]]);
    let baseline_tick = u32::from_le_bytes([in_mtu_buf[8], in_mtu_buf[9], in_mtu_buf[10], in_mtu_buf[11]]);
    let vis_count = in_mtu_buf[12] as usize;
    let acoustic_count = in_mtu_buf[13] as usize;
    let expected_crc = u16::from_le_bytes([in_mtu_buf[14], in_mtu_buf[15]]);

    // First scan record lengths to determine exact packet byte slice for CRC verification
    let mut scan_cursor = DATAGRAM_HEADER_BYTES;
    for _ in 0..vis_count {
        if scan_cursor >= in_mtu_buf.len() {
            return (-4, 0);
        }
        let rec_len = if (in_mtu_buf[scan_cursor] & 0x80) != 0 {
            DELTA_RECORD_BYTES
        } else {
            FULL_RECORD_BYTES
        };
        if scan_cursor + rec_len > in_mtu_buf.len() {
            return (-4, 0);
        }
        scan_cursor += rec_len;
    }
    if scan_cursor + acoustic_count * ACOUSTIC_RECORD_BYTES > in_mtu_buf.len() {
        return (-4, 0);
    }
    let total_pkt_len = scan_cursor + acoustic_count * ACOUSTIC_RECORD_BYTES;

    // Verify CRC-16 with bytes 14..16 zeroed
    let mut crc_check: u16 = 0xFFFF;
    for (idx, &b_orig) in in_mtu_buf[..total_pkt_len].iter().enumerate() {
        let b = if idx == 14 || idx == 15 { 0u8 } else { b_orig };
        crc_check ^= (b as u16) << 8;
        for _ in 0..8 {
            if (crc_check & 0x8000) != 0 {
                crc_check = (crc_check << 1) ^ 0x1021;
            } else {
                crc_check <<= 1;
            }
        }
    }
    if crc_check != expected_crc {
        return (-5, 0);
    }

    let cur_slot = (server_tick as usize) & 63;
    let base_slot = (baseline_tick as usize) & 63;

    // Zero ALL 64 player slots at cur_slot in client RAM first (anti-DMA scrub!)
    for p in 0..NT_MAX_PLAYERS {
        client_arena.tier_a_cylinders[cur_slot][p] = TierACylinder::default();
        client_arena.tier_b_poses_plates[cur_slot][p] = PlayerTickState::default();
        client_arena.client_baseline_snapshots.quantized_ring[cur_slot][p] =
            QuantizedEntityState::default();
    }

    let mut unpacked_mask: u64 = 0;
    let mut cursor = DATAGRAM_HEADER_BYTES;

    for _ in 0..vis_count {
        let tag = in_mtu_buf[cursor];
        let pid = (tag & 0x3F) as usize;
        let is_delta = (tag & 0x80) != 0;

        let q_state = if is_delta {
            let base_q = {
                let ring_q = client_arena.client_baseline_snapshots.quantized_ring[base_slot][pid];
                if ring_q.valid != 0 {
                    ring_q
                } else {
                    client_arena.client_baseline_snapshots.observer_entity_cache[obs_id][pid]
                }
            };
            let dx = i16::from_le_bytes([in_mtu_buf[cursor + 1], in_mtu_buf[cursor + 2]]);
            let dy = i16::from_le_bytes([in_mtu_buf[cursor + 3], in_mtu_buf[cursor + 4]]);
            let dz = i16::from_le_bytes([in_mtu_buf[cursor + 5], in_mtu_buf[cursor + 6]]);
            cursor += DELTA_RECORD_BYTES;
            QuantizedEntityState {
                qx_2mm: base_q.qx_2mm.wrapping_add(dx),
                qy_2mm: base_q.qy_2mm.wrapping_add(dy),
                qz_2mm: base_q.qz_2mm.wrapping_add(dz),
                qvx_16: in_mtu_buf[cursor - 5] as i8,
                qvy_16: in_mtu_buf[cursor - 4] as i8,
                qvz_16: 0,
                yaw_q8: in_mtu_buf[cursor - 3] as i8,
                pitch_q8: in_mtu_buf[cursor - 2] as i8,
                lean_q8: base_q.lean_q8,
                stance_u16: (in_mtu_buf[cursor - 1] as u16) | NT_STANCE_ACTIVE,
                valid: 1,
            }
        } else {
            let qx = i16::from_le_bytes([in_mtu_buf[cursor + 1], in_mtu_buf[cursor + 2]]);
            let qy = i16::from_le_bytes([in_mtu_buf[cursor + 3], in_mtu_buf[cursor + 4]]);
            let qz = i16::from_le_bytes([in_mtu_buf[cursor + 5], in_mtu_buf[cursor + 6]]);
            let q = QuantizedEntityState {
                qx_2mm: qx,
                qy_2mm: qy,
                qz_2mm: qz,
                qvx_16: in_mtu_buf[cursor + 7] as i8,
                qvy_16: in_mtu_buf[cursor + 8] as i8,
                qvz_16: in_mtu_buf[cursor + 9] as i8,
                yaw_q8: in_mtu_buf[cursor + 10] as i8,
                pitch_q8: in_mtu_buf[cursor + 11] as i8,
                lean_q8: in_mtu_buf[cursor + 12] as i8,
                stance_u16: (in_mtu_buf[cursor + 13] as u16) | NT_STANCE_ACTIVE,
                valid: 1,
            };
            cursor += FULL_RECORD_BYTES;
            q
        };

        client_arena.client_baseline_snapshots.quantized_ring[cur_slot][pid] = q_state;
        client_arena.client_baseline_snapshots.observer_entity_cache[obs_id][pid] = q_state;
        let (pstate, cyl) = reconstruct_state_from_quantized(&q_state);
        client_arena.tier_b_poses_plates[cur_slot][pid] = pstate;
        client_arena.tier_a_cylinders[cur_slot][pid] = cyl;
        unpacked_mask |= 1u64 << pid;
    }

    // Scrub entity cache for any actor NOT currently inside observer's Minkowski PVS cone
    for p in 0..NT_MAX_PLAYERS {
        if ((unpacked_mask >> p) & 1) == 0 {
            client_arena.client_baseline_snapshots.observer_entity_cache[obs_id][p] =
                QuantizedEntityState::default();
        }
    }

    // Unpack anonymous AcousticImpulse structs (zero entity_id, zero xyz)
    client_arena.client_baseline_snapshots.unpacked_acoustics.fill(AcousticImpulse::default());
    let mut written_acoustics = 0u32;
    for k in 0..acoustic_count {
        let arr_tick = u32::from_le_bytes([
            in_mtu_buf[cursor],
            in_mtu_buf[cursor + 1],
            in_mtu_buf[cursor + 2],
            in_mtu_buf[cursor + 3],
        ]);
        let az = i16::from_le_bytes([in_mtu_buf[cursor + 4], in_mtu_buf[cursor + 5]]);
        let el = i16::from_le_bytes([in_mtu_buf[cursor + 6], in_mtu_buf[cursor + 7]]);
        let bands = [
            in_mtu_buf[cursor + 8],
            in_mtu_buf[cursor + 9],
            in_mtu_buf[cursor + 10],
            in_mtu_buf[cursor + 11],
        ];
        cursor += ACOUSTIC_RECORD_BYTES;

        let imp = AcousticImpulse {
            arrival_tick: arr_tick,
            azimuth_q12: az,
            elevation_q12: el,
            band_energy_q8: bands,
        };
        if k < 256 {
            client_arena.client_baseline_snapshots.unpacked_acoustics[k] = imp;
        }
        if let Some(ref mut out_slice) = out_acoustics {
            if k < out_slice.len() {
                out_slice[k] = imp;
            }
        }
        written_acoustics += 1;
    }

    client_arena.client_baseline_snapshots.unpacked_acoustic_count = written_acoustics;
    client_arena.client_baseline_snapshots.unpacked_observer_id = obs_id as u32;

    client_arena.telemetry_and_header.server_tick = server_tick;
    client_arena.telemetry_and_header.active_player_mask = unpacked_mask;
    client_arena.telemetry_and_header.last_datagram_bytes = total_pkt_len as u32;
    client_arena.telemetry_and_header.last_pvs_visible_count = vis_count as u32;
    client_arena.telemetry_and_header.last_acoustic_impulses = written_acoustics;
    client_arena.telemetry_and_header.state_digest_u64 = hash_arena_canonical(client_arena);

    (vis_count as i32, written_acoustics)
}
