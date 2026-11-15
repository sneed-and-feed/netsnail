//! Layer 4 & Layer 5: Minkowski Reachability PVS (`R_j(dt) = v_max * (0.5 * RTT_i + tau_margin)`),
//! Anonymous Portal-Diffracted `AcousticImpulse` Serializer, and `< 1200 B` Modular Baseline-Delta
//! Bitpacker (`12-14 B` records) with Strict Anti-DMA Observational Non-Interference.

use crate::arena::{
    arena_force_scalar, hash_arena_canonical, AcousticImpulse, NetTwinArena, PlayerTickState,
    QuantizedEntityState, TierACylinder, NT_DT_SEC, NT_MAX_ACOUSTIC_EMITTERS, NT_MAX_PLAYERS,
    NT_MTU_CEILING_BYTES, NT_STANCE_ACTIVE, NT_STANCE_CROUCH, NT_STANCE_MASK_7BIT,
};
use crate::jitter_morse::circular_tick_age;
use crate::kinematics::{quantize_player_state, SPEED_SPRINT_MPS};
use crate::map_bvh::{compute_anonymous_acoustic_arrival, is_reachability_ball_visible};
use crate::math::canon_f32;

pub const DATAGRAM_MAGIC_U16: u16 = 0x4E53; // "NS"
pub const DATAGRAM_HEADER_BYTES: usize = 16;
pub const DELTA_RECORD_BYTES: usize = 12;      // Short modular delta (tag 0x80 | pid)
pub const DELTA_WIDE_RECORD_BYTES: usize = 14; // Wide ZMod 2^16 modular delta (tag 0xC0 | pid)
pub const FULL_RECORD_BYTES: usize = 14;       // Full baseline-entry state (tag 0x00 | pid)
pub const ACOUSTIC_RECORD_BYTES: usize = 12;
pub const TAU_MARGIN_SEC: f32 = 0.016f32;      // 2-tick jitter safety margin (~15.6 ms)

/// 4-wide SIMD / Scalar horizontal distance-squared broadphase helper (`0-ULP` identical across
/// `x86_64`, `AArch64 NEON`, `WASM32 +simd128`, and `Scalar`).
#[inline]
pub fn pvs_horiz_dist2_quad(
    tx: &[f32; 4],
    ty: &[f32; 4],
    ox: f32,
    oy: f32,
    force_scalar: bool,
) -> [f32; 4] {
    if force_scalar {
        let dx0 = tx[0] - ox;
        let dy0 = ty[0] - oy;
        let dx1 = tx[1] - ox;
        let dy1 = ty[1] - oy;
        let dx2 = tx[2] - ox;
        let dy2 = ty[2] - oy;
        let dx3 = tx[3] - ox;
        let dy3 = ty[3] - oy;
        return [
            canon_f32(dx0 * dx0 + dy0 * dy0),
            canon_f32(dx1 * dx1 + dy1 * dy1),
            canon_f32(dx2 * dx2 + dy2 * dy2),
            canon_f32(dx3 * dx3 + dy3 * dy3),
        ];
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let vx = _mm_sub_ps(_mm_loadu_ps(tx.as_ptr()), _mm_set1_ps(ox));
        let vy = _mm_sub_ps(_mm_loadu_ps(ty.as_ptr()), _mm_set1_ps(oy));
        let sum = _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy));
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vx = vsubq_f32(vld1q_f32(tx.as_ptr()), vdupq_n_f32(ox));
        let vy = vsubq_f32(vld1q_f32(ty.as_ptr()), vdupq_n_f32(oy));
        let sum = vaddq_f32(vmulq_f32(vx, vx), vmulq_f32(vy, vy));
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let vx = f32x4_sub(v128_load(tx.as_ptr() as *const v128), f32x4_splat(ox));
        let vy = f32x4_sub(v128_load(ty.as_ptr() as *const v128), f32x4_splat(oy));
        let sum = f32x4_add(f32x4_mul(vx, vx), f32x4_mul(vy, vy));
        let mut out = [0.0f32; 4];
        v128_store(out.as_mut_ptr() as *mut v128, sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[allow(unreachable_code)]
    {
        let dx0 = tx[0] - ox;
        let dy0 = ty[0] - oy;
        let dx1 = tx[1] - ox;
        let dy1 = ty[1] - oy;
        let dx2 = tx[2] - ox;
        let dy2 = ty[2] - oy;
        let dx3 = tx[3] - ox;
        let dy3 = ty[3] - oy;
        [
            canon_f32(dx0 * dx0 + dy0 * dy0),
            canon_f32(dx1 * dx1 + dy1 * dy1),
            canon_f32(dx2 * dx2 + dy2 * dy2),
            canon_f32(dx3 * dx3 + dy3 * dy3),
        ]
    }
}

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

/// Packs an observer-specific UDP datagram (`<= 1200 B` MTU) into `out_mtu_buf`.
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
    let force_scalar = arena_force_scalar(arena);
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

    // Evaluate active actors in 4-wide SIMD broadphase groups (200 m ceiling covers entire raid map)
    const MAX_PVS_HORIZ_SQ: f32 = 40_000.0f32;
    for grp in 0..16 {
        let base_j = grp * 4;
        let grp_bits = (active_mask >> base_j) & 0xF;
        if grp_bits == 0 {
            continue;
        }
        let mut tx = [obs_eye[0]; 4];
        let mut ty = [obs_eye[1]; 4];
        for lane in 0..4 {
            if ((grp_bits >> lane) & 1) != 0 {
                let p = arena.tier_b_poses_plates[cur_slot][base_j + lane].pos;
                tx[lane] = p[0];
                ty[lane] = p[1];
            }
        }
        let d2_quad = pvs_horiz_dist2_quad(&tx, &ty, obs_eye[0], obs_eye[1], force_scalar);

        for lane in 0..4 {
            if ((grp_bits >> lane) & 1) == 0 {
                continue;
            }
            let j = base_j + lane;

            // Ego observer is always included; every other actor j != obs MUST pass Minkowski ReachBall PVS
            let is_visible = if j == obs {
                true
            } else if d2_quad[lane] > MAX_PVS_HORIZ_SQ {
                false
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

            // Refresh full 16-byte QuantizedEntityState in ring for visible actor j
            let cur_q = quantize_player_state(&arena.tier_b_poses_plates[cur_slot][j]);
            arena.client_baseline_snapshots.quantized_ring[cur_slot][j] = cur_q;

            let stance_7b = (cur_q.stance_u16 & NT_STANCE_MASK_7BIT) as u8;

            let baseline_q_opt = if has_valid_baseline && ((prev_obs_pvs_mask >> j) & 1) != 0 {
                let bq = arena.client_baseline_snapshots.quantized_ring[base_slot][j];
                if bq.valid != 0 {
                    Some(bq)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(bq) = baseline_q_opt {
                let dx_i16 = cur_q.qx_2mm.wrapping_sub(bq.qx_2mm);
                let dy_i16 = cur_q.qy_2mm.wrapping_sub(bq.qy_2mm);
                let dz_i16 = cur_q.qz_2mm.wrapping_sub(bq.qz_2mm);

                let dvx = cur_q.qvx_16.wrapping_sub(bq.qvx_16) as u8;
                let dvy = cur_q.qvy_16.wrapping_sub(bq.qvy_16) as u8;
                let dvz = cur_q.qvz_16.wrapping_sub(bq.qvz_16) as u8;
                let dyaw = cur_q.yaw_q8.wrapping_sub(bq.yaw_q8) as u8;
                let dpitch = cur_q.pitch_q8.wrapping_sub(bq.pitch_q8) as u8;
                let dlean = cur_q.lean_q8.wrapping_sub(bq.lean_q8) as u8;

                if (-128..=127).contains(&dx_i16)
                    && (-128..=127).contains(&dy_i16)
                    && (-128..=127).contains(&dz_i16)
                {
                    if cursor + DELTA_RECORD_BYTES > max_cap {
                        break;
                    }
                    // Tag 0b10_pppppp (0x80): 12-byte short modular delta record (0 drift!)
                    out_mtu_buf[cursor] = (j as u8) | 0x80;
                    out_mtu_buf[cursor + 1] = (dx_i16 as i8) as u8;
                    out_mtu_buf[cursor + 2] = (dy_i16 as i8) as u8;
                    out_mtu_buf[cursor + 3] = (dz_i16 as i8) as u8;
                    out_mtu_buf[cursor + 4] = dvx;
                    out_mtu_buf[cursor + 5] = dvy;
                    out_mtu_buf[cursor + 6] = dvz;
                    out_mtu_buf[cursor + 7] = dyaw;
                    out_mtu_buf[cursor + 8] = dpitch;
                    out_mtu_buf[cursor + 9] = dlean;
                    out_mtu_buf[cursor + 10] = stance_7b;
                    out_mtu_buf[cursor + 11] = 0;
                    cursor += DELTA_RECORD_BYTES;
                } else {
                    if cursor + DELTA_WIDE_RECORD_BYTES > max_cap {
                        break;
                    }
                    // Tag 0b11_pppppp (0xC0): 14-byte ZMod 2^16 wide modular delta record (0 drift!)
                    out_mtu_buf[cursor] = (j as u8) | 0xC0;
                    out_mtu_buf[cursor + 1..cursor + 3].copy_from_slice(&dx_i16.to_le_bytes());
                    out_mtu_buf[cursor + 3..cursor + 5].copy_from_slice(&dy_i16.to_le_bytes());
                    out_mtu_buf[cursor + 5..cursor + 7].copy_from_slice(&dz_i16.to_le_bytes());
                    out_mtu_buf[cursor + 7] = dvx;
                    out_mtu_buf[cursor + 8] = dvy;
                    out_mtu_buf[cursor + 9] = dvz;
                    out_mtu_buf[cursor + 10] = dyaw;
                    out_mtu_buf[cursor + 11] = dpitch;
                    out_mtu_buf[cursor + 12] = dlean;
                    out_mtu_buf[cursor + 13] = stance_7b;
                    cursor += DELTA_WIDE_RECORD_BYTES;
                }
            } else {
                if cursor + FULL_RECORD_BYTES > max_cap {
                    break;
                }
                // Tag 0b00_pppppp (0x00): 14-byte baseline-entry state (0 pop-in!)
                out_mtu_buf[cursor] = j as u8;
                out_mtu_buf[cursor + 1..cursor + 3].copy_from_slice(&cur_q.qx_2mm.to_le_bytes());
                out_mtu_buf[cursor + 3..cursor + 5].copy_from_slice(&cur_q.qy_2mm.to_le_bytes());
                out_mtu_buf[cursor + 5..cursor + 7].copy_from_slice(&cur_q.qz_2mm.to_le_bytes());
                out_mtu_buf[cursor + 7] = cur_q.qvx_16 as u8;
                out_mtu_buf[cursor + 8] = cur_q.qvy_16 as u8;
                out_mtu_buf[cursor + 9] = cur_q.qvz_16 as u8;
                out_mtu_buf[cursor + 10] = cur_q.yaw_q8 as u8;
                out_mtu_buf[cursor + 11] = cur_q.pitch_q8 as u8;
                out_mtu_buf[cursor + 12] = cur_q.lean_q8 as u8;
                out_mtu_buf[cursor + 13] = stance_7b;
                cursor += FULL_RECORD_BYTES;
            }

            vis_mask |= 1u64 << j;
            vis_count = vis_count.saturating_add(1);
        }
    }

    // Record observer's PVS bitmask at cur_slot so future ticks can delta-encode safely
    arena.client_baseline_snapshots.observer_pvs_ring[obs][cur_slot] = vis_mask;

    // Collect anonymous portal-diffracted AcousticImpulses emitted on server_tick
    let mut pending_acoustics = [AcousticImpulse::default(); 16];
    let mut num_acoustics = 0usize;

    let emitter_count = arena.projectiles_acoustic_slab.emitter_count as usize;
    let emitter_head = arena.projectiles_acoustic_slab.emitter_head as usize;
    for k in 0..emitter_count {
        let idx =
            (emitter_head + NT_MAX_ACOUSTIC_EMITTERS - 1 - k) & (NT_MAX_ACOUSTIC_EMITTERS - 1);
        let ev = &arena.projectiles_acoustic_slab.emitters[idx];
        if ev.active == 0 || ev.emit_tick != server_tick {
            if ev.emit_tick != server_tick
                && circular_tick_age(server_tick, ev.emit_tick).unwrap_or(99) > 0
            {
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

    let stance = (q.stance_u16 & NT_STANCE_MASK_7BIT) | NT_STANCE_ACTIVE;
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

#[inline]
fn record_len_from_tag(tag: u8) -> usize {
    match tag & 0xC0 {
        0x80 => DELTA_RECORD_BYTES,
        0xC0 => DELTA_WIDE_RECORD_BYTES,
        _ => FULL_RECORD_BYTES,
    }
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
    let server_tick =
        u32::from_le_bytes([in_mtu_buf[4], in_mtu_buf[5], in_mtu_buf[6], in_mtu_buf[7]]);
    let baseline_tick =
        u32::from_le_bytes([in_mtu_buf[8], in_mtu_buf[9], in_mtu_buf[10], in_mtu_buf[11]]);
    let vis_count = in_mtu_buf[12] as usize;
    let acoustic_count = in_mtu_buf[13] as usize;
    let expected_crc = u16::from_le_bytes([in_mtu_buf[14], in_mtu_buf[15]]);

    // Scan record lengths to determine exact packet byte slice for CRC verification
    let mut scan_cursor = DATAGRAM_HEADER_BYTES;
    for _ in 0..vis_count {
        if scan_cursor >= in_mtu_buf.len() {
            return (-4, 0);
        }
        let rec_len = record_len_from_tag(in_mtu_buf[scan_cursor]);
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

    // Save any in-place baseline before cur_slot == base_slot overwrite (though age > 0 guarantees cur_slot != base_slot)
    let saved_base_ring = client_arena.client_baseline_snapshots.quantized_ring[base_slot];

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
        let kind = tag & 0xC0;

        let q_state = if kind == 0x80 {
            let base_q = {
                let ring_q = saved_base_ring[pid];
                if ring_q.valid != 0 {
                    ring_q
                } else {
                    client_arena.client_baseline_snapshots.observer_entity_cache[obs_id][pid]
                }
            };
            let dx = (in_mtu_buf[cursor + 1] as i8) as i16;
            let dy = (in_mtu_buf[cursor + 2] as i8) as i16;
            let dz = (in_mtu_buf[cursor + 3] as i8) as i16;
            let dvx = in_mtu_buf[cursor + 4] as i8;
            let dvy = in_mtu_buf[cursor + 5] as i8;
            let dvz = in_mtu_buf[cursor + 6] as i8;
            let dyaw = in_mtu_buf[cursor + 7] as i8;
            let dpitch = in_mtu_buf[cursor + 8] as i8;
            let dlean = in_mtu_buf[cursor + 9] as i8;
            let stance_7b = (in_mtu_buf[cursor + 10] as u16) & NT_STANCE_MASK_7BIT;
            cursor += DELTA_RECORD_BYTES;

            QuantizedEntityState {
                qx_2mm: base_q.qx_2mm.wrapping_add(dx),
                qy_2mm: base_q.qy_2mm.wrapping_add(dy),
                qz_2mm: base_q.qz_2mm.wrapping_add(dz),
                qvx_16: base_q.qvx_16.wrapping_add(dvx),
                qvy_16: base_q.qvy_16.wrapping_add(dvy),
                qvz_16: base_q.qvz_16.wrapping_add(dvz),
                yaw_q8: base_q.yaw_q8.wrapping_add(dyaw),
                pitch_q8: base_q.pitch_q8.wrapping_add(dpitch),
                lean_q8: base_q.lean_q8.wrapping_add(dlean),
                stance_u16: stance_7b | NT_STANCE_ACTIVE,
                valid: 1,
            }
        } else if kind == 0xC0 {
            let base_q = {
                let ring_q = saved_base_ring[pid];
                if ring_q.valid != 0 {
                    ring_q
                } else {
                    client_arena.client_baseline_snapshots.observer_entity_cache[obs_id][pid]
                }
            };
            let dx = i16::from_le_bytes([in_mtu_buf[cursor + 1], in_mtu_buf[cursor + 2]]);
            let dy = i16::from_le_bytes([in_mtu_buf[cursor + 3], in_mtu_buf[cursor + 4]]);
            let dz = i16::from_le_bytes([in_mtu_buf[cursor + 5], in_mtu_buf[cursor + 6]]);
            let dvx = in_mtu_buf[cursor + 7] as i8;
            let dvy = in_mtu_buf[cursor + 8] as i8;
            let dvz = in_mtu_buf[cursor + 9] as i8;
            let dyaw = in_mtu_buf[cursor + 10] as i8;
            let dpitch = in_mtu_buf[cursor + 11] as i8;
            let dlean = in_mtu_buf[cursor + 12] as i8;
            let stance_7b = (in_mtu_buf[cursor + 13] as u16) & NT_STANCE_MASK_7BIT;
            cursor += DELTA_WIDE_RECORD_BYTES;

            QuantizedEntityState {
                qx_2mm: base_q.qx_2mm.wrapping_add(dx),
                qy_2mm: base_q.qy_2mm.wrapping_add(dy),
                qz_2mm: base_q.qz_2mm.wrapping_add(dz),
                qvx_16: base_q.qvx_16.wrapping_add(dvx),
                qvy_16: base_q.qvy_16.wrapping_add(dvy),
                qvz_16: base_q.qvz_16.wrapping_add(dvz),
                yaw_q8: base_q.yaw_q8.wrapping_add(dyaw),
                pitch_q8: base_q.pitch_q8.wrapping_add(dpitch),
                lean_q8: base_q.lean_q8.wrapping_add(dlean),
                stance_u16: stance_7b | NT_STANCE_ACTIVE,
                valid: 1,
            }
        } else {
            let qx = i16::from_le_bytes([in_mtu_buf[cursor + 1], in_mtu_buf[cursor + 2]]);
            let qy = i16::from_le_bytes([in_mtu_buf[cursor + 3], in_mtu_buf[cursor + 4]]);
            let qz = i16::from_le_bytes([in_mtu_buf[cursor + 5], in_mtu_buf[cursor + 6]]);
            let stance_7b = (in_mtu_buf[cursor + 13] as u16) & NT_STANCE_MASK_7BIT;
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
                stance_u16: stance_7b | NT_STANCE_ACTIVE,
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
    client_arena
        .client_baseline_snapshots
        .unpacked_acoustics
        .fill(AcousticImpulse::default());
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
