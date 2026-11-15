//! Layer 2 & Layer 3b: `320 KB` Sub-Tick Multi-Plate Rewind (`alpha = fire_subtick_u8 / 256.0`),
//! 4-Wide `0-ULP` SIMD/Scalar Broadphase Cylinder Sweep, 30-Zone Anatomical Multi-Plate
//! Obliquity Ricochet vs Penetration, Mutual Occluder Horizon Cutoff, and Long-Range
//! (`> 25 m`) Finite-Velocity Ballistic Projectile Spawning.

use crate::arena::{
    hash_arena_canonical, BallisticProjectile, ClientInputCmd, HitRecordOut, NetTwinArena,
    PlayerTickState, NT_HIT_FLAG_HIT, NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED,
    NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED, NT_HIT_FLAG_PENETRATED, NT_HIT_FLAG_RICOCHET,
    NT_MAX_PLAYERS, NT_MAX_PROJECTILES, NT_STANCE_ACTIVE, NT_STANCE_CROUCH,
};
use crate::jitter_morse::circular_tick_age;
use crate::kinematics::{
    push_server_acoustic_emitter, HEIGHT_CROUCH_M, HEIGHT_STAND_M, PLAYER_RADIUS_M,
};
use crate::map_bvh::{is_capsule_100pct_occluded, raycast_hard_cover};
use crate::math::{
    abs_f32, canon_f32, cos_q15_deterministic, dist2_xy_quad_dispatch, sin_q15_deterministic,
    sqrt_f32_deterministic,
};

/// Default 60-bit (30 zones * 2 bits) armor material tier bitfield:
/// Tier 3 = Hardened Titan/Steel (zones 0..2 helmet shell, zone 6 upper front plate)
/// Tier 2 = Ceramic Level IV (zones 7..11 thorax/cummerbund plates)
/// Tier 1 = Aramid IIIA soft armor (zones 3 faceshield, 5 neck, 12..13 deltoids, 16..18 abdomen/pelvis, 24..25 knees)
/// Tier 0 = Unarmored tissue (zones 4 jaw, 14..15 forearms, 19 hands, 20..23 thighs, 26..29 shins/feet)
pub const fn default_plate_material_bits_u64() -> u64 {
    let mut bits: u64 = 0;
    let tiers: [u64; 30] = [
        3, 3, 3, 1, 0, // 0..4: helmet crown, nape, ears, faceshield, jaw
        1,             // 5: neck collar
        3, 2, 2, 2,    // 6..9: upper/lower front & back thorax plates
        2, 2,          // 10..11: left/right cummerbund side plates
        1, 1, 0, 0,    // 12..15: upper arms & forearms
        1, 1, 1, 0,    // 16..19: stomach, lumbar, pelvis, hands
        0, 0, 0, 0,    // 20..23: upper/lower thighs
        1, 1,          // 24..25: knee pads
        0, 0, 0, 0,    // 26..29: shins & boots
    ];
    let mut i = 0;
    while i < 30 {
        bits |= (tiers[i] & 0x3) << (i * 2);
        i += 1;
    }
    bits
}

pub const DEFAULT_INTACT_PLATE_MASK: u32 = 0x3FFF_FFFF; // 30 zones active

#[inline]
fn weapon_profile(slot: u8) -> (f32, f32, f32, f32) {
    // Returns (muzzle_vel_mps, base_damage, pen_power, drag_coeff)
    match slot {
        1 => (840.0f32, 68.0f32, 64.0f32, 0.0009f32), // 7.62x51 AP Marksman
        2 => (430.0f32, 54.0f32, 21.0f32, 0.0022f32), // 9x19 SMG
        _ => (910.0f32, 46.0f32, 43.0f32, 0.0012f32), // 5.56x45 M855A1 Carbine
    }
}

#[inline]
fn map_hit_to_30_plate_zone(
    u_z: f32,
    front_cos: f32,
    right_cos: f32,
) -> (u16, usize) {
    // Returns (zone_id 0..29, armor_pool_idx 0..3: 0=helmet, 1=front, 2=back/side, 3=limb)
    let u = u_z.clamp(0.0f32, 1.0f32);
    if u >= 0.84f32 {
        if u >= 0.93f32 {
            (0, 0)
        } else if front_cos < -0.35f32 {
            (1, 0)
        } else if front_cos > 0.45f32 {
            if u >= 0.88f32 {
                (3, 0)
            } else {
                (4, 0)
            }
        } else {
            (2, 0)
        }
    } else if u >= 0.79f32 {
        (5, 1)
    } else if u >= 0.50f32 {
        if front_cos > 0.45f32 {
            if u >= 0.65f32 {
                (6, 1)
            } else {
                (7, 1)
            }
        } else if front_cos < -0.45f32 {
            if u >= 0.65f32 {
                (8, 2)
            } else {
                (9, 2)
            }
        } else if u >= 0.68f32 {
            if right_cos < 0.0f32 {
                (12, 3)
            } else {
                (13, 3)
            }
        } else if u >= 0.52f32 {
            if right_cos < 0.0f32 {
                (10, 2)
            } else {
                (11, 2)
            }
        } else if right_cos < 0.0f32 {
            (14, 3)
        } else {
            (15, 3)
        }
    } else if u >= 0.38f32 {
        if abs_f32(right_cos) > 0.75f32 {
            (19, 3)
        } else if front_cos >= 0.0f32 {
            if u >= 0.43f32 {
                (16, 1)
            } else {
                (18, 1)
            }
        } else {
            (17, 2)
        }
    } else {
        let is_right = right_cos >= 0.0f32;
        if u >= 0.26f32 {
            (if is_right { 21 } else { 20 }, 3)
        } else if u >= 0.19f32 {
            (if is_right { 23 } else { 22 }, 3)
        } else if u >= 0.14f32 {
            (if is_right { 25 } else { 24 }, 3)
        } else if u >= 0.05f32 {
            (if is_right { 27 } else { 26 }, 3)
        } else {
            (if is_right { 29 } else { 28 }, 3)
        }
    }
}

#[inline]
fn spawn_long_range_projectile(
    arena: &mut NetTwinArena,
    shooter_id: u32,
    weapon_slot: u8,
    start_pos: [f32; 3],
    dir: [f32; 3],
    initial_traveled_m: f32,
) {
    let (muzzle_vel, base_dmg, pen_power, drag_coeff) = weapon_profile(weapon_slot);
    let slab = &mut arena.projectiles_acoustic_slab;
    let slot = (slab.projectile_head as usize) & (NT_MAX_PROJECTILES - 1);
    slab.projectile_head = slab.projectile_head.wrapping_add(1);

    slab.projectiles[slot] = BallisticProjectile {
        pos: [
            canon_f32(start_pos[0]),
            canon_f32(start_pos[1]),
            canon_f32(start_pos[2]),
        ],
        vel: [
            canon_f32(dir[0] * muzzle_vel),
            canon_f32(dir[1] * muzzle_vel),
            canon_f32(dir[2] * muzzle_vel),
        ],
        drag_coeff,
        mass_kg: 0.004f32,
        pen_power,
        base_damage: base_dmg,
        spawn_tick: arena.telemetry_and_header.server_tick,
        shooter_id: shooter_id as u16,
        weapon_id: weapon_slot,
        active: 1,
        traveled_dist_m: canon_f32(initial_traveled_m),
        ricochet_count: 0,
        _pad: [0u8; 8],
    };

    // Refresh active projectile counter
    let mut count = 0u32;
    for p in &slab.projectiles {
        if p.active != 0 {
            count += 1;
        }
    }
    arena.telemetry_and_header.active_projectile_count = count;
}

/// Ray vs leaned upright player cylinder/dome intersection at sub-tick lerped state.
/// Returns `Some((t_hit, hit_pt, outward_normal, u_z))`.
fn intersect_ray_subtick_capsule(
    origin: &[f32; 3],
    dir: &[f32; 3],
    s0: &PlayerTickState,
    s1: &PlayerTickState,
    alpha: f32,
    max_dist: f32,
    force_scalar: bool,
) -> Option<(f32, [f32; 3], [f32; 3], f32)> {
    let base_x = s0.pos[0] + alpha * (s1.pos[0] - s0.pos[0]);
    let base_y = s0.pos[1] + alpha * (s1.pos[1] - s0.pos[1]);
    let base_z = s0.pos[2] + alpha * (s1.pos[2] - s0.pos[2]);

    let h0 = if (s0.stance_flags & NT_STANCE_CROUCH) != 0 {
        HEIGHT_CROUCH_M
    } else {
        HEIGHT_STAND_M
    };
    let h1 = if (s1.stance_flags & NT_STANCE_CROUCH) != 0 {
        HEIGHT_CROUCH_M
    } else {
        HEIGHT_STAND_M
    };
    let height = h0 + alpha * (h1 - h0);

    // Shoulder-lean shifts upper torso center along player's right vector
    let lean_f = (s0.lean_q15 as f32) + alpha * ((s1.lean_q15 as f32) - (s0.lean_q15 as f32));
    let lean_shift_m = lean_f * (0.20f32 / 4096.0f32);
    let cy = cos_q15_deterministic(s0.yaw_q15);
    let sy = sin_q15_deterministic(s0.yaw_q15);
    // Right vector = (sy, -cy)
    let cx = base_x + sy * lean_shift_m * 0.65f32;
    let cy_pos = base_y - cy * lean_shift_m * 0.65f32;

    // 2D quadratic in XY for ray vs cylinder of radius PLAYER_RADIUS_M
    let ox = origin[0] - cx;
    let oy = origin[1] - cy_pos;
    let a = dir[0] * dir[0] + dir[1] * dir[1];
    if a < 1e-8f32 {
        return None;
    }
    let b_half = ox * dir[0] + oy * dir[1];
    let c = (ox * ox + oy * oy) - PLAYER_RADIUS_M * PLAYER_RADIUS_M;
    let disc = b_half * b_half - a * c;
    if disc < 0.0f32 {
        return None;
    }
    let root = sqrt_f32_deterministic(disc, force_scalar);
    let mut t = (-b_half - root) / a;
    if t < 0.0f32 {
        t = (-b_half + root) / a;
    }
    if t < 0.0f32 || t > max_dist {
        return None;
    }

    let hx = canon_f32(origin[0] + dir[0] * t);
    let hy = canon_f32(origin[1] + dir[1] * t);
    let hz = canon_f32(origin[2] + dir[2] * t);

    if hz < base_z || hz > base_z + height {
        return None;
    }

    let u_z = ((hz - base_z) / height.max(0.5f32)).clamp(0.0f32, 1.0f32);
    let mut nx = (hx - cx) * (1.0f32 / PLAYER_RADIUS_M);
    let mut ny = (hy - cy_pos) * (1.0f32 / PLAYER_RADIUS_M);
    // Helmet dome upward curvature on top 12% of capsule
    let mut nz = if u_z > 0.88f32 {
        (u_z - 0.88f32) * 5.0f32
    } else {
        0.0f32
    };
    let n_sq = (nx * nx + ny * ny) + nz * nz;
    if n_sq > 1e-8f32 {
        let inv_n = 1.0f32 / sqrt_f32_deterministic(n_sq, force_scalar);
        nx = canon_f32(nx * inv_n);
        ny = canon_f32(ny * inv_n);
        nz = canon_f32(nz * inv_n);
    }

    Some((t, [hx, hy, hz], [nx, ny, nz], u_z))
}

/// Resolves an authoritative sub-tick shot (`nt_resolve_subtick_fire`), enforcing:
/// 1. Layer-3 Morse/H_clock lag-switch rewind horizon clamping (`NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED`),
/// 2. 4-wide `0-ULP` SIMD/Scalar broadphase sweep over `tier_a_cylinders`,
/// 3. Sub-tick `alpha = fire_subtick_u8 / 256.0` capsule + 30-plate narrowphase,
/// 4. Mutual Occluder Horizon Cutoff (`min(rtt_target_ticks / 2, 8)` -> `NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED`),
/// 5. 30-zone obliquity ricochet vs penetration, and `> 25 m` finite-velocity projectile spawning.
pub fn resolve_subtick_fire_impl(
    arena: &mut NetTwinArena,
    shooter_id: u32,
    fire_cmd: &ClientInputCmd,
) -> HitRecordOut {
    let force_scalar = arena.telemetry_and_header.simd_mode_override != 0;
    let server_tick = arena.telemetry_and_header.server_tick;
    let cur_slot = (server_tick as usize) & 63;
    let sid = (shooter_id as usize) & (NT_MAX_PLAYERS - 1);

    arena.telemetry_and_header.total_shots_resolved =
        arena.telemetry_and_header.total_shots_resolved.wrapping_add(1);

    let alpha = canon_f32((fire_cmd.fire_subtick_u8 as f32) * (1.0f32 / 256.0f32));

    // Determine innovation-gated rewind tick
    let shooter_gate = &arena.client_jitter_gates[sid];
    let w_eff = shooter_gate.clamped_rewind_max_ticks.min(62);
    let requested_age = circular_tick_age(server_tick, fire_cmd.cmd_tick).unwrap_or(0);

    let mut base_flags: u8 = 0;
    let rewind_tick = if requested_age > w_eff {
        base_flags |= NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED;
        server_tick.wrapping_sub(w_eff)
    } else {
        fire_cmd.cmd_tick
    };

    let slot0 = (rewind_tick as usize) & 63;
    let slot1 = if rewind_tick == server_tick {
        slot0
    } else {
        (rewind_tick.wrapping_add(1) as usize) & 63
    };

    // Shooter muzzle origin & unit direction from modular q15 angles
    let shooter_state = arena.tier_b_poses_plates[cur_slot][sid];
    let eye_z = if (shooter_state.stance_flags & NT_STANCE_CROUCH) != 0 {
        0.95f32
    } else {
        1.55f32
    };
    let s_cy = cos_q15_deterministic(fire_cmd.yaw_pitch_q15[0]);
    let s_sy = sin_q15_deterministic(fire_cmd.yaw_pitch_q15[0]);
    let s_cp = cos_q15_deterministic(fire_cmd.yaw_pitch_q15[1].clamp(-16000, 16000));
    let s_sp = sin_q15_deterministic(fire_cmd.yaw_pitch_q15[1].clamp(-16000, 16000));

    let lean_m = (shooter_state.lean_q15 as f32) * (0.20f32 / 4096.0f32);
    let muzzle = [
        canon_f32(shooter_state.pos[0] + s_sy * lean_m),
        canon_f32(shooter_state.pos[1] - s_cy * lean_m),
        canon_f32(shooter_state.pos[2] + eye_z),
    ];
    let dir = [
        canon_f32(s_cp * s_cy),
        canon_f32(s_cp * s_sy),
        canon_f32(s_sp),
    ];

    // Emit server-side gunshot acoustic emitter (audible up to 32 m through portals, zero entity_id on wire!)
    push_server_acoustic_emitter(arena, shooter_id, muzzle, 32.0f32, [250, 245, 225, 195]);

    // Maximum unobstructed ray distance against static concrete geometry
    let wall_dist = match raycast_hard_cover(&arena.static_voxel_bvh, &muzzle, &dir, 200.0f32) {
        Some((tw, _)) => tw,
        None => 200.0f32,
    };

    // 4-wide SIMD / Scalar broadphase sweep across Tier-A cylinders
    let active_mask = arena.telemetry_and_header.active_player_mask & !(1u64 << sid);
    let dir_xy_sq = dir[0] * dir[0] + dir[1] * dir[1];
    let inv_dir_xy_sq = if dir_xy_sq > 1e-8f32 {
        1.0f32 / dir_xy_sq
    } else {
        0.0f32
    };

    let mut candidate_mask: u64 = 0;
    for grp in 0..16 {
        let base_p = grp * 4;
        let grp_bits = (active_mask >> base_p) & 0xF;
        if grp_bits == 0 {
            continue;
        }
        let mut dx_arr = [0.0f32; 4];
        let mut dy_arr = [0.0f32; 4];
        let mut r2_arr = [0.0f32; 4];

        for lane in 0..4 {
            let pid = base_p + lane;
            let c0 = arena.tier_a_cylinders[slot0][pid];
            let c1 = arena.tier_a_cylinders[slot1][pid];
            let cx = 0.5f32 * (c0.x + c1.x);
            let cy = 0.5f32 * (c0.y + c1.y);
            let step_sep_sq = (c1.x - c0.x) * (c1.x - c0.x) + (c1.y - c0.y) * (c1.y - c0.y);
            let r_bound = c0.radius_m().max(c1.radius_m())
                + 0.5f32 * sqrt_f32_deterministic(step_sep_sq, force_scalar)
                + 0.25f32;

            let rel_x = cx - muzzle[0];
            let rel_y = cy - muzzle[1];
            let t_proj = ((rel_x * dir[0] + rel_y * dir[1]) * inv_dir_xy_sq).clamp(0.0f32, wall_dist);
            dx_arr[lane] = (muzzle[0] + dir[0] * t_proj) - cx;
            dy_arr[lane] = (muzzle[1] + dir[1] * t_proj) - cy;
            r2_arr[lane] = r_bound * r_bound;
        }

        let d2_quad = dist2_xy_quad_dispatch(&dx_arr, &dy_arr, force_scalar);
        for lane in 0..4 {
            if ((grp_bits >> lane) & 1) != 0 && d2_quad[lane] <= r2_arr[lane] {
                candidate_mask |= 1u64 << (base_p + lane);
            }
        }
    }

    // Narrowphase sub-tick capsule & 30-plate evaluation
    let mut best_t = wall_dist;
    let mut best_hit: Option<(usize, [f32; 3], [f32; 3], f32)> = None;

    for vid in 0..NT_MAX_PLAYERS {
        if (candidate_mask & (1u64 << vid)) == 0 {
            continue;
        }
        let s0 = &arena.tier_b_poses_plates[slot0][vid];
        let s1 = &arena.tier_b_poses_plates[slot1][vid];
        if (s0.stance_flags & NT_STANCE_ACTIVE) == 0 && (s1.stance_flags & NT_STANCE_ACTIVE) == 0 {
            continue;
        }
        if let Some((t_hit, hit_pt, normal, u_z)) =
            intersect_ray_subtick_capsule(&muzzle, &dir, s0, s1, alpha, best_t, force_scalar)
        {
            best_t = t_hit;
            best_hit = Some((vid, hit_pt, normal, u_z));
        }
    }

    // Spawn long-range (> 25 m) finite-velocity ballistic projectile if ray travels > 25 m
    if wall_dist > 25.0f32 {
        let start_25m = [
            muzzle[0] + dir[0] * 25.0f32,
            muzzle[1] + dir[1] * 25.0f32,
            muzzle[2] + dir[2] * 25.0f32,
        ];
        spawn_long_range_projectile(
            arena,
            shooter_id,
            fire_cmd.weapon_slot_u8,
            start_25m,
            dir,
            25.0f32,
        );
    }

    let Some((vid, _hit_pt, normal, u_z)) = best_hit else {
        arena.telemetry_and_header.state_digest_u64 = hash_arena_canonical(arena);
        return HitRecordOut {
            shooter_id,
            victim_id: 0xFFFF_FFFF,
            rewind_tick,
            plate_zone_id: 0xFFFF,
            outcome_flags: base_flags,
            _pad0: 0,
            subtick_alpha: alpha,
            damage_applied: 0.0f32,
            armor_damage: 0.0f32,
            hit_dist_m: canon_f32(wall_dist),
        };
    };

    // Resolve victim plate zone first for full telemetry fidelity
    let v_state_rew = arena.tier_b_poses_plates[slot0][vid];
    let v_cy = cos_q15_deterministic(v_state_rew.yaw_q15);
    let v_sy = sin_q15_deterministic(v_state_rew.yaw_q15);
    let front_cos = normal[0] * v_cy + normal[1] * v_sy;
    let right_cos = normal[0] * v_sy - normal[1] * v_cy;
    let (zone_id, pool_idx) = map_hit_to_30_plate_zone(u_z, front_cos, right_cos);

    // ========================================================================
    // Layer 3b: Mutual Occluder Horizon Cutoff
    // Reject behind-wall hits if victim's capsule over [server_tick - tau_fair, server_tick]
    // (where tau_fair = min(rtt_target_ticks / 2, 8) <= 62.5 ms at 128 Hz) has been
    // 100% occluded behind StaticVoxelBvh hard cover from shooter's muzzle!
    // ========================================================================
    let rtt_target_ticks = arena.client_jitter_gates[vid].est_rtt_ticks;
    let tau_fair = (rtt_target_ticks / 2).min(8);
    let rewind_span = server_tick.wrapping_sub(rewind_tick);

    if rewind_span > tau_fair {
        let mut all_occluded = true;
        for step_back in 0..=tau_fair {
            let check_tick = server_tick.wrapping_sub(step_back);
            let check_slot = (check_tick as usize) & 63;
            let st = &arena.tier_b_poses_plates[check_slot][vid];
            let h = if (st.stance_flags & NT_STANCE_CROUCH) != 0 {
                HEIGHT_CROUCH_M
            } else {
                HEIGHT_STAND_M
            };
            if !is_capsule_100pct_occluded(
                &arena.static_voxel_bvh,
                &muzzle,
                &st.pos,
                PLAYER_RADIUS_M,
                h,
                force_scalar,
            ) {
                all_occluded = false;
                break;
            }
        }

        if all_occluded {
            arena.telemetry_and_header.total_occluder_cutoffs =
                arena.telemetry_and_header.total_occluder_cutoffs.wrapping_add(1);
            arena.telemetry_and_header.state_digest_u64 = hash_arena_canonical(arena);
            return HitRecordOut {
                shooter_id,
                victim_id: vid as u32,
                rewind_tick,
                plate_zone_id: zone_id,
                outcome_flags: base_flags | NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
                _pad0: 0,
                subtick_alpha: alpha,
                damage_applied: 0.0f32,
                armor_damage: 0.0f32,
                hit_dist_m: canon_f32(best_t),
            };
        }
    }

    // ========================================================================
    // 30-Zone Multi-Plate Obliquity Ricochet vs Penetration Evaluation
    // ========================================================================
    let (_muzzle_vel, base_dmg, pen_power, _drag) = weapon_profile(fire_cmd.weapon_slot_u8);
    let cos_obliquity =
        abs_f32((dir[0] * normal[0] + dir[1] * normal[1]) + dir[2] * normal[2]).clamp(0.05f32, 1.0f32);

    let mut cur_victim = arena.tier_b_poses_plates[cur_slot][vid];
    let zone_bit = 1u32 << (zone_id as u32);
    let plate_intact = (cur_victim.plate_intact_mask & zone_bit) != 0;
    let mat_tier = if plate_intact {
        ((cur_victim.plate_material_bits >> ((zone_id as u32) * 2)) & 0x3) as u32
    } else {
        0u32
    };

    let zone_mult = if zone_id <= 4 {
        2.15f32 // Head / Cranial multiplier
    } else if zone_id <= 11 {
        1.05f32 // Thorax vital organs
    } else if zone_id <= 18 {
        0.90f32 // Abdomen / Pelvis
    } else {
        0.65f32 // Extremities
    };

    let mut flags = base_flags | NT_HIT_FLAG_HIT;
    let (flesh_dmg, armor_dmg) = if mat_tier >= 2 && cos_obliquity < 0.44f32 {
        // Steep obliquity (> ~64 deg from plate normal) against Ceramic / Hardened Titan -> RICOCHET!
        flags |= NT_HIT_FLAG_RICOCHET;
        arena.telemetry_and_header.total_ricochets =
            arena.telemetry_and_header.total_ricochets.wrapping_add(1);
        (canon_f32(1.5f32), canon_f32(6.0f32))
    } else if mat_tier == 0 {
        // Unarmored zone -> clean penetration
        flags |= NT_HIT_FLAG_PENETRATED;
        arena.telemetry_and_header.total_penetrations =
            arena.telemetry_and_header.total_penetrations.wrapping_add(1);
        (canon_f32(base_dmg * zone_mult), 0.0f32)
    } else {
        let base_resist = match mat_tier {
            1 => 18.0f32, // Aramid IIIA
            2 => 36.0f32, // Ceramic IV
            _ => 42.0f32, // Hardened Titan/Steel
        };
        let eff_resist = base_resist / cos_obliquity.max(0.30f32);
        if pen_power >= eff_resist {
            flags |= NT_HIT_FLAG_PENETRATED;
            arena.telemetry_and_header.total_penetrations =
                arena.telemetry_and_header.total_penetrations.wrapping_add(1);
            let retained = ((pen_power - 0.35f32 * eff_resist) / pen_power).clamp(0.45f32, 1.0f32);
            (
                canon_f32(base_dmg * zone_mult * retained),
                canon_f32(base_dmg * 0.55f32),
            )
        } else {
            // Stopped by plate -> heavy armor durability loss + minor blunt trauma
            (canon_f32(base_dmg * 0.12f32), canon_f32(base_dmg * 0.85f32))
        }
    };

    // Apply authoritative armor pool & health degradation
    if armor_dmg > 0.0f32 {
        let loss_q8 = ((armor_dmg * 256.0f32) as u32).min(65535) as u16;
        let rem = cur_victim.plate_hp_pools_q8[pool_idx].saturating_sub(loss_q8);
        cur_victim.plate_hp_pools_q8[pool_idx] = rem;
        if rem == 0 {
            cur_victim.plate_intact_mask &= !zone_bit;
        }
    }
    if flesh_dmg > 0.0f32 {
        let hp_loss_q8 = ((flesh_dmg * 256.0f32) as u32).min(65535) as u16;
        cur_victim.health_q8 = cur_victim.health_q8.saturating_sub(hp_loss_q8);
    }
    arena.tier_b_poses_plates[cur_slot][vid] = cur_victim;

    arena.telemetry_and_header.total_hits_confirmed =
        arena.telemetry_and_header.total_hits_confirmed.wrapping_add(1);
    arena.telemetry_and_header.state_digest_u64 = hash_arena_canonical(arena);

    HitRecordOut {
        shooter_id,
        victim_id: vid as u32,
        rewind_tick,
        plate_zone_id: zone_id,
        outcome_flags: flags,
        _pad0: 0,
        subtick_alpha: alpha,
        damage_applied: flesh_dmg,
        armor_damage: armor_dmg,
        hit_dist_m: canon_f32(best_t),
    }
}
