//! Layer 2 & Layer 3b: `320 KB` Sub-Tick Multi-Plate Rewind (`alpha = fire_subtick_u8 / 256.0`),
//! 4-Wide `0-ULP` SIMD/Scalar Broadphase Cylinder Sweep (`AVX2/SSE2`, `NEON`, `WASM32 simd128`, `Scalar`),
//! 30-Zone Anatomical Multi-Plate Obliquity Ricochet vs Penetration, Mutual Occluder Horizon Cutoff,
//! and Long-Range (`> 25 m`) / Ricochet Finite-Velocity Ballistic Projectiles.

use crate::arena::{
    arena_force_scalar, hash_arena_canonical, BallisticProjectile, ClientInputCmd, HitRecordOut,
    NetTwinArena, PlayerTickState, NT_DT_SEC, NT_HIT_FLAG_HIT,
    NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED, NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
    NT_HIT_FLAG_PENETRATED, NT_HIT_FLAG_RICOCHET, NT_MAX_PLAYERS, NT_MAX_PROJECTILES,
    NT_STANCE_ACTIVE, NT_STANCE_CROUCH,
};
use crate::jitter_morse::circular_tick_age;
use crate::kinematics::{
    push_server_acoustic_emitter, quantize_player_state, GRAVITY_MPS2, HEIGHT_CROUCH_M,
    HEIGHT_STAND_M, PLAYER_RADIUS_M,
};
use crate::map_bvh::{is_capsule_100pct_occluded, raycast_hard_cover};
use crate::math::{
    abs_f32, canon_f32, cos_q15_deterministic, sin_q15_deterministic, sqrt_f32_deterministic,
};

/// Maximum instantaneous CQB hitscan distance before handing off to finite-velocity projectiles.
pub const CQB_HITSCAN_MAX_M: f32 = 25.0f32;

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
pub fn weapon_profile(slot: u8) -> (f32, f32, f32, f32) {
    // Returns (muzzle_vel_mps, base_damage, pen_power, drag_coeff)
    match slot {
        1 => (840.0f32, 68.0f32, 64.0f32, 0.0009f32), // 7.62x51 AP Marksman
        2 => (430.0f32, 54.0f32, 21.0f32, 0.0022f32), // 9x19 SMG
        _ => (910.0f32, 46.0f32, 43.0f32, 0.0012f32), // 5.56x45 M855A1 Carbine
    }
}

/// Shortest modular `i16` arc interpolation for `yaw_q15` across the `±pi` (`-32768 / +32767`) seam.
#[inline]
pub fn lerp_yaw_q15_shortest(yaw0: i16, yaw1: i16, alpha: f32) -> i16 {
    let dyaw: i16 = yaw1.wrapping_sub(yaw0);
    let step: i16 = ((dyaw as f32) * alpha) as i16;
    yaw0.wrapping_add(step)
}

#[inline]
pub fn map_hit_to_30_plate_zone(u_z: f32, front_cos: f32, right_cos: f32) -> (u16, usize) {
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

/// Resolves the 30-zone anatomical armor zone at sub-tick `alpha` using shortest-arc `yaw_q15` interpolation.
#[inline]
pub fn resolve_30_plate_zone(
    s0: &PlayerTickState,
    s1: &PlayerTickState,
    alpha: f32,
    normal: &[f32; 3],
    u_z: f32,
) -> (u16, usize) {
    let v_yaw = lerp_yaw_q15_shortest(s0.yaw_q15, s1.yaw_q15, alpha);
    let v_cy = cos_q15_deterministic(v_yaw);
    let v_sy = sin_q15_deterministic(v_yaw);
    let front_cos = normal[0] * v_cy + normal[1] * v_sy;
    let right_cos = normal[0] * v_sy - normal[1] * v_cy;
    map_hit_to_30_plate_zone(u_z, front_cos, right_cos)
}

/// 4-wide SIMD / Scalar swept cylinder test returning a 4-bit lane hit mask (`(dx_i^2 + dy_i^2) <= r2_i`)
/// with strict `0-ULP` parity across `x86_64`, `AArch64 NEON`, `WASM32 +simd128`, and `Scalar`.
#[inline]
pub fn sweep_cylinders_quad_mask(
    dx: &[f32; 4],
    dy: &[f32; 4],
    r2: &[f32; 4],
    force_scalar: bool,
) -> u8 {
    if force_scalar {
        let mut mask = 0u8;
        for i in 0..4 {
            let d2 = canon_f32(dx[i] * dx[i] + dy[i] * dy[i]);
            if d2 <= r2[i] {
                mask |= 1u8 << i;
            }
        }
        return mask;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let vx = _mm_loadu_ps(dx.as_ptr());
        let vy = _mm_loadu_ps(dy.as_ptr());
        let vr2 = _mm_loadu_ps(r2.as_ptr());
        let d2 = _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy));
        let cmp = _mm_cmple_ps(d2, vr2);
        return (_mm_movemask_ps(cmp) as u8) & 0x0F;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vx = vld1q_f32(dx.as_ptr());
        let vy = vld1q_f32(dy.as_ptr());
        let vr2 = vld1q_f32(r2.as_ptr());
        let d2 = vaddq_f32(vmulq_f32(vx, vx), vmulq_f32(vy, vy));
        let cmp = vcleq_f32(d2, vr2);
        let b0 = (vgetq_lane_u32::<0>(cmp) & 1) as u8;
        let b1 = (vgetq_lane_u32::<1>(cmp) & 1) as u8;
        let b2 = (vgetq_lane_u32::<2>(cmp) & 1) as u8;
        let b3 = (vgetq_lane_u32::<3>(cmp) & 1) as u8;
        return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let vx = v128_load(dx.as_ptr() as *const v128);
        let vy = v128_load(dy.as_ptr() as *const v128);
        let vr2 = v128_load(r2.as_ptr() as *const v128);
        let d2 = f32x4_add(f32x4_mul(vx, vx), f32x4_mul(vy, vy));
        let cmp = f32x4_le(d2, vr2);
        return i32x4_bitmask(cmp) & 0x0F;
    }
    #[allow(unreachable_code)]
    {
        let mut mask = 0u8;
        for i in 0..4 {
            let d2 = canon_f32(dx[i] * dx[i] + dy[i] * dy[i]);
            if d2 <= r2[i] {
                mask |= 1u8 << i;
            }
        }
        mask
    }
}

#[inline]
fn encode_ignore_vid_pad(ignore_vid: u16) -> [u8; 8] {
    let b = ignore_vid.to_le_bytes();
    [b[0], b[1], 0, 0, 0, 0, 0, 0]
}

#[inline]
fn decode_ignore_vid_pad(pad: &[u8; 8]) -> u16 {
    // Explicit sentinel 0xFFFF means no ignored victim; 0..63 is a valid victim_id
    u16::from_le_bytes([pad[0], pad[1]])
}

#[inline]
fn spawn_ballistic_projectile_custom(
    arena: &mut NetTwinArena,
    shooter_id: u32,
    weapon_slot: u8,
    start_pos: [f32; 3],
    vel: [f32; 3],
    initial_traveled_m: f32,
    ricochet_count: u32,
    ignore_vid: u16,
) {
    let (_muzzle_vel, base_dmg, pen_power, drag_coeff) = weapon_profile(weapon_slot);
    let slab = &mut arena.projectiles_acoustic_slab;
    let slot = (slab.projectile_head as usize) & (NT_MAX_PROJECTILES - 1);
    slab.projectile_head = slab.projectile_head.wrapping_add(1);

    slab.projectiles[slot] = BallisticProjectile {
        pos: [
            canon_f32(start_pos[0]),
            canon_f32(start_pos[1]),
            canon_f32(start_pos[2]),
        ],
        vel: [canon_f32(vel[0]), canon_f32(vel[1]), canon_f32(vel[2])],
        drag_coeff,
        mass_kg: 0.004f32,
        pen_power,
        base_damage: base_dmg,
        spawn_tick: arena.telemetry_and_header.server_tick,
        shooter_id: shooter_id as u16,
        weapon_id: weapon_slot,
        active: 1,
        traveled_dist_m: canon_f32(initial_traveled_m),
        ricochet_count,
        _pad: encode_ignore_vid_pad(ignore_vid),
    };

    let mut count = 0u32;
    for p in &slab.projectiles {
        if p.active != 0 {
            count += 1;
        }
    }
    arena.telemetry_and_header.active_projectile_count = count;
}

/// Ray vs leaned upright player cylinder/dome intersection at sub-tick lerped state.
/// Interpolates both `yaw_q15` (along shortest modular `i16` arc) and `lean_q15` linearly by `alpha`.
/// Returns `Some((t_hit, hit_pt, outward_normal, u_z))`.
pub fn intersect_ray_subtick_capsule(
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

    // Interpolate both shoulder-lean AND shortest-arc modular yaw by alpha
    let lean_f = (s0.lean_q15 as f32) + alpha * ((s1.lean_q15 as f32) - (s0.lean_q15 as f32));
    let lean_shift_m = lean_f * (0.20f32 / 4096.0f32);
    let lerp_yaw = lerp_yaw_q15_shortest(s0.yaw_q15, s1.yaw_q15, alpha);
    let cy = cos_q15_deterministic(lerp_yaw);
    let sy = sin_q15_deterministic(lerp_yaw);
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

/// Evaluates 30-zone multi-plate obliquity ricochet vs penetration and applies authoritative
/// `saturating_sub` armor durability & health degradation at `cur_slot`.
/// Returns `(outcome_flags, flesh_dmg, armor_dmg)`.
fn apply_30_plate_ballistic_impact(
    arena: &mut NetTwinArena,
    cur_slot: usize,
    vid: usize,
    zone_id: u16,
    pool_idx: usize,
    dir: &[f32; 3],
    normal: &[f32; 3],
    base_dmg: f32,
    pen_power: f32,
    base_flags: u8,
) -> (u8, f32, f32) {
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
    arena.client_baseline_snapshots.quantized_ring[cur_slot][vid] =
        quantize_player_state(&cur_victim);
    arena.telemetry_and_header.total_hits_confirmed =
        arena.telemetry_and_header.total_hits_confirmed.wrapping_add(1);

    (flags, flesh_dmg, armor_dmg)
}

/// Steps all active finite-velocity ballistic projectiles (`> 25 m` long-range handoff & ricochets)
/// against BOTH `static_voxel_bvh` hard cover (with shallow ricochet) AND active enemy capsules +
/// 30-plate armor zones (ignoring `shooter_id` and any bounced victim).
pub fn step_ballistic_projectiles(
    arena: &mut NetTwinArena,
    cur_slot: usize,
    force_scalar: bool,
) -> u32 {
    let active_mask = arena.telemetry_and_header.active_player_mask;
    let mut active_proj = 0u32;

    for idx in 0..NT_MAX_PROJECTILES {
        let mut p = arena.projectiles_acoustic_slab.projectiles[idx];
        if p.active == 0 {
            continue;
        }
        let speed_sq = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]) + p.vel[2] * p.vel[2];
        if speed_sq < 25.0f32 || p.traveled_dist_m > 400.0f32 {
            p.active = 0;
            arena.projectiles_acoustic_slab.projectiles[idx] = p;
            continue;
        }

        let speed = sqrt_f32_deterministic(speed_sq, force_scalar);
        let inv_spd = 1.0f32 / speed;
        let dir = [p.vel[0] * inv_spd, p.vel[1] * inv_spd, p.vel[2] * inv_spd];
        let step_len = speed * NT_DT_SEC;

        let wall_hit = raycast_hard_cover(&arena.static_voxel_bvh, &p.pos, &dir, step_len);
        let seg_limit = match wall_hit {
            Some((tw, _)) => tw,
            None => step_len,
        };

        // Sweep [p.pos, seg_limit] against active enemy capsules (ignoring shooter & prior ricochet victim)
        let ignore_vid = decode_ignore_vid_pad(&p._pad) as usize;
        let mut best_t = seg_limit;
        let mut best_capsule_hit: Option<(usize, [f32; 3], [f32; 3], f32)> = None;

        for vid in 0..NT_MAX_PLAYERS {
            if (active_mask & (1u64 << vid)) == 0 {
                continue;
            }
            if vid == (p.shooter_id as usize) || vid == ignore_vid {
                continue;
            }
            let st = &arena.tier_b_poses_plates[cur_slot][vid];
            if (st.stance_flags & NT_STANCE_ACTIVE) == 0 {
                continue;
            }
            if let Some((t_hit, hit_pt, normal, u_z)) =
                intersect_ray_subtick_capsule(&p.pos, &dir, st, st, 0.0f32, best_t, force_scalar)
            {
                best_t = t_hit;
                best_capsule_hit = Some((vid, hit_pt, normal, u_z));
            }
        }

        if let Some((vid, hit_pt, normal, u_z)) = best_capsule_hit {
            let st = arena.tier_b_poses_plates[cur_slot][vid];
            let (zone_id, pool_idx) = resolve_30_plate_zone(&st, &st, 0.0f32, &normal, u_z);
            let (ref_muzzle_vel, _bd, _pp, _dc) = weapon_profile(p.weapon_id);
            let vel_ratio = (speed / ref_muzzle_vel.max(1.0f32)).clamp(0.25f32, 1.0f32);
            let eff_dmg = canon_f32(p.base_damage * vel_ratio);
            let eff_pen = canon_f32(p.pen_power * vel_ratio);

            let (flags, _flesh_dmg, _armor_dmg) = apply_30_plate_ballistic_impact(
                arena, cur_slot, vid, zone_id, pool_idx, &dir, &normal, eff_dmg, eff_pen, 0,
            );

            if (flags & NT_HIT_FLAG_RICOCHET) != 0 && p.ricochet_count == 0 {
                let vn = (p.vel[0] * normal[0] + p.vel[1] * normal[1]) + p.vel[2] * normal[2];
                p.pos = [
                    canon_f32(hit_pt[0] + normal[0] * 0.02f32),
                    canon_f32(hit_pt[1] + normal[1] * 0.02f32),
                    canon_f32(hit_pt[2] + normal[2] * 0.02f32),
                ];
                p.vel = [
                    canon_f32((p.vel[0] - 2.0f32 * vn * normal[0]) * 0.55f32),
                    canon_f32((p.vel[1] - 2.0f32 * vn * normal[1]) * 0.55f32),
                    canon_f32((p.vel[2] - 2.0f32 * vn * normal[2]) * 0.55f32),
                ];
                p.ricochet_count += 1;
                p.traveled_dist_m = canon_f32(p.traveled_dist_m + best_t);
                p._pad = encode_ignore_vid_pad(vid as u16);
                active_proj += 1;
            } else {
                p.active = 0;
            }
        } else if let Some((t_hit, normal)) = wall_hit {
            let cos_inc = abs_f32((dir[0] * normal[0] + dir[1] * normal[1]) + dir[2] * normal[2]);
            if cos_inc < 0.35f32 && p.ricochet_count == 0 {
                let vn = (p.vel[0] * normal[0] + p.vel[1] * normal[1]) + p.vel[2] * normal[2];
                p.pos[0] = canon_f32(p.pos[0] + dir[0] * t_hit + normal[0] * 0.01f32);
                p.pos[1] = canon_f32(p.pos[1] + dir[1] * t_hit + normal[1] * 0.01f32);
                p.pos[2] = canon_f32(p.pos[2] + dir[2] * t_hit + normal[2] * 0.01f32);
                p.vel[0] = canon_f32((p.vel[0] - 2.0f32 * vn * normal[0]) * 0.65f32);
                p.vel[1] = canon_f32((p.vel[1] - 2.0f32 * vn * normal[1]) * 0.65f32);
                p.vel[2] = canon_f32((p.vel[2] - 2.0f32 * vn * normal[2]) * 0.65f32);
                p.ricochet_count += 1;
                p.traveled_dist_m = canon_f32(p.traveled_dist_m + t_hit);
                arena.telemetry_and_header.total_ricochets =
                    arena.telemetry_and_header.total_ricochets.wrapping_add(1);
                active_proj += 1;
            } else {
                p.active = 0;
            }
        } else {
            p.pos[0] = canon_f32(p.pos[0] + p.vel[0] * NT_DT_SEC);
            p.pos[1] = canon_f32(p.pos[1] + p.vel[1] * NT_DT_SEC);
            p.pos[2] = canon_f32(p.pos[2] + p.vel[2] * NT_DT_SEC);
            if p.pos[2] <= 0.0f32 {
                p.active = 0;
            } else {
                let drag_factor = (1.0f32 - p.drag_coeff * speed * NT_DT_SEC).max(0.5f32);
                p.vel[0] = canon_f32(p.vel[0] * drag_factor);
                p.vel[1] = canon_f32(p.vel[1] * drag_factor);
                p.vel[2] = canon_f32(p.vel[2] * drag_factor - GRAVITY_MPS2 * NT_DT_SEC);
                p.traveled_dist_m = canon_f32(p.traveled_dist_m + step_len);
                active_proj += 1;
            }
        }
        arena.projectiles_acoustic_slab.projectiles[idx] = p;
    }

    active_proj
}

/// Resolves an authoritative sub-tick shot (`nt_resolve_subtick_fire`), enforcing:
/// 1. Layer-3 Morse/H_clock lag-switch rewind horizon clamping (`NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED`),
/// 2. Instantaneous CQB (`<= 25.0 m`) 4-wide `0-ULP` SIMD/Scalar broadphase sweep over `tier_a_cylinders`,
/// 3. Sub-tick `alpha = fire_subtick_u8 / 256.0` capsule + shortest-arc yaw/lean 30-plate narrowphase,
/// 4. Mutual Occluder Horizon Cutoff (`min(rtt_target_ticks / 2, 8)` -> `NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED`),
/// 5. Handoff to `BallisticProjectile` at `25.0 m` (or shallow ricochet point) with zero `[0, 25 m]` double-sweep.
pub fn resolve_subtick_fire_impl(
    arena: &mut NetTwinArena,
    shooter_id: u32,
    fire_cmd: &ClientInputCmd,
) -> HitRecordOut {
    let force_scalar = arena_force_scalar(arena);
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

    // Unobstructed ray distance against static hard cover (via Voxel DDA + BVH)
    let wall_hit = raycast_hard_cover(&arena.static_voxel_bvh, &muzzle, &dir, 200.0f32);
    let wall_dist = match wall_hit {
        Some((tw, _)) => tw,
        None => 200.0f32,
    };
    let cqb_limit = wall_dist.min(CQB_HITSCAN_MAX_M);

    // 4-wide SIMD / Scalar broadphase sweep across Tier-A cylinders within CQB [0, cqb_limit]
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
        let grp_bits = ((active_mask >> base_p) & 0xF) as u8;
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
            let t_proj = ((rel_x * dir[0] + rel_y * dir[1]) * inv_dir_xy_sq).clamp(0.0f32, cqb_limit);
            dx_arr[lane] = (muzzle[0] + dir[0] * t_proj) - cx;
            dy_arr[lane] = (muzzle[1] + dir[1] * t_proj) - cy;
            r2_arr[lane] = r_bound * r_bound;
        }

        let hit_lanes = sweep_cylinders_quad_mask(&dx_arr, &dy_arr, &r2_arr, force_scalar) & grp_bits;
        candidate_mask |= (hit_lanes as u64) << base_p;
    }

    // Narrowphase sub-tick capsule evaluation restricted to CQB [0, cqb_limit]
    let mut best_t = cqb_limit;
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

    let (muzzle_vel, base_dmg, pen_power, _drag) = weapon_profile(fire_cmd.weapon_slot_u8);

    // No CQB player hit within <= 25.0 m -> spawn long-range (> 25 m) OR shallow wall-ricochet projectile!
    let Some((vid, hit_pt, normal, u_z)) = best_hit else {
        if wall_dist > CQB_HITSCAN_MAX_M {
            let handoff_25m = [
                muzzle[0] + dir[0] * CQB_HITSCAN_MAX_M,
                muzzle[1] + dir[1] * CQB_HITSCAN_MAX_M,
                muzzle[2] + dir[2] * CQB_HITSCAN_MAX_M,
            ];
            let init_vel = [dir[0] * muzzle_vel, dir[1] * muzzle_vel, dir[2] * muzzle_vel];
            spawn_ballistic_projectile_custom(
                arena,
                shooter_id,
                fire_cmd.weapon_slot_u8,
                handoff_25m,
                init_vel,
                CQB_HITSCAN_MAX_M,
                0,
                0xFFFF,
            );
        } else if let Some((tw, w_norm)) = wall_hit {
            let cos_inc =
                abs_f32((dir[0] * w_norm[0] + dir[1] * w_norm[1]) + dir[2] * w_norm[2]);
            if cos_inc < 0.35f32 {
                let dn = (dir[0] * w_norm[0] + dir[1] * w_norm[1]) + dir[2] * w_norm[2];
                let r_speed = muzzle_vel * 0.65f32;
                let r_pos = [
                    muzzle[0] + dir[0] * tw + w_norm[0] * 0.01f32,
                    muzzle[1] + dir[1] * tw + w_norm[1] * 0.01f32,
                    muzzle[2] + dir[2] * tw + w_norm[2] * 0.01f32,
                ];
                let r_vel = [
                    (dir[0] - 2.0f32 * dn * w_norm[0]) * r_speed,
                    (dir[1] - 2.0f32 * dn * w_norm[1]) * r_speed,
                    (dir[2] - 2.0f32 * dn * w_norm[2]) * r_speed,
                ];
                base_flags |= NT_HIT_FLAG_RICOCHET;
                arena.telemetry_and_header.total_ricochets =
                    arena.telemetry_and_header.total_ricochets.wrapping_add(1);
                spawn_ballistic_projectile_custom(
                    arena,
                    shooter_id,
                    fire_cmd.weapon_slot_u8,
                    r_pos,
                    r_vel,
                    tw,
                    1,
                    0xFFFF,
                );
            }
        }

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

    // Resolve victim 30-plate zone using sub-tick shortest-arc yaw interpolation
    let v_s0 = arena.tier_b_poses_plates[slot0][vid];
    let v_s1 = arena.tier_b_poses_plates[slot1][vid];
    let (zone_id, pool_idx) = resolve_30_plate_zone(&v_s0, &v_s1, alpha, &normal, u_z);

    // ========================================================================
    // Layer 3b: Mutual Occluder Horizon Cutoff
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
    let (flags, flesh_dmg, armor_dmg) = apply_30_plate_ballistic_impact(
        arena,
        cur_slot,
        vid,
        zone_id,
        pool_idx,
        &dir,
        &normal,
        base_dmg,
        pen_power,
        base_flags,
    );

    // If CQB hit ricocheted off hard armor plate, spawn secondary ricochet projectile (ignoring vid)
    if (flags & NT_HIT_FLAG_RICOCHET) != 0 {
        let dn = (dir[0] * normal[0] + dir[1] * normal[1]) + dir[2] * normal[2];
        let r_speed = muzzle_vel * 0.55f32;
        let r_pos = [
            hit_pt[0] + normal[0] * 0.02f32,
            hit_pt[1] + normal[1] * 0.02f32,
            hit_pt[2] + normal[2] * 0.02f32,
        ];
        let r_vel = [
            (dir[0] - 2.0f32 * dn * normal[0]) * r_speed,
            (dir[1] - 2.0f32 * dn * normal[1]) * r_speed,
            (dir[2] - 2.0f32 * dn * normal[2]) * r_speed,
        ];
        spawn_ballistic_projectile_custom(
            arena,
            shooter_id,
            fire_cmd.weapon_slot_u8,
            r_pos,
            r_vel,
            best_t,
            1,
            vid as u16,
        );
    }

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
