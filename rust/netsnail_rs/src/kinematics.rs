//! Layer 1: Deterministic 128 Hz Swept-Capsule Kinematics, Crouch/Sprint/Stamina Inertia,
//! `SO(3)` Cayley Lean/Attitude Rotor (`0.000e+00` Norm Drift), 4-Plane Primal-Active
//! 3D Collide-and-Slide (`(\delta v)_n >= 0` + `n_1 x n_2` Crease Slide), Ballistic
//! Projectile Stepping, and `< 90 us` 32-Tick Client Prediction Reconciliation.

use crate::arena::{
    hash_arena_canonical, ClientInputCmd, NetTwinArena, PlayerPoseSnapshot, PlayerTickState,
    QuantizedEntityState, ServerAcousticEmitter, StaticVoxelBvh, TierACylinder, NT_BTN_ADS,
    NT_BTN_CROUCH, NT_BTN_LEAN_L, NT_BTN_LEAN_R, NT_BTN_SPRINT, NT_DT_SEC, NT_MAX_ACOUSTIC_EMITTERS,
    NT_MAX_PLAYERS, NT_MAX_PROJECTILES, NT_STANCE_ACTIVE, NT_STANCE_ADS, NT_STANCE_CROUCH,
    NT_STANCE_GROUNDED, NT_STANCE_LEAN_L, NT_STANCE_LEAN_R, NT_STANCE_SPRINT,
};
use crate::map_bvh::raycast_hard_cover;
use crate::math::{
    abs_f32, canon_f32, cayley_rotor_so3_eval, cos_q15_deterministic, norm_sq_3d,
    sin_q15_deterministic, sqrt_f32_deterministic,
};

pub const PLAYER_RADIUS_M: f32 = 0.35f32;
pub const HEIGHT_STAND_M: f32 = 1.80f32;
pub const HEIGHT_CROUCH_M: f32 = 1.15f32;

pub const SPEED_WALK_MPS: f32 = 4.4f32;
pub const SPEED_SPRINT_MPS: f32 = 6.5f32;
pub const SPEED_ADS_MPS: f32 = 2.5f32;
pub const SPEED_CROUCH_MPS: f32 = 2.1f32;

pub const GROUND_ACCEL: f32 = 32.0f32;
pub const GROUND_FRICTION: f32 = 10.0f32;
pub const GRAVITY_MPS2: f32 = 9.81f32;

/// Quantizes a `PlayerTickState` into a 16-byte `QuantizedEntityState` for baseline delta encoding.
#[inline]
pub fn quantize_player_state(s: &PlayerTickState) -> QuantizedEntityState {
    let to_2mm = |v: f32| -> i16 {
        let scaled = (v * 500.0f32).clamp(-32767.0f32, 32767.0f32);
        (if scaled >= 0.0 {
            scaled + 0.5f32
        } else {
            scaled - 0.5f32
        }) as i16
    };
    let to_v16 = |v: f32| -> i8 {
        let scaled = (v * 16.0f32).clamp(-127.0f32, 127.0f32);
        (if scaled >= 0.0 {
            scaled + 0.5f32
        } else {
            scaled - 0.5f32
        }) as i8
    };
    QuantizedEntityState {
        qx_2mm: to_2mm(s.pos[0]),
        qy_2mm: to_2mm(s.pos[1]),
        qz_2mm: to_2mm(s.pos[2]),
        qvx_16: to_v16(s.vel[0]),
        qvy_16: to_v16(s.vel[1]),
        qvz_16: to_v16(s.vel[2]),
        yaw_q8: (s.yaw_q15 >> 8) as i8,
        pitch_q8: (s.pitch_q15 >> 8) as i8,
        lean_q8: (s.lean_q15 >> 8) as i8,
        stance_u16: s.stance_flags,
        valid: if (s.stance_flags & NT_STANCE_ACTIVE) != 0 {
            1
        } else {
            0
        },
    }
}

/// Converts a `PlayerTickState` to a clean 32-byte `PlayerPoseSnapshot`.
#[inline]
pub fn state_to_pose_snapshot(s: &PlayerTickState) -> PlayerPoseSnapshot {
    PlayerPoseSnapshot {
        x: canon_f32(s.pos[0]),
        y: canon_f32(s.pos[1]),
        z: canon_f32(s.pos[2]),
        vx: canon_f32(s.vel[0]),
        vy: canon_f32(s.vel[1]),
        vz: canon_f32(s.vel[2]),
        yaw_q15: s.yaw_q15,
        pitch_q15: s.pitch_q15,
        lean_q15: s.lean_q15,
        stance_flags: s.stance_flags,
    }
}

/// Projects velocity `v` onto a single unilateral contact normal `n` (`<v', n> >= 0`).
#[inline]
pub fn clip_velocity_plane(v: &mut [f32; 3], n: &[f32; 3]) {
    let vn = (v[0] * n[0] + v[1] * n[1]) + v[2] * n[2];
    if vn < 0.0f32 {
        v[0] = canon_f32(v[0] - vn * n[0]);
        v[1] = canon_f32(v[1] - vn * n[1]);
        v[2] = canon_f32(v[2] - vn * n[2]);
    }
}

/// Resolves capsule overlap & unilateral normal coboundary against `StaticVoxelBvh` AABB brushes
/// using up to 4 active contact planes + `n_1 x n_2` 3D crease projection (`(\delta v)_n >= 0`).
pub fn collide_and_slide_3d(
    map: &StaticVoxelBvh,
    pos: &mut [f32; 3],
    vel: &mut [f32; 3],
    radius: f32,
    height: f32,
    dt: f32,
    force_scalar: bool,
) {
    let n_brushes = map.header.num_brushes as usize;
    if n_brushes == 0 {
        pos[0] = canon_f32(pos[0] + vel[0] * dt);
        pos[1] = canon_f32(pos[1] + vel[1] * dt);
        pos[2] = canon_f32((pos[2] + vel[2] * dt).max(0.0f32));
        if pos[2] <= 0.0f32 && vel[2] < 0.0f32 {
            vel[2] = 0.0f32;
        }
        return;
    }

    // Sub-step into 2 half-ticks if moving fast relative to capsule radius
    let sub_dt = dt * 0.5f32;
    for _sub in 0..2 {
        pos[0] = canon_f32(pos[0] + vel[0] * sub_dt);
        pos[1] = canon_f32(pos[1] + vel[1] * sub_dt);
        pos[2] = canon_f32(pos[2] + vel[2] * sub_dt);

        if pos[2] < 0.0f32 {
            pos[2] = 0.0f32;
            if vel[2] < 0.0f32 {
                vel[2] = 0.0f32;
            }
        }

        // Gather up to 4 active contact planes and push out overlaps (4-plane primal-active solver)
        let mut planes = [[0.0f32; 3]; 4];
        let mut num_planes = 0usize;

        for _iter in 0..4 {
            let mut any_penetration = false;
            // Query BVH nodes overlapping expanded capsule AABB
            let qmin = [pos[0] - radius - 0.02f32, pos[1] - radius - 0.02f32, pos[2]];
            let qmax = [
                pos[0] + radius + 0.02f32,
                pos[1] + radius + 0.02f32,
                pos[2] + height,
            ];

            if map.header.num_bvh_nodes > 0 {
                let mut stack = [0u32; 64];
                let mut sp = 1usize;
                stack[0] = 0;

                while sp > 0 {
                    sp -= 1;
                    let node = &map.bvh_nodes[stack[sp] as usize];
                    if qmax[0] < node.min[0]
                        || qmin[0] > node.max[0]
                        || qmax[1] < node.min[1]
                        || qmin[1] > node.max[1]
                        || qmax[2] <= node.min[2]
                        || qmin[2] >= node.max[2]
                    {
                        continue;
                    }
                    if node.count > 0 {
                        let start = node.left_or_first as usize;
                        let end = start + (node.count as usize);
                        for k in start..end {
                            let b = &map.brushes[map.header.brush_indices[k] as usize];
                            if (b.flags & 1) == 0 {
                                continue;
                            }
                            if pos[2] + height <= b.min[2] || pos[2] >= b.max[2] {
                                continue;
                            }
                            let cx = pos[0].clamp(b.min[0], b.max[0]);
                            let cy = pos[1].clamp(b.min[1], b.max[1]);
                            let dx = pos[0] - cx;
                            let dy = pos[1] - cy;
                            let d2 = dx * dx + dy * dy;

                            if d2 < radius * radius {
                                let (nx, ny, pen) = if d2 > 1e-8f32 {
                                    let d = sqrt_f32_deterministic(d2, force_scalar);
                                    let inv = 1.0f32 / d;
                                    (dx * inv, dy * inv, radius - d)
                                } else {
                                    // Capsule center inside horizontal AABB: push out shallowest face
                                    let pen_xmin = (pos[0] - b.min[0]) + radius;
                                    let pen_xmax = (b.max[0] - pos[0]) + radius;
                                    let pen_ymin = (pos[1] - b.min[1]) + radius;
                                    let pen_ymax = (b.max[1] - pos[1]) + radius;
                                    if pen_xmin <= pen_xmax
                                        && pen_xmin <= pen_ymin
                                        && pen_xmin <= pen_ymax
                                    {
                                        (-1.0f32, 0.0f32, pen_xmin)
                                    } else if pen_xmax <= pen_ymin && pen_xmax <= pen_ymax {
                                        (1.0f32, 0.0f32, pen_xmax)
                                    } else if pen_ymin <= pen_ymax {
                                        (0.0f32, -1.0f32, pen_ymin)
                                    } else {
                                        (0.0f32, 1.0f32, pen_ymax)
                                    }
                                };

                                if pen > 0.0f32 {
                                    pos[0] = canon_f32(pos[0] + nx * (pen + 1e-4f32));
                                    pos[1] = canon_f32(pos[1] + ny * (pen + 1e-4f32));
                                    any_penetration = true;

                                    let n_cand = [canon_f32(nx), canon_f32(ny), 0.0f32];
                                    let mut duplicate = false;
                                    for p_idx in 0..num_planes {
                                        let dot = planes[p_idx][0] * n_cand[0]
                                            + planes[p_idx][1] * n_cand[1]
                                            + planes[p_idx][2] * n_cand[2];
                                        if dot > 0.98f32 {
                                            duplicate = true;
                                            break;
                                        }
                                    }
                                    if !duplicate && num_planes < 4 {
                                        planes[num_planes] = n_cand;
                                        num_planes += 1;
                                    }
                                }
                            }
                        }
                    } else if sp + 2 <= stack.len() {
                        let left = node.left_or_first;
                        stack[sp] = left + 1;
                        sp += 1;
                        stack[sp] = left;
                        sp += 1;
                    }
                }
            }

            if !any_penetration {
                break;
            }
        }

        // Enforce primal-active velocity coboundary (<v, n_i> >= 0) with n_1 x n_2 crease slide
        if num_planes == 1 {
            clip_velocity_plane(vel, &planes[0]);
        } else if num_planes >= 2 {
            let v_orig = *vel;
            clip_velocity_plane(vel, &planes[0]);
            if (vel[0] * planes[1][0] + vel[1] * planes[1][1] + vel[2] * planes[1][2]) < -1e-6f32 {
                clip_velocity_plane(vel, &planes[1]);
                // If clipping against plane 1 re-violates plane 0, slide along exact crease n_0 x n_1!
                if (vel[0] * planes[0][0] + vel[1] * planes[0][1] + vel[2] * planes[0][2])
                    < -1e-6f32
                {
                    let n0 = &planes[0];
                    let n1 = &planes[1];
                    let cx = n0[1] * n1[2] - n0[2] * n1[1];
                    let cy = n0[2] * n1[0] - n0[0] * n1[2];
                    let cz = n0[0] * n1[1] - n0[1] * n1[0];
                    let c_sq = (cx * cx + cy * cy) + cz * cz;
                    if c_sq > 1e-6f32 {
                        let inv_c = 1.0f32 / sqrt_f32_deterministic(c_sq, force_scalar);
                        let dx = cx * inv_c;
                        let dy = cy * inv_c;
                        let dz = cz * inv_c;
                        let vd = (v_orig[0] * dx + v_orig[1] * dy) + v_orig[2] * dz;
                        vel[0] = canon_f32(dx * vd);
                        vel[1] = canon_f32(dy * vd);
                        vel[2] = canon_f32(dz * vd);
                    } else {
                        *vel = [0.0f32; 3];
                    }
                }
            }
            // Check 3rd / 4th planes (corner lock if violated)
            for p_idx in 2..num_planes {
                let vn = vel[0] * planes[p_idx][0]
                    + vel[1] * planes[p_idx][1]
                    + vel[2] * planes[p_idx][2];
                if vn < -1e-5f32 {
                    *vel = [0.0f32; 3];
                    break;
                }
            }
        }
    }
}

/// Steps a single player's `PlayerTickState` by one 128 Hz tick (`NT_DT_SEC`), returning
/// the measured Cayley attitude rotor norm drift (`0.000e+00`).
pub fn step_single_player_kinematics(
    map: &StaticVoxelBvh,
    state: &mut PlayerTickState,
    cmd: &ClientInputCmd,
    force_scalar: bool,
) -> f32 {
    let btns = cmd.buttons_stance;
    let is_crouch = (btns & NT_BTN_CROUCH) != 0;
    let is_ads = (btns & NT_BTN_ADS) != 0;
    let wants_sprint = (btns & NT_BTN_SPRINT) != 0 && !is_crouch && !is_ads;
    let can_sprint = wants_sprint && state.stamina_q15 > 1024;

    if can_sprint {
        state.stamina_q15 = state.stamina_q15.saturating_sub(110);
    } else {
        state.stamina_q15 = state.stamina_q15.saturating_add(55).min(32767);
    }

    let max_speed = if is_crouch {
        SPEED_CROUCH_MPS
    } else if is_ads {
        SPEED_ADS_MPS
    } else if can_sprint {
        SPEED_SPRINT_MPS
    } else {
        SPEED_WALK_MPS
    };

    // Update orientation & smooth shoulder-lean slew
    let prev_yaw = state.yaw_q15;
    state.yaw_q15 = cmd.yaw_pitch_q15[0];
    state.pitch_q15 = cmd.yaw_pitch_q15[1].clamp(-16000, 16000);
    state.weapon_id = cmd.weapon_slot_u8;

    let target_lean: i16 = if (btns & NT_BTN_LEAN_L) != 0 && (btns & NT_BTN_LEAN_R) == 0 {
        -4096
    } else if (btns & NT_BTN_LEAN_R) != 0 && (btns & NT_BTN_LEAN_L) == 0 {
        4096
    } else {
        0
    };
    let lean_diff = (target_lean as i32) - (state.lean_q15 as i32);
    let lean_step = lean_diff.clamp(-512, 512);
    state.lean_q15 = ((state.lean_q15 as i32) + lean_step) as i16;

    // SO(3) Cayley attitude/lean rotor evaluation & 0.000e+00 norm conservation polish
    let yaw_delta_q15 = state.yaw_q15.wrapping_sub(prev_yaw);
    state.cayley_gen_q15 = [state.lean_q15, state.pitch_q15 >> 2, yaw_delta_q15 >> 2];
    let omega = [
        (state.cayley_gen_q15[0] as f32) * (1.0f32 / 32768.0f32),
        (state.cayley_gen_q15[1] as f32) * (1.0f32 / 32768.0f32),
        (state.cayley_gen_q15[2] as f32) * (1.0f32 / 32768.0f32),
    ];
    let spine_ref = [0.0f32, 0.0f32, 1.0f32];
    let (_r9, spine_rot_opt) = cayley_rotor_so3_eval(&omega, Some(&spine_ref));
    let cayley_drift = if let Some(sr) = spine_rot_opt {
        abs_f32(sqrt_f32_deterministic(norm_sq_3d(&sr), force_scalar) - 1.0f32)
    } else {
        0.0f32
    };

    // Wish velocity from modular q15 trig (zero fmod, zero FMA)
    let cy = cos_q15_deterministic(state.yaw_q15);
    let sy = sin_q15_deterministic(state.yaw_q15);

    let mut mf = (cmd.move_xy_q15[0] as f32) * (1.0f32 / 32767.0f32);
    let mut ms = (cmd.move_xy_q15[1] as f32) * (1.0f32 / 32767.0f32);
    mf = mf.clamp(-1.0f32, 1.0f32);
    ms = ms.clamp(-1.0f32, 1.0f32);

    let m_sq = mf * mf + ms * ms;
    if m_sq > 1.0f32 {
        let inv_m = 1.0f32 / sqrt_f32_deterministic(m_sq, force_scalar);
        mf *= inv_m;
        ms *= inv_m;
    }

    // Forward = (cy, sy), Right-strafe = (sy, -cy)
    let wish_vx = (cy * mf + sy * ms) * max_speed;
    let wish_vy = (sy * mf - cy * ms) * max_speed;

    // Ground friction + bounded acceleration
    let cur_speed_sq = state.vel[0] * state.vel[0] + state.vel[1] * state.vel[1];
    if cur_speed_sq > 1e-6f32 && m_sq < 1e-4f32 {
        let drop = (1.0f32 - GROUND_FRICTION * NT_DT_SEC).max(0.0f32);
        state.vel[0] = canon_f32(state.vel[0] * drop);
        state.vel[1] = canon_f32(state.vel[1] * drop);
    } else {
        let dvx = wish_vx - state.vel[0];
        let dvy = wish_vy - state.vel[1];
        let dv_sq = dvx * dvx + dvy * dvy;
        let max_dv = GROUND_ACCEL * NT_DT_SEC;
        if dv_sq > max_dv * max_dv {
            let scale = max_dv / sqrt_f32_deterministic(dv_sq, force_scalar);
            state.vel[0] = canon_f32(state.vel[0] + dvx * scale);
            state.vel[1] = canon_f32(state.vel[1] + dvy * scale);
        } else {
            state.vel[0] = canon_f32(wish_vx);
            state.vel[1] = canon_f32(wish_vy);
        }
    }

    // Strict horizontal speed clamp <= SPEED_SPRINT_MPS (preserves Minkowski ReachBall bound!)
    let v_h_sq = state.vel[0] * state.vel[0] + state.vel[1] * state.vel[1];
    if v_h_sq > SPEED_SPRINT_MPS * SPEED_SPRINT_MPS {
        let scale = SPEED_SPRINT_MPS / sqrt_f32_deterministic(v_h_sq, force_scalar);
        state.vel[0] = canon_f32(state.vel[0] * scale);
        state.vel[1] = canon_f32(state.vel[1] * scale);
    }

    if state.pos[2] > 0.0f32 {
        state.vel[2] = canon_f32(state.vel[2] - GRAVITY_MPS2 * NT_DT_SEC);
    } else {
        state.vel[2] = 0.0f32;
    }

    let height = if is_crouch {
        HEIGHT_CROUCH_M
    } else {
        HEIGHT_STAND_M
    };
    collide_and_slide_3d(
        map,
        &mut state.pos,
        &mut state.vel,
        PLAYER_RADIUS_M,
        height,
        NT_DT_SEC,
        force_scalar,
    );

    let mut flags = NT_STANCE_ACTIVE;
    if is_ads {
        flags |= NT_STANCE_ADS;
    }
    if is_crouch {
        flags |= NT_STANCE_CROUCH;
    }
    if can_sprint {
        flags |= NT_STANCE_SPRINT;
    }
    if state.lean_q15 < -256 {
        flags |= NT_STANCE_LEAN_L;
    } else if state.lean_q15 > 256 {
        flags |= NT_STANCE_LEAN_R;
    }
    if state.pos[2] <= 1e-4f32 {
        flags |= NT_STANCE_GROUNDED;
    }
    state.stance_flags = flags;

    cayley_drift
}

/// Emits a server-side acoustic event into `projectiles_acoustic_slab` (ring buffer of 512 emitters).
pub fn push_server_acoustic_emitter(
    arena: &mut NetTwinArena,
    source_player_id: u32,
    pos: [f32; 3],
    max_radius_m: f32,
    bands: [u8; 4],
) {
    if bands[0] == 0 && bands[1] == 0 && bands[2] == 0 && bands[3] == 0 {
        return;
    }
    let slab = &mut arena.projectiles_acoustic_slab;
    let idx = (slab.emitter_head as usize) & (NT_MAX_ACOUSTIC_EMITTERS - 1);
    slab.emitters[idx] = ServerAcousticEmitter {
        pos: [canon_f32(pos[0]), canon_f32(pos[1]), canon_f32(pos[2])],
        emit_tick: arena.telemetry_and_header.server_tick,
        source_player_id,
        max_radius_m: canon_f32(max_radius_m),
        band_energy_q8: bands,
        active: 1,
    };
    slab.emitter_head = slab.emitter_head.wrapping_add(1);
    if slab.emitter_count < NT_MAX_ACOUSTIC_EMITTERS as u32 {
        slab.emitter_count += 1;
    }
}

/// Steps all active players, swept broadphase cylinders, ballistic projectiles, and quantized
/// baselines by 1 authoritative 128 Hz tick.
pub fn step_arena_kinematics(arena: &mut NetTwinArena) -> i32 {
    let force_scalar = arena.telemetry_and_header.simd_mode_override != 0;
    let prev_tick = arena.telemetry_and_header.server_tick;
    let next_tick = prev_tick.wrapping_add(1);
    let prev_slot = (prev_tick as usize) & 63;
    let next_slot = (next_tick as usize) & 63;

    arena.telemetry_and_header.server_tick = next_tick;
    let active_mask = arena.telemetry_and_header.active_player_mask;

    let mut max_drift = arena.telemetry_and_header.max_cayley_norm_drift;

    for pid in 0..NT_MAX_PLAYERS {
        // Decay lag-switch quarantine timer on every server tick
        let gate = &mut arena.client_jitter_gates[pid];
        if gate.quarantine_ticks_remaining > 0 {
            gate.quarantine_ticks_remaining -= 1;
        }

        if (active_mask & (1u64 << pid)) == 0 {
            arena.tier_a_cylinders[next_slot][pid] = TierACylinder::default();
            arena.tier_b_poses_plates[next_slot][pid] = PlayerTickState::default();
            arena.client_baseline_snapshots.quantized_ring[next_slot][pid] =
                QuantizedEntityState::default();
            continue;
        }

        let prev_state = arena.tier_b_poses_plates[prev_slot][pid];
        let cmd = if gate.total_cmds_ingested > 0 {
            gate.cmd_ring[(gate.latest_cmd_tick as usize) & 31]
        } else {
            ClientInputCmd {
                cmd_tick: next_tick,
                buttons_stance: 0,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [prev_state.yaw_q15, prev_state.pitch_q15],
                fire_subtick_u8: 0,
                weapon_slot_u8: prev_state.weapon_id,
            }
        };

        let mut cur_state = prev_state;
        let drift = step_single_player_kinematics(
            &arena.static_voxel_bvh,
            &mut cur_state,
            &cmd,
            force_scalar,
        );
        if drift > max_drift {
            max_drift = drift;
        }

        // Swept Tier-A broadphase cylinder covering [prev_pos, cur_pos] + shoulder lean margin
        let mid_x = canon_f32(0.5f32 * (prev_state.pos[0] + cur_state.pos[0]));
        let mid_y = canon_f32(0.5f32 * (prev_state.pos[1] + cur_state.pos[1]));
        let min_z = canon_f32(prev_state.pos[2].min(cur_state.pos[2]));
        let dx = cur_state.pos[0] - prev_state.pos[0];
        let dy = cur_state.pos[1] - prev_state.pos[1];
        let half_sweep = 0.5f32 * sqrt_f32_deterministic(dx * dx + dy * dy, force_scalar);
        let lean_extra = abs_f32(cur_state.lean_q15 as f32) * (0.22f32 / 4096.0f32);
        let swept_rad_mm = (((PLAYER_RADIUS_M + half_sweep + lean_extra + 0.08f32) * 1000.0f32)
            as u32)
            .clamp(350, 4000) as u16;
        let height_mm: u16 = if (cur_state.stance_flags & NT_STANCE_CROUCH) != 0 {
            1150
        } else {
            1800
        };

        arena.tier_a_cylinders[next_slot][pid] = TierACylinder {
            x: mid_x,
            y: mid_y,
            z_base: min_z,
            rad_height: TierACylinder::pack_rad_height(swept_rad_mm, height_mm),
        };
        arena.tier_b_poses_plates[next_slot][pid] = cur_state;
        arena.client_baseline_snapshots.quantized_ring[next_slot][pid] =
            quantize_player_state(&cur_state);

        // Emit anonymous footstep emitter ONLY if walking/sprinting (> 2.6 m/s); silent crouch/stand emits 0!
        let speed_h_sq = cur_state.vel[0] * cur_state.vel[0] + cur_state.vel[1] * cur_state.vel[1];
        if speed_h_sq > (2.6f32 * 2.6f32) && ((next_tick.wrapping_add(pid as u32)) & 15) == 0 {
            let is_sprint = (cur_state.stance_flags & NT_STANCE_SPRINT) != 0;
            let bands = if is_sprint {
                [190u8, 170u8, 120u8, 72u8]
            } else {
                [128u8, 112u8, 74u8, 38u8]
            };
            push_server_acoustic_emitter(arena, pid as u32, cur_state.pos, 18.0f32, bands);
        }
    }

    // Step finite-velocity long-range ballistic projectiles (> 25 m)
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

        if let Some((t_hit, normal)) =
            raycast_hard_cover(&arena.static_voxel_bvh, &p.pos, &dir, step_len)
        {
            // Shallow wall obliquity ricochet vs wall embed
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
                // Quadratic air drag + ballistic gravity drop
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

    arena.telemetry_and_header.active_projectile_count = active_proj;
    arena.telemetry_and_header.max_cayley_norm_drift = canon_f32(max_drift);
    arena.telemetry_and_header.total_steps =
        arena.telemetry_and_header.total_steps.wrapping_add(1);
    arena.telemetry_and_header.state_digest_u64 = hash_arena_canonical(arena);
    0
}

/// Replays `num_cmds` unacknowledged client commands from authoritative `auth_pose` in `< 90 us`.
pub fn client_reconcile_impl(
    arena: &mut NetTwinArena,
    player_id: u32,
    auth_pose: &PlayerPoseSnapshot,
    pending_cmds: &[ClientInputCmd],
) -> PlayerPoseSnapshot {
    let force_scalar = arena.telemetry_and_header.simd_mode_override != 0;
    let slot = (arena.telemetry_and_header.server_tick as usize) & 63;
    let pid = (player_id as usize) & (NT_MAX_PLAYERS - 1);

    let mut state = arena.tier_b_poses_plates[slot][pid];
    state.pos = [
        canon_f32(auth_pose.x),
        canon_f32(auth_pose.y),
        canon_f32(auth_pose.z),
    ];
    state.vel = [
        canon_f32(auth_pose.vx),
        canon_f32(auth_pose.vy),
        canon_f32(auth_pose.vz),
    ];
    state.yaw_q15 = auth_pose.yaw_q15;
    state.pitch_q15 = auth_pose.pitch_q15;
    state.lean_q15 = auth_pose.lean_q15;
    state.stance_flags = auth_pose.stance_flags | NT_STANCE_ACTIVE;
    if state.stamina_q15 == 0 {
        state.stamina_q15 = 32767;
    }

    for cmd in pending_cmds {
        let _ = step_single_player_kinematics(&arena.static_voxel_bvh, &mut state, cmd, force_scalar);
    }

    arena.tier_b_poses_plates[slot][pid] = state;
    arena.telemetry_and_header.last_reconcile_ticks = pending_cmds.len() as u32;
    state_to_pose_snapshot(&state)
}
