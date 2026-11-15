//! Layer 3a: Circular 32-bit Tick Window (`ZMod 2^32`), 3-Point Discrete Morse
//! Arrival-Jitter Gate, Acklam `\Phi^{-1}` Pullback, and `H_clock` Lag-Switch Quarantine.

use crate::arena::{
    ClientInputCmd, NetTwinArena, NT_GATE_GLINT_REJECTED,
    NT_GATE_LAG_SWITCH_QUARANTINED, NT_GATE_NOMINAL, NT_GATE_PENDING_IMPULSE,
    NT_GATE_STEP_MANEUVER, NT_MAX_PLAYERS,
};
use crate::math::{abs_f32, canon_f32, inv_normal_cdf_f32};

/// Modular 32-bit half-space ordering: returns `true` iff `0 < (t_new - t_old) mod 2^32 < 2^31`.
#[inline(always)]
pub const fn circular_tick_is_newer(t_new: u32, t_old: u32) -> bool {
    let diff = t_new.wrapping_sub(t_old);
    diff > 0 && diff < 0x8000_0000
}

/// Returns `Some(now_tick - past_tick)` in `[0, 2^31 - 1]` if `past_tick` is at or before `now_tick`
/// in the modular 32-bit window, else `None`.
#[inline(always)]
pub const fn circular_tick_age(now_tick: u32, past_tick: u32) -> Option<u32> {
    if now_tick == past_tick {
        return Some(0);
    }
    let diff = now_tick.wrapping_sub(past_tick);
    if diff < 0x8000_0000 {
        Some(diff)
    } else {
        None
    }
}

/// Computes the innovation-gated rewind horizon `w_eff`:
/// `w_eff(rtt_ticks, i_morse, w_max) = min(w_max, max(0, rtt_ticks + margin - kappa * max(0, i_morse)))`.
#[inline]
pub fn compute_effective_rewind_ticks(
    rtt_ticks: u32,
    i_morse: f32,
    quarantine_active: bool,
    w_max: u32,
) -> u32 {
    if quarantine_active {
        return 2u32.min(w_max);
    }
    let base = (rtt_ticks + 4).min(w_max) as f32;
    let pos_morse = if i_morse > 0.0f32 { i_morse } else { 0.0f32 };
    // kappa = 0.25 ticks per unit Morse impulse
    let penalized = (base - 0.25f32 * pos_morse).max(0.0f32);
    (penalized as u32).min(w_max)
}

/// Ingests a `ClientInputCmd` for `player_id`, updating the 64-tick `ack_mask_u64` ring,
/// evaluating the 3-point discrete Morse arrival-skew gate + Acklam `\Phi^{-1}` pullback,
/// and detecting `H_clock` periodic lag-switch (`clumsy`) micro-burst packet clumping.
///
/// Returns the updated gate status (`NT_GATE_*`, `0..=4`) or `-1` on invalid arguments.
pub fn ingest_client_cmd_impl(
    arena: &mut NetTwinArena,
    player_id: u32,
    cmd: &ClientInputCmd,
) -> i32 {
    let pid = player_id as usize;
    if pid >= NT_MAX_PLAYERS {
        return -1;
    }

    let server_tick = arena.telemetry_and_header.server_tick;
    let g = &mut arena.client_jitter_gates[pid];

    // Store command in 32-slot client command ring
    g.cmd_ring[(cmd.cmd_tick as usize) & 31] = *cmd;

    let one_way_transit = match circular_tick_age(server_tick, cmd.cmd_tick) {
        Some(age) if age <= 512 => age,
        _ => g.est_rtt_ticks / 2,
    };

    // First command bootstrap for this player
    if g.total_cmds_ingested == 0 {
        g.latest_cmd_tick = cmd.cmd_tick;
        g.last_arrival_server_tick = server_tick;
        g.ack_mask_u64 = 1u64;
        if one_way_transit > 0 {
            g.est_rtt_ticks = (one_way_transit * 2).clamp(2, 64);
        } else if g.est_rtt_ticks == 0 {
            g.est_rtt_ticks = 6;
        }
        let j0 = canon_f32(one_way_transit as f32);
        g.morse_j_prev2 = j0;
        g.morse_j_prev1 = j0;
        g.morse_j_curr = j0;
        g.morse_pending_j = j0;
        g.morse_impulse_last = 0.0f32;
        g.jitter_var_ema = 1.0f32;
        g.acklam_pullback_z = 1.0f32;
        g.h_clock_burst_score = 0.0f32;
        g.gate_status = NT_GATE_NOMINAL;
        g.clamped_rewind_max_ticks =
            compute_effective_rewind_ticks(g.est_rtt_ticks, 0.0f32, false, 32);
        g.arrival_skew_hist[0] = 0.0f32;
        g.hist_head = 1;
        g.total_cmds_ingested = 1;
        return NT_GATE_NOMINAL as i32;
    }

    // Compute inter-arrival vs command-tick deltas BEFORE updating latest_cmd_tick
    let dt_srv = server_tick.wrapping_sub(g.last_arrival_server_tick);
    let dt_cmd = if circular_tick_is_newer(cmd.cmd_tick, g.latest_cmd_tick) {
        cmd.cmd_tick.wrapping_sub(g.latest_cmd_tick)
    } else {
        0u32
    };

    // Update 64-tick circular ack_mask_u64 window
    if circular_tick_is_newer(cmd.cmd_tick, g.latest_cmd_tick) {
        let shift = cmd.cmd_tick.wrapping_sub(g.latest_cmd_tick);
        g.ack_mask_u64 = if shift >= 64 {
            1u64
        } else {
            (g.ack_mask_u64 << shift) | 1u64
        };
        g.latest_cmd_tick = cmd.cmd_tick;
    } else if let Some(back) = circular_tick_age(g.latest_cmd_tick, cmd.cmd_tick) {
        if back < 64 {
            g.ack_mask_u64 |= 1u64 << back;
        }
    }
    g.last_arrival_server_tick = server_tick;
    g.total_cmds_ingested = g.total_cmds_ingested.wrapping_add(1);

    // Arrival-skew observation j_obs (tracks one-way transit ticks in float32)
    let j_obs = canon_f32(one_way_transit as f32);
    let inter_skew = canon_f32((dt_srv as f32) - (dt_cmd as f32));

    let h_idx = (g.hist_head as usize) & 63;
    g.arrival_skew_hist[h_idx] = inter_skew;
    g.hist_head = g.hist_head.wrapping_add(1);

    // Acklam \Phi^{-1} dynamic pullback over inter-arrival interval (tau_ref = 128 ticks = 1.0 s)
    let dt_ratio = ((dt_srv.max(1) as f32) * (1.0f32 / 128.0f32)).clamp(0.001f32, 0.999f32);
    let phi_inv = inv_normal_cdf_f32(1.0f32 - 0.5f32 * dt_ratio).max(0.5f32);
    g.acklam_pullback_z = phi_inv;

    // H_clock periodic lag-switch (`clumsy`) micro-burst clumping detector:
    // Clumped burst packets arrive on the SAME server_tick (dt_srv == 0 && dt_cmd >= 1)
    // or jump > 10 ticks (~78 ms) above established baseline one-way transit.
    let baseline_one_way = (g.est_rtt_ticks as f32) * 0.5f32;
    let transit_excess = j_obs - baseline_one_way;

    if dt_srv == 0 && dt_cmd >= 1 {
        g.total_burst_clumps = g.total_burst_clumps.wrapping_add(1);
        g.h_clock_burst_score = canon_f32(g.h_clock_burst_score + 1.15f32);
    } else if transit_excess >= 10.0f32 || abs_f32(inter_skew) >= 10.0f32 {
        g.total_burst_clumps = g.total_burst_clumps.wrapping_add(1);
        g.h_clock_burst_score = canon_f32(g.h_clock_burst_score + 1.85f32);
    } else {
        g.h_clock_burst_score = canon_f32(g.h_clock_burst_score * 0.88f32);
    }

    // 3-Point Discrete Morse Innovation Gate with 1-Sample Causal Deferral:
    // I_Morse(j) = (j_t - j_{t-1}) * (j_t - j_{t+1})
    let mut new_status: u32;
    if g.gate_status == NT_GATE_PENDING_IMPULSE {
        let j_tm1 = g.morse_j_curr;
        let j_t = g.morse_pending_j;
        let j_tp1 = j_obs;

        let i_morse = canon_f32((j_t - j_tm1) * (j_t - j_tp1));
        g.morse_impulse_last = i_morse;

        let return_diff = abs_f32(j_tp1 - j_tm1);
        if i_morse > 0.0f32 && return_diff <= 2.5f32 {
            // Isolated single-packet jitter spike / glint
            new_status = NT_GATE_GLINT_REJECTED;
            g.morse_j_prev2 = j_tm1;
            g.morse_j_prev1 = j_t;
            g.morse_j_curr = j_tp1;
        } else {
            // Sustained step shift or burst plateau
            new_status = NT_GATE_STEP_MANEUVER;
            g.morse_j_prev2 = j_tm1;
            g.morse_j_prev1 = j_t;
            g.morse_j_curr = j_tp1;
            if i_morse > 24.0f32 {
                g.h_clock_burst_score = canon_f32(g.h_clock_burst_score + 1.5f32);
            }
        }
    } else {
        let diff = abs_f32(j_obs - g.morse_j_curr);
        let var_safe = g.jitter_var_ema.max(0.25f32);
        let nis = canon_f32((diff * diff) / var_safe);
        let dynamic_thresh = 6.5f32 * phi_inv;

        // Also compute immediate running 3-point Morse over (prev1, curr, obs)
        let running_morse = canon_f32((g.morse_j_curr - g.morse_j_prev1) * (g.morse_j_curr - j_obs));
        g.morse_impulse_last = running_morse;

        if nis > dynamic_thresh && diff >= 3.0f32 {
            new_status = NT_GATE_PENDING_IMPULSE;
            g.morse_pending_j = j_obs;
        } else {
            new_status = NT_GATE_NOMINAL;
            g.morse_j_prev2 = g.morse_j_prev1;
            g.morse_j_prev1 = g.morse_j_curr;
            g.morse_j_curr = j_obs;
            let d2 = (diff * diff).clamp(0.05f32, 16.0f32);
            g.jitter_var_ema = canon_f32(0.92f32 * g.jitter_var_ema + 0.08f32 * d2);

            // Slowly adapt honest RTT estimate ONLY on nominal non-burst packets
            if g.h_clock_burst_score < 1.0f32 && g.quarantine_ticks_remaining == 0 && one_way_transit > 0 {
                let target_rtt = (one_way_transit * 2).clamp(2, 64);
                if target_rtt > g.est_rtt_ticks {
                    g.est_rtt_ticks += 1;
                } else if target_rtt < g.est_rtt_ticks {
                    g.est_rtt_ticks -= 1;
                }
            }
        }
    }

    // Trigger H_clock lag-switch quarantine when burst clumping score crosses threshold
    if g.h_clock_burst_score >= 2.5f32 {
        if g.quarantine_ticks_remaining == 0 {
            g.total_quarantine_triggers = g.total_quarantine_triggers.wrapping_add(1);
            arena.telemetry_and_header.total_jitter_quarantines =
                arena.telemetry_and_header.total_jitter_quarantines.wrapping_add(1);
        }
        g.quarantine_ticks_remaining = 64; // Quarantine rewind horizon for 500 ms (64 ticks)
    }

    let quarantine_active = g.quarantine_ticks_remaining > 0;
    if quarantine_active {
        new_status = NT_GATE_LAG_SWITCH_QUARANTINED;
    }
    g.gate_status = new_status;
    g.clamped_rewind_max_ticks = compute_effective_rewind_ticks(
        g.est_rtt_ticks,
        g.morse_impulse_last,
        quarantine_active,
        32,
    );

    new_status as i32
}
