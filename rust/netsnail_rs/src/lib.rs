//! `netsnail_rs`: Allocation-free `#![no_std]` deterministic 128 Hz multiplayer causal twin,
//! `320 KB` sub-tick 30-plate rewind kernel, 3-point discrete Morse arrival-jitter gate,
//! and Minkowski reachability PVS anti-DMA serializer.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod arena;
pub mod ballistics;
pub mod jitter_morse;
pub mod kinematics;
pub mod map_bvh;
pub mod math;
pub mod replication;

use core::cell::UnsafeCell;
use core::sync::atomic::Ordering;

pub use arena::{
    AcousticImpulse, ClientInputCmd, ClientJitterGate, HitRecordOut, NetTwinArena,
    PlayerPoseSnapshot, PlayerTickState, TelemetryAndHeader, TierACylinder, NT_ARENA_ALIGN,
    NT_ARENA_BYTES, NT_MAX_PLAYERS, NT_MTU_CEILING_BYTES, NT_POSE_HAS_STAMINA, NT_RING_TICKS,
    NT_STANCE_MASK_7BIT, NT_TICK_HZ, NT_WASM_MTU_BUF_BYTES,
};

use arena::{hash_arena_canonical, init_arena_in_place, QuantizedEntityState, NT_STANCE_ACTIVE, NT_STANCE_GROUNDED};
use ballistics::{default_plate_material_bits_u64, resolve_subtick_fire_impl, DEFAULT_INTACT_PLATE_MASK};
use jitter_morse::ingest_client_cmd_impl;
use kinematics::{
    client_reconcile_impl, push_server_acoustic_emitter, quantize_player_state,
    state_to_pose_snapshot, step_arena_kinematics,
};
use map_bvh::populate_default_raid_map;
use math::{
    atan2_f32_deterministic, canon_f32, cayley_rotor_so3_eval, cos_q15_deterministic,
    dot16_pairwise_dispatch, sin_q15_deterministic, sqrt_f32_deterministic, MxcsrGuard,
    CANONICAL_NAN_BITS, GLOBAL_SIMD_OVERRIDE,
};
use replication::{pack_client_datagram_impl, unpack_client_datagram_impl};

// ============================================================================
// Zero-Allocation #![no_std] Panic Handler
// ============================================================================

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::unreachable()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        loop {
            core::hint::spin_loop();
        }
    }
}

#[cfg(all(
    not(feature = "std"),
    not(test),
    not(windows),
    not(target_arch = "wasm32")
))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

// ============================================================================
// Zero-Initialized (.bss) 64-Byte Aligned Static Bridge Buffers
// ============================================================================

#[repr(C, align(64))]
struct AlignedBssBuf<const N: usize>(UnsafeCell<[u8; N]>);

unsafe impl<const N: usize> Sync for AlignedBssBuf<N> {}

static GLOBAL_SERVER_ARENA_BSS: AlignedBssBuf<NT_ARENA_BYTES> =
    AlignedBssBuf(UnsafeCell::new([0u8; NT_ARENA_BYTES]));

static GLOBAL_CLIENT_ARENA_BSS: AlignedBssBuf<NT_ARENA_BYTES> =
    AlignedBssBuf(UnsafeCell::new([0u8; NT_ARENA_BYTES]));

static GLOBAL_CMD_BUF_BSS: AlignedBssBuf<{ 64 * core::mem::size_of::<ClientInputCmd>() }> =
    AlignedBssBuf(UnsafeCell::new(
        [0u8; 64 * core::mem::size_of::<ClientInputCmd>()],
    ));

static GLOBAL_MTU_BUF_BSS: AlignedBssBuf<NT_WASM_MTU_BUF_BYTES> =
    AlignedBssBuf(UnsafeCell::new([0u8; NT_WASM_MTU_BUF_BYTES]));

static GLOBAL_HIT_BUF_BSS: AlignedBssBuf<{ 64 * core::mem::size_of::<HitRecordOut>() }> =
    AlignedBssBuf(UnsafeCell::new(
        [0u8; 64 * core::mem::size_of::<HitRecordOut>()],
    ));

static GLOBAL_ACOUSTIC_BUF_BSS: AlignedBssBuf<{ 64 * core::mem::size_of::<AcousticImpulse>() }> =
    AlignedBssBuf(UnsafeCell::new(
        [0u8; 64 * core::mem::size_of::<AcousticImpulse>()],
    ));

// ============================================================================
// Exported C-ABI Entry Points (Native & wasm32-unknown-unknown)
// ============================================================================

/// Zero-initializes a `1,572,864`-byte `NetTwinArena` and sets canonical defaults.
///
/// # Safety
/// `arena` must point to a writable, 64-byte-aligned `NetTwinArena` allocation (or null).
#[no_mangle]
pub unsafe extern "C" fn nt_arena_init(arena: *mut NetTwinArena) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() {
        return -1;
    }
    init_arena_in_place(arena);
    0
}

/// Sets the SIMD execution backend mode (`0 = auto SIMD`, `1 = forced portable scalar`).
///
/// When `arena` is non-null, sets ONLY `(*arena).telemetry_and_header.simd_mode_override`
/// without mutating `GLOBAL_SIMD_OVERRIDE` (guaranteeing per-arena / thread-parallel isolation).
/// When `arena` is null, updates `GLOBAL_SIMD_OVERRIDE` for standalone Tier-0 primitives.
///
/// # Safety
/// If non-null, `arena` must point to a valid `NetTwinArena`.
#[no_mangle]
pub unsafe extern "C" fn nt_set_simd_backend_override(
    arena: *mut NetTwinArena,
    mode: u32,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    let norm_mode = if mode != 0 { 1u32 } else { 0u32 };
    if arena.is_null() {
        GLOBAL_SIMD_OVERRIDE.store(norm_mode, Ordering::Relaxed);
        return 0;
    }
    (*arena).telemetry_and_header.simd_mode_override = norm_mode;
    0
}

/// Populates `static_voxel_bvh` (`128x128x16` 1-bit grid + AABB brushes + static BVH + portals)
/// with the default tactical multi-room raid hallway and central concrete pillar.
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena` (or null).
#[no_mangle]
pub unsafe extern "C" fn nt_load_default_raid_map(arena: *mut NetTwinArena) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() {
        return -1;
    }
    let a = &mut *arena;
    populate_default_raid_map(&mut a.static_voxel_bvh);
    a.telemetry_and_header.state_digest_u64 = hash_arena_canonical(a);
    0
}

/// Spawns `player_id` (`0..63`) at `(x, y, z)` with `yaw_q15` and full 30-zone multi-plate armor.
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena` (or null).
#[no_mangle]
pub unsafe extern "C" fn nt_spawn_player(
    arena: *mut NetTwinArena,
    player_id: u32,
    x: f32,
    y: f32,
    z: f32,
    yaw_q15: i16,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || (player_id as usize) >= NT_MAX_PLAYERS {
        return -1;
    }
    let a = &mut *arena;
    let pid = player_id as usize;
    let cur_slot = (a.telemetry_and_header.server_tick as usize) & 63;

    let clean_x = canon_f32(x);
    let clean_y = canon_f32(y);
    let clean_z = canon_f32(z.max(0.0f32));

    let pstate = PlayerTickState {
        pos: [clean_x, clean_y, clean_z],
        vel: [0.0f32; 3],
        yaw_q15,
        pitch_q15: 0,
        lean_q15: 0,
        stance_flags: NT_STANCE_ACTIVE | NT_STANCE_GROUNDED,
        cayley_gen_q15: [0; 3],
        stamina_q15: 32767,
        health_q8: 25600, // 100.0 HP in Q8.8
        team_id: (pid & 1) as u8,
        weapon_id: 0,
        plate_intact_mask: DEFAULT_INTACT_PLATE_MASK,
        plate_material_bits: default_plate_material_bits_u64(),
        plate_hp_pools_q8: [12800, 20480, 17920, 10240], // Helmet 50, Front 80, Back/Side 70, Limb 40
    };
    let cyl = TierACylinder {
        x: clean_x,
        y: clean_y,
        z_base: clean_z,
        rad_height: TierACylinder::pack_rad_height(350, 1800),
    };
    let qstate: QuantizedEntityState = quantize_player_state(&pstate);

    a.tier_b_poses_plates[cur_slot][pid] = pstate;
    a.tier_a_cylinders[cur_slot][pid] = cyl;
    a.client_baseline_snapshots.quantized_ring[cur_slot][pid] = qstate;
    a.telemetry_and_header.active_player_mask |= 1u64 << pid;
    a.telemetry_and_header.state_digest_u64 = hash_arena_canonical(a);
    0
}

/// Reads the authoritative current `PlayerPoseSnapshot` (`32 B`) for `player_id` (`0..63`).
///
/// # Safety
/// `arena` and `out_pose` must be valid pointers (or null).
#[no_mangle]
pub unsafe extern "C" fn nt_get_player_pose(
    arena: *const NetTwinArena,
    player_id: u32,
    out_pose: *mut PlayerPoseSnapshot,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || out_pose.is_null() || (player_id as usize) >= NT_MAX_PLAYERS {
        return -1;
    }
    let a = &*arena;
    let pid = player_id as usize;
    let cur_slot = (a.telemetry_and_header.server_tick as usize) & 63;
    if (a.telemetry_and_header.active_player_mask & (1u64 << pid)) == 0 {
        *out_pose = PlayerPoseSnapshot::default();
        return 0;
    }
    *out_pose = state_to_pose_snapshot(&a.tier_b_poses_plates[cur_slot][pid]);
    0
}

/// Explicitly sets estimated RTT in 128 Hz ticks for `player_id` (`0..63`).
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena` (or null).
#[no_mangle]
pub unsafe extern "C" fn nt_set_player_rtt_ticks(
    arena: *mut NetTwinArena,
    player_id: u32,
    rtt_ticks: u32,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || (player_id as usize) >= NT_MAX_PLAYERS {
        return -1;
    }
    let a = &mut *arena;
    let pid = player_id as usize;
    let clamped = rtt_ticks.clamp(1, 64);
    let g = &mut a.client_jitter_gates[pid];
    g.est_rtt_ticks = clamped;
    g.clamped_rewind_max_ticks = jitter_morse::compute_effective_rewind_ticks(
        clamped,
        g.morse_impulse_last,
        g.quarantine_ticks_remaining > 0,
        32,
    );
    0
}

/// Emits a server-side 4-band acoustic event for portal-diffracted anonymous delivery.
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena`; `band_energy_q8` must point to 4 bytes.
#[no_mangle]
pub unsafe extern "C" fn nt_emit_acoustic_event(
    arena: *mut NetTwinArena,
    source_player_id: u32,
    x: f32,
    y: f32,
    z: f32,
    band_energy_q8: *const u8,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || band_energy_q8.is_null() || (source_player_id as usize) >= NT_MAX_PLAYERS {
        return -1;
    }
    let bands = [
        *band_energy_q8,
        *band_energy_q8.add(1),
        *band_energy_q8.add(2),
        *band_energy_q8.add(3),
    ];
    push_server_acoustic_emitter(&mut *arena, source_player_id, [x, y, z], 24.0f32, bands);
    0
}

/// Ingests a `ClientInputCmd` (`16 B`) through the 3-point discrete Morse arrival-jitter gate,
/// Acklam `\Phi^{-1}` pullback, and `H_clock` lag-switch quarantine detector.
///
/// # Safety
/// `arena` and `cmd` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nt_ingest_client_cmd(
    arena: *mut NetTwinArena,
    player_id: u32,
    cmd: *const ClientInputCmd,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || cmd.is_null() {
        return -1;
    }
    ingest_client_cmd_impl(&mut *arena, player_id, &*cmd)
}

/// Advances the authoritative causal twin by one 128 Hz tick (`dt = 1/128 s`).
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena`.
#[no_mangle]
pub unsafe extern "C" fn nt_step_kinematics(arena: *mut NetTwinArena) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() {
        return -1;
    }
    step_arena_kinematics(&mut *arena)
}

/// Reconciles local client prediction from `auth_pose` across `num_cmds` unacknowledged inputs.
///
/// # Safety
/// `arena` and `auth_pose` must be non-null; if `num_cmds > 0`, `pending_cmds` must point to at
/// least `num_cmds` `ClientInputCmd` elements.
#[no_mangle]
pub unsafe extern "C" fn nt_client_reconcile(
    arena: *mut NetTwinArena,
    player_id: u32,
    auth_pose: *const PlayerPoseSnapshot,
    pending_cmds: *const ClientInputCmd,
    num_cmds: u32,
    out_predicted_pose: *mut PlayerPoseSnapshot,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || auth_pose.is_null() || (player_id as usize) >= NT_MAX_PLAYERS {
        return -1;
    }
    if num_cmds > 0 && pending_cmds.is_null() {
        return -1;
    }
    let n = (num_cmds as usize).min(256);
    let cmds = if n > 0 {
        core::slice::from_raw_parts(pending_cmds, n)
    } else {
        &[]
    };
    let pred = client_reconcile_impl(&mut *arena, player_id, &*auth_pose, cmds);
    if !out_predicted_pose.is_null() {
        *out_predicted_pose = pred;
    }
    0
}

/// Resolves an authoritative sub-tick shot over the `320 KB` L2 rewind ring with 30-zone plate
/// obliquity ricochet/penetration and Mutual Occluder Horizon Cutoff.
///
/// # Safety
/// `arena`, `fire_cmd`, and `out_hit` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nt_resolve_subtick_fire(
    arena: *mut NetTwinArena,
    shooter_id: u32,
    fire_cmd: *const ClientInputCmd,
    out_hit: *mut HitRecordOut,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null()
        || fire_cmd.is_null()
        || out_hit.is_null()
        || (shooter_id as usize) >= NT_MAX_PLAYERS
    {
        return -1;
    }
    let rec = resolve_subtick_fire_impl(&mut *arena, shooter_id, &*fire_cmd);
    *out_hit = rec;
    0
}

/// Serializes an observer-filtered `< 1200 B` UDP delta datagram enforcing Minkowski Reachability
/// PVS and strict byte-for-byte non-interference outside `R_j(dt)`.
///
/// # Safety
/// `arena` and `out_mtu_buf` must be valid pointers; `out_mtu_buf` must have `max_buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn nt_pack_client_datagram(
    arena: *mut NetTwinArena,
    observer_id: u32,
    baseline_ack_tick: u32,
    out_mtu_buf: *mut u8,
    max_buf_len: u32,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() || out_mtu_buf.is_null() || max_buf_len < 16 {
        return -1;
    }
    let slice = core::slice::from_raw_parts_mut(out_mtu_buf, max_buf_len as usize);
    pack_client_datagram_impl(&mut *arena, observer_id, baseline_ack_tick, slice)
}

/// Unpacks an incoming UDP datagram into `client_arena`, scrubbing all out-of-cone enemy slots
/// to `0x00` and extracting optional anonymous `AcousticImpulse` structs.
///
/// # Safety
/// `client_arena` and `in_mtu_buf` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn nt_unpack_client_datagram(
    client_arena: *mut NetTwinArena,
    in_mtu_buf: *const u8,
    buf_len: u32,
    out_acoustics: *mut AcousticImpulse,
    max_acoustics: u32,
    out_num_acoustics: *mut u32,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if client_arena.is_null() || in_mtu_buf.is_null() || buf_len < 16 {
        return -1;
    }
    let pkt = core::slice::from_raw_parts(in_mtu_buf, buf_len as usize);
    let ac_opt = if !out_acoustics.is_null() && max_acoustics > 0 {
        Some(core::slice::from_raw_parts_mut(
            out_acoustics,
            max_acoustics as usize,
        ))
    } else {
        None
    };
    let (vis_rc, num_ac) = unpack_client_datagram_impl(&mut *client_arena, pkt, ac_opt);
    if !out_num_acoustics.is_null() {
        *out_num_acoustics = num_ac;
    }
    vis_rc
}

/// Computes the canonical 64-bit deterministic digest of `arena` (`0-ULP` identical across
/// `x86_64 SIMD`, `Scalar`, and `WASM32`).
///
/// # Safety
/// `arena` must point to a valid `NetTwinArena` (returns `0` if null).
#[no_mangle]
pub unsafe extern "C" fn nt_hash_arena_deterministic(arena: *const NetTwinArena) -> u64 {
    let _mxcsr = MxcsrGuard::enter();
    if arena.is_null() {
        return 0;
    }
    hash_arena_canonical(&*arena)
}

// ============================================================================
// Tier-0 Deterministic Math & SIMD C-ABI Exports
// ============================================================================

#[no_mangle]
pub extern "C" fn nt_sin_q15(phase_q15: i16) -> f32 {
    let _mxcsr = MxcsrGuard::enter();
    sin_q15_deterministic(phase_q15)
}

#[no_mangle]
pub extern "C" fn nt_cos_q15(phase_q15: i16) -> f32 {
    let _mxcsr = MxcsrGuard::enter();
    cos_q15_deterministic(phase_q15)
}

#[no_mangle]
pub extern "C" fn nt_atan2_f32(y: f32, x: f32) -> f32 {
    let _mxcsr = MxcsrGuard::enter();
    atan2_f32_deterministic(y, x)
}

#[no_mangle]
pub extern "C" fn nt_sqrt_f32(x: f32) -> f32 {
    let _mxcsr = MxcsrGuard::enter();
    let force_scalar = GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed) != 0;
    sqrt_f32_deterministic(x, force_scalar)
}

/// Balanced pairwise 16-lane dot product (`0-ULP` identical across AVX2, SSE2, NEON, WASM32, Scalar).
///
/// # Safety
/// `a` and `b` must each point to 16 `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn nt_dot16_pairwise_f32(a: *const f32, b: *const f32) -> f32 {
    let _mxcsr = MxcsrGuard::enter();
    if a.is_null() || b.is_null() {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    let a16 = &*(a as *const [f32; 16]);
    let b16 = &*(b as *const [f32; 16]);
    let force_scalar = GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed) != 0;
    dot16_pairwise_dispatch(a16, b16, force_scalar)
}

/// Evaluates the exact `SO(3)` Cayley attitude rotor `R = (I - A)(I + A)^(-1)` and rotates
/// optional `v_in_3 -> v_out_3` with `0.000e+00` norm drift polish.
///
/// # Safety
/// `omega_3` must point to 3 `f32`s; optional `v_in_3`/`v_out_3` to 3 `f32`s; optional `r_out_9` to 9 `f32`s.
#[no_mangle]
pub unsafe extern "C" fn nt_cayley_rotor_so3_f32(
    omega_3: *const f32,
    v_in_3: *const f32,
    r_out_9: *mut f32,
    v_out_3: *mut f32,
) -> i32 {
    let _mxcsr = MxcsrGuard::enter();
    if omega_3.is_null() {
        return -1;
    }
    let w = [*omega_3, *omega_3.add(1), *omega_3.add(2)];
    let vin = if !v_in_3.is_null() {
        Some([*v_in_3, *v_in_3.add(1), *v_in_3.add(2)])
    } else {
        None
    };
    let (r9, vrot) = cayley_rotor_so3_eval(&w, vin.as_ref());
    if !r_out_9.is_null() {
        for i in 0..9 {
            *r_out_9.add(i) = r9[i];
        }
    }
    if let (Some(vr), false) = (vrot, v_out_3.is_null()) {
        *v_out_3 = vr[0];
        *v_out_3.add(1) = vr[1];
        *v_out_3.add(2) = vr[2];
    }
    0
}

// ============================================================================
// Zero-Alloc WASM32 & Native Static Bridge Helpers
// ============================================================================

#[no_mangle]
pub extern "C" fn nt_wasm_global_arena() -> *mut NetTwinArena {
    GLOBAL_SERVER_ARENA_BSS.0.get() as *mut NetTwinArena
}

#[no_mangle]
pub extern "C" fn nt_wasm_client_arena() -> *mut NetTwinArena {
    GLOBAL_CLIENT_ARENA_BSS.0.get() as *mut NetTwinArena
}

#[no_mangle]
pub extern "C" fn nt_wasm_cmd_buf() -> *mut ClientInputCmd {
    GLOBAL_CMD_BUF_BSS.0.get() as *mut ClientInputCmd
}

#[no_mangle]
pub extern "C" fn nt_wasm_mtu_buf() -> *mut u8 {
    GLOBAL_MTU_BUF_BSS.0.get() as *mut u8
}

#[no_mangle]
pub extern "C" fn nt_wasm_hit_buf() -> *mut HitRecordOut {
    GLOBAL_HIT_BUF_BSS.0.get() as *mut HitRecordOut
}

#[no_mangle]
pub extern "C" fn nt_wasm_acoustic_buf() -> *mut AcousticImpulse {
    GLOBAL_ACOUSTIC_BUF_BSS.0.get() as *mut AcousticImpulse
}

// ============================================================================
// Comprehensive Unit & Adversarial Self-Verification Suite
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{
        NT_BTN_CROUCH, NT_BTN_FIRE, NT_BTN_LEAN_L, NT_BTN_LEAN_R, NT_BTN_SPRINT,
        NT_GATE_LAG_SWITCH_QUARANTINED, NT_GATE_NOMINAL, NT_HIT_FLAG_HIT,
        NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED, NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
        NT_HIT_FLAG_PENETRATED, NT_HIT_FLAG_RICOCHET,
    };
    use core::mem::{align_of, offset_of, size_of};
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    use std::time::Instant;

    struct OwnedArena {
        ptr: *mut NetTwinArena,
        layout: Layout,
    }

    impl OwnedArena {
        fn new_initialized() -> Self {
            let layout = Layout::from_size_align(NT_ARENA_BYTES, NT_ARENA_ALIGN).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) as *mut NetTwinArena };
            assert!(!ptr.is_null());
            assert_eq!(unsafe { nt_arena_init(ptr) }, 0);
            Self { ptr, layout }
        }
    }

    impl Drop for OwnedArena {
        fn drop(&mut self) {
            unsafe {
                dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }

    #[test]
    fn test_frozen_abi_sizes_alignments_and_offsets() {
        assert_eq!(size_of::<NetTwinArena>(), 1_572_864);
        assert_eq!(align_of::<NetTwinArena>(), 64);
        assert_eq!(offset_of!(NetTwinArena, tier_a_cylinders), 0);
        assert_eq!(offset_of!(NetTwinArena, tier_b_poses_plates), 65_536);
        assert_eq!(offset_of!(NetTwinArena, static_voxel_bvh), 327_680);
        assert_eq!(offset_of!(NetTwinArena, client_jitter_gates), 589_824);
        assert_eq!(offset_of!(NetTwinArena, projectiles_acoustic_slab), 655_360);
        assert_eq!(offset_of!(NetTwinArena, client_baseline_snapshots), 720_896);
        assert_eq!(offset_of!(NetTwinArena, triad_cascade_scratch), 1_245_184);
        assert_eq!(offset_of!(NetTwinArena, telemetry_and_header), 1_507_328);

        assert_eq!(size_of::<ClientInputCmd>(), 16);
        assert_eq!(size_of::<HitRecordOut>(), 32);
        assert_eq!(size_of::<AcousticImpulse>(), 12);
        assert_eq!(size_of::<PlayerPoseSnapshot>(), 32);
        assert_eq!(size_of::<PlayerTickState>(), 64);
        assert_eq!(size_of::<ClientJitterGate>(), 1024);
        assert_eq!(size_of::<TelemetryAndHeader>(), 65_536);
    }

    #[test]
    fn test_tier0_q15_trig_sqrt_dot16_mxcsr_and_cayley_zero_drift() {
        // Cardinal q15 phases
        assert_eq!(nt_sin_q15(0).to_bits(), 0.0f32.to_bits());
        assert_eq!(nt_sin_q15(-32768).to_bits(), 0.0f32.to_bits());
        assert_eq!(nt_sin_q15(16384).to_bits(), 1.0f32.to_bits());
        assert_eq!(nt_sin_q15(-16384).to_bits(), (-1.0f32).to_bits());
        assert_eq!(nt_cos_q15(0).to_bits(), 1.0f32.to_bits());
        assert_eq!(nt_cos_q15(16384).to_bits(), 0.0f32.to_bits());
        assert_eq!(nt_cos_q15(-32768).to_bits(), (-1.0f32).to_bits());

        // Verify pythagorean accuracy across all 65,536 modular q15 angles
        for p in i16::MIN..=i16::MAX {
            let s = nt_sin_q15(p);
            let c = nt_cos_q15(p);
            let err = (s * s + c * c - 1.0f32).abs();
            assert!(err < 2.5e-7, "q15 trig unit circle error {} at {}", err, p);
            assert_ne!(s.to_bits(), (-0.0f32).to_bits());
            assert_ne!(c.to_bits(), (-0.0f32).to_bits());
        }

        // Verify 0-ULP bit identity of sqrt_f32 (SIMD vs Scalar) across normals, subnormals, -0.0, NaN
        assert_eq!(math::sqrt_f32_deterministic(-0.0f32, false).to_bits(), 0);
        assert_eq!(math::sqrt_f32_deterministic(-1.0f32, false).to_bits(), CANONICAL_NAN_BITS);
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        for i in 0..50_000u32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = if i < 2000 {
                // Subnormal stress range
                (rng as u32) & 0x007f_ffff
            } else {
                ((rng as u32) & 0x7f7f_ffff).max(1)
            };
            let x = f32::from_bits(bits);
            let b_simd = math::sqrt_f32_deterministic(x, false).to_bits();
            let b_scal = math::sqrt_f32_deterministic(x, true).to_bits();
            assert_eq!(b_simd, b_scal, "sqrt 0-ULP mismatch on bits={:#010x}", bits);
        }

        // Verify 0-ULP bit identity of dot16_pairwise_f32 across AVX2, SSE2, Scalar, and hostile MXCSR FTZ/DAZ!
        for case in 0..2_000 {
            let mut a = [0.0f32; 16];
            let mut b = [0.0f32; 16];
            for k in 0..16 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let sub = (case % 5 == 0) && (k % 3 == 0);
                a[k] = if sub {
                    f32::from_bits(((rng as u32) & 0x003f_ffff) | 1)
                } else {
                    (((rng >> 32) as i32) as f32) * 1.37e-7f32
                };
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                b[k] = if sub {
                    1.25f32
                } else {
                    (((rng >> 32) as i32) as f32) * -2.91e-7f32
                };
            }
            let d_scal = math::dot16_pairwise_scalar(&a, &b).to_bits();
            let d_auto = math::dot16_pairwise_dispatch(&a, &b, false).to_bits();
            assert_eq!(d_auto, d_scal, "dot16 SIMD vs Scalar mismatch at case {}", case);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let d_sse2 = math::dot16_pairwise_sse2(&a, &b).to_bits();
                assert_eq!(d_sse2, d_scal);
                if math::x86_has_avx2() {
                    let d_avx2 = math::dot16_pairwise_avx2(&a, &b).to_bits();
                    assert_eq!(d_avx2, d_scal);
                }
                // Hostile caller MXCSR test: set FTZ=1 (0x8000) and DAZ=1 (0x0040)
                let mut orig_csr: u32 = 0;
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut orig_csr, options(nostack, preserves_flags));
                let hostile_csr = orig_csr | 0x8040u32;
                core::arch::asm!("ldmxcsr [{}]", in(reg) &hostile_csr, options(nostack, preserves_flags));
                let d_protected = nt_dot16_pairwise_f32(a.as_ptr(), b.as_ptr()).to_bits();
                let mut after_csr: u32 = 0;
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut after_csr, options(nostack, preserves_flags));
                core::arch::asm!("ldmxcsr [{}]", in(reg) &orig_csr, options(nostack, preserves_flags));
                assert_eq!(d_protected, d_scal, "MXCSR guard failed to block FTZ/DAZ!");
                assert_eq!(after_csr & 0x8040u32, 0x8040u32, "MXCSR guard failed to restore caller FTZ/DAZ!");
            }
        }

        // Verify exact 0.000e+00 norm conservation in SO(3) Cayley rotor
        let mut v = [0.6f32, -0.8f32, 0.5f32];
        let orig_norm = nt_sqrt_f32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        let mut r9 = [0.0f32; 9];
        for step in 0..1000 {
            let w = [
                0.15f32 * ((step as f32) * 0.03f32).sin(),
                -0.22f32,
                0.09f32,
            ];
            let mut v_next = [0.0f32; 3];
            assert_eq!(
                unsafe { nt_cayley_rotor_so3_f32(w.as_ptr(), v.as_ptr(), r9.as_mut_ptr(), v_next.as_mut_ptr()) },
                0
            );
            v = v_next;
        }
        let final_norm = nt_sqrt_f32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        assert_eq!(final_norm.to_bits(), orig_norm.to_bits(), "Non-zero Cayley norm drift!");
    }

    #[test]
    fn test_10000_ticks_simd_vs_scalar_zero_ulp_parity() {
        let arena_simd = OwnedArena::new_initialized();
        let arena_scal = OwnedArena::new_initialized();

        unsafe {
            nt_set_simd_backend_override(arena_simd.ptr, 0);
            nt_set_simd_backend_override(arena_scal.ptr, 1);
            nt_load_default_raid_map(arena_simd.ptr);
            nt_load_default_raid_map(arena_scal.ptr);

            let spawns: [(f32, f32, i16); 8] = [
                (-6.0, -4.0, 4096),
                (6.0, 4.0, -28000),
                (-4.0, 5.0, -8192),
                (5.0, -5.0, 12000),
                (-15.0, -3.0, 0),
                (15.0, 3.0, -32768),
                (-3.0, -16.0, 16384),
                (3.0, 16.0, -16384),
            ];
            for (pid, &(x, y, yaw)) in spawns.iter().enumerate() {
                nt_spawn_player(arena_simd.ptr, pid as u32, x, y, 0.0, yaw);
                nt_spawn_player(arena_scal.ptr, pid as u32, x, y, 0.0, yaw);
            }

            let mut state_rng: u64 = 0xCAFE_BABE_DEAD_BEEF;
            let mut hit_simd = HitRecordOut::default();
            let mut hit_scal = HitRecordOut::default();

            for tick in 0..10_000u32 {
                for pid in 0..8u32 {
                    state_rng = state_rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let btns = ((state_rng >> 16) as u16) & 0x3F;
                    let mx = ((state_rng >> 32) as i16).clamp(-32767, 32767);
                    let my = ((state_rng >> 48) as i16).clamp(-32767, 32767);
                    let yaw = (state_rng as i16).wrapping_add((tick as i16).wrapping_mul(97));
                    let pitch = (((state_rng >> 24) as i16) % 4096).clamp(-4000, 4000);
                    let cmd = ClientInputCmd {
                        cmd_tick: tick.saturating_sub(4),
                        buttons_stance: btns,
                        move_xy_q15: [mx, my],
                        yaw_pitch_q15: [yaw, pitch],
                        fire_subtick_u8: (state_rng >> 8) as u8,
                        weapon_slot_u8: (pid & 3) as u8,
                    };
                    nt_ingest_client_cmd(arena_simd.ptr, pid, &cmd);
                    nt_ingest_client_cmd(arena_scal.ptr, pid, &cmd);

                    if (btns & NT_BTN_FIRE) != 0 && (tick % 17 == pid) {
                        nt_resolve_subtick_fire(arena_simd.ptr, pid, &cmd, &mut hit_simd);
                        nt_resolve_subtick_fire(arena_scal.ptr, pid, &cmd, &mut hit_scal);
                        assert_eq!(hit_simd.outcome_flags, hit_scal.outcome_flags);
                        assert_eq!(hit_simd.damage_applied.to_bits(), hit_scal.damage_applied.to_bits());
                        assert_eq!(hit_simd.hit_dist_m.to_bits(), hit_scal.hit_dist_m.to_bits());
                    }
                }

                nt_step_kinematics(arena_simd.ptr);
                nt_step_kinematics(arena_scal.ptr);

                if (tick & 511) == 511 || tick == 9_999 {
                    let h_simd = nt_hash_arena_deterministic(arena_simd.ptr);
                    let h_scal = nt_hash_arena_deterministic(arena_scal.ptr);
                    assert_eq!(
                        h_simd, h_scal,
                        "0-ULP digest mismatch at tick {}: simd={:#018x} scal={:#018x}",
                        tick, h_simd, h_scal
                    );
                }
            }

            // Equalize simd_mode_override and verify raw 1,572,864-byte arena equality!
            nt_set_simd_backend_override(arena_scal.ptr, 0);
            let bytes_simd = core::slice::from_raw_parts(arena_simd.ptr as *const u8, NT_ARENA_BYTES);
            let bytes_scal = core::slice::from_raw_parts(arena_scal.ptr as *const u8, NT_ARENA_BYTES);
            assert_eq!(bytes_simd, bytes_scal, "Raw 1.50 MB arena byte mismatch after 10,000 ticks!");
        }
    }

    #[test]
    fn test_32_tick_client_reconcile_latency_under_90_us() {
        let arena = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena.ptr);
            nt_spawn_player(arena.ptr, 0, -2.5, -2.5, 0.0, 4096);

            let auth_pose = PlayerPoseSnapshot {
                x: -2.5,
                y: -2.5,
                z: 0.0,
                vx: 3.0,
                vy: 2.0,
                vz: 0.0,
                yaw_q15: 4096,
                pitch_q15: 0,
                lean_q15: 0,
                stance_flags: NT_STANCE_ACTIVE | NT_STANCE_GROUNDED,
            };

            let mut cmds = [ClientInputCmd::default(); 32];
            for (i, c) in cmds.iter_mut().enumerate() {
                *c = ClientInputCmd {
                    cmd_tick: (i + 1) as u32,
                    buttons_stance: if i % 4 == 0 {
                        NT_BTN_SPRINT | NT_BTN_LEAN_R
                    } else {
                        NT_BTN_LEAN_L
                    },
                    move_xy_q15: [30000, 12000],
                    yaw_pitch_q15: [4096 + (i as i16) * 64, 0],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
            }

            let mut out_pose = PlayerPoseSnapshot::default();
            // Warmup
            for _ in 0..16 {
                nt_client_reconcile(arena.ptr, 0, &auth_pose, cmds.as_ptr(), 32, &mut out_pose);
            }

            let runs = 200u32;
            let t0 = Instant::now();
            for _ in 0..runs {
                nt_client_reconcile(arena.ptr, 0, &auth_pose, cmds.as_ptr(), 32, &mut out_pose);
                core::hint::black_box(&out_pose);
            }
            let avg_us = (t0.elapsed().as_secs_f64() * 1e6) / (runs as f64);
            assert!(
                avg_us < 90.0,
                "32-tick nt_client_reconcile took {:.2} us (budget < 90.0 us)",
                avg_us
            );
        }
    }

    #[test]
    fn test_behind_pillar_mutual_occluder_cutoff_and_plate_ricochet() {
        let arena = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena.ptr);
            // Central concrete pillar is [-1.5, 1.5] x [-1.5, 1.5] x [0.0, 3.5].
            // Shooter 0 at (-5.0, -2.2, 0.0) with 220 ms RTT (28 ticks);
            // Anchor 1 starts sprinting north from (2.0, -2.72, 0.0) with honest 30 ms RTT (4 ticks -> tau_fair = 2 ticks).
            nt_spawn_player(arena.ptr, 0, -5.0, -2.2, 0.0, 0);
            nt_spawn_player(arena.ptr, 1, 2.0, -2.72, 0.0, 16384);
            nt_set_player_rtt_ticks(arena.ptr, 0, 28);
            nt_set_player_rtt_ticks(arena.ptr, 1, 4);

            // Warm up Anchor's sprint velocity to full 6.6 m/s up to exposed lane at y ~ -2.04 (clear LOS south of pillar)
            for _ in 0..26 {
                let cur_t = (*arena.ptr).telemetry_and_header.server_tick;
                let anchor_cmd = ClientInputCmd {
                    cmd_tick: cur_t.saturating_sub(2),
                    buttons_stance: NT_BTN_SPRINT,
                    move_xy_q15: [32767, 0],
                    yaw_pitch_q15: [16384, 0],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
                nt_ingest_client_cmd(arena.ptr, 1, &anchor_cmd);
                nt_step_kinematics(arena.ptr);
            }
            let exposed_tick = (*arena.ptr).telemetry_and_header.server_tick;
            let mut exposed_pose = PlayerPoseSnapshot::default();
            nt_get_player_pose(arena.ptr, 1, &mut exposed_pose);
            assert!(
                exposed_pose.y < -1.85f32,
                "Anchor should still be exposed south of pillar at exposed_tick, y={}",
                exposed_pose.y
            );

            // Now Anchor continues sprinting north for 26 ticks (~203 ms) deep behind the concrete pillar to y ~ -0.7
            for _ in 0..26u32 {
                let cur_t = (*arena.ptr).telemetry_and_header.server_tick;
                let anchor_cmd = ClientInputCmd {
                    cmd_tick: cur_t.saturating_sub(2),
                    buttons_stance: NT_BTN_SPRINT,
                    move_xy_q15: [32767, 0],
                    yaw_pitch_q15: [16384, 0],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
                nt_ingest_client_cmd(arena.ptr, 1, &anchor_cmd);
                nt_step_kinematics(arena.ptr);
            }

            let mut covered_pose = PlayerPoseSnapshot::default();
            nt_get_player_pose(arena.ptr, 1, &mut covered_pose);
            assert!(
                covered_pose.y > -0.95f32 && covered_pose.y < 1.25f32,
                "Anchor should be behind central pillar, got y={}",
                covered_pose.y
            );

            // Aim from Shooter (-5.0, -2.2) directly at Anchor's OLD exposed position (exposed_pose.x, exposed_pose.y)
            let aim_rad = nt_atan2_f32(exposed_pose.y - (-2.2f32), exposed_pose.x - (-5.0f32));
            let aim_yaw_q15 = ((aim_rad / core::f32::consts::PI) * 32768.0f32) as i16;
            let late_fire_cmd = ClientInputCmd {
                cmd_tick: exposed_tick, // 26 ticks ago (~203 ms rewind!)
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [aim_yaw_q15, -400], // Aim at chest
                fire_subtick_u8: 0,
                weapon_slot_u8: 0,
            };

            let mut hit_out = HitRecordOut::default();
            nt_resolve_subtick_fire(arena.ptr, 0, &late_fire_cmd, &mut hit_out);
            assert_eq!(hit_out.victim_id, 1);
            assert_ne!(
                hit_out.outcome_flags & NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
                0,
                "Expected Mutual Occluder Horizon Cutoff rejection behind concrete pillar! flags={:#x}",
                hit_out.outcome_flags
            );
            assert_eq!(hit_out.outcome_flags & NT_HIT_FLAG_HIT, 0);
            assert_eq!(hit_out.damage_applied, 0.0f32);

            // Now test 30-plate penetration vs grazing obliquity ricochet in clear line-of-sight!
            nt_spawn_player(arena.ptr, 2, -3.0, -5.0, 0.0, 0);
            nt_spawn_player(arena.ptr, 3, 5.0, -5.0, 0.0, -32768); // Facing Shooter 2 at 8m open lane
            let now_tick = (*arena.ptr).telemetry_and_header.server_tick;

            // 1) Center-mass shot -> penetration
            let center_cmd = ClientInputCmd {
                cmd_tick: now_tick,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [0, -300],
                fire_subtick_u8: 0,
                weapon_slot_u8: 1, // AP rifle
            };
            let mut center_hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena.ptr, 2, &center_cmd, &mut center_hit);
            assert_eq!(center_hit.victim_id, 3);
            assert_ne!(center_hit.outcome_flags & NT_HIT_FLAG_HIT, 0);
            assert_ne!(center_hit.outcome_flags & NT_HIT_FLAG_PENETRATED, 0);
            assert!(center_hit.damage_applied > 20.0f32);

            // 2) Grazing edge shot (~0.335 m off-center on 0.35 m hard Ceramic IV cummerbund plate) -> RICOCHET!
            let graze_rad = nt_atan2_f32(0.335f32, 8.0f32);
            let graze_yaw_q15 = ((graze_rad / core::f32::consts::PI) * 32768.0f32) as i16;
            let graze_cmd = ClientInputCmd {
                cmd_tick: now_tick,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [graze_yaw_q15, -750],
                fire_subtick_u8: 0,
                weapon_slot_u8: 0,
            };
            let mut graze_hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena.ptr, 2, &graze_cmd, &mut graze_hit);
            assert_eq!(graze_hit.victim_id, 3);
            assert_ne!(graze_hit.outcome_flags & NT_HIT_FLAG_RICOCHET, 0);
        }
    }

    #[test]
    fn test_clumsy_lag_switch_quarantine_clamps_rewind() {
        let arena = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena.ptr);
            nt_spawn_player(arena.ptr, 0, -4.0, -5.0, 0.0, 0);
            nt_spawn_player(arena.ptr, 1, 4.0, -5.0, 0.0, -32768);

            // 30 ticks of honest steady 160 ms RTT (20 ticks RTT = 10 ticks one-way) -> ZERO quarantine!
            for _ in 0..30 {
                nt_step_kinematics(arena.ptr);
                let srv = (*arena.ptr).telemetry_and_header.server_tick;
                if srv >= 10 {
                    let honest_cmd = ClientInputCmd {
                        cmd_tick: srv - 10,
                        buttons_stance: 0,
                        move_xy_q15: [0, 0],
                        yaw_pitch_q15: [0, 0],
                        fire_subtick_u8: 0,
                        weapon_slot_u8: 0,
                    };
                    let status = nt_ingest_client_cmd(arena.ptr, 0, &honest_cmd);
                    assert_eq!(status as u32, NT_GATE_NOMINAL);
                }
            }
            assert_eq!((*arena.ptr).telemetry_and_header.total_jitter_quarantines, 0);

            // Now simulate a 200 ms `clumsy` lag-switch hold (25 server ticks with 0 packets),
            // followed by dumping all 25 held commands in a single micro-burst clump!
            let hold_start_cmd = (*arena.ptr).telemetry_and_header.server_tick - 9;
            for _ in 0..25 {
                nt_step_kinematics(arena.ptr);
            }
            let mut last_status = 0;
            for k in 0..25u32 {
                let burst_cmd = ClientInputCmd {
                    cmd_tick: hold_start_cmd + k,
                    buttons_stance: 0,
                    move_xy_q15: [0, 0],
                    yaw_pitch_q15: [0, 0],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
                last_status = nt_ingest_client_cmd(arena.ptr, 0, &burst_cmd);
            }
            assert_eq!(last_status as u32, NT_GATE_LAG_SWITCH_QUARANTINED);
            assert!((*arena.ptr).telemetry_and_header.total_jitter_quarantines >= 1);
            assert!((*arena.ptr).client_jitter_gates[0].clamped_rewind_max_ticks <= 2);

            // Attempt a 20-tick backward shot while quarantined -> must clamp rewind & set flag 0x10!
            let srv_now = (*arena.ptr).telemetry_and_header.server_tick;
            let exploit_shot = ClientInputCmd {
                cmd_tick: srv_now - 20,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [0, -300],
                fire_subtick_u8: 0,
                weapon_slot_u8: 0,
            };
            let mut hit_rec = HitRecordOut::default();
            nt_resolve_subtick_fire(arena.ptr, 0, &exploit_shot, &mut hit_rec);
            assert_ne!(hit_rec.outcome_flags & NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED, 0);
            assert!(srv_now - hit_rec.rewind_tick <= 2);
        }
    }

    #[test]
    fn test_anti_dma_byte_for_byte_non_interference_and_anonymous_acoustics() {
        let srv_a = OwnedArena::new_initialized();
        let srv_b = OwnedArena::new_initialized();
        let cli_a = OwnedArena::new_initialized();
        let cli_b = OwnedArena::new_initialized();

        unsafe {
            nt_load_default_raid_map(srv_a.ptr);
            nt_load_default_raid_map(srv_b.ptr);
            nt_load_default_raid_map(cli_a.ptr);
            nt_load_default_raid_map(cli_b.ptr);

            // Observer 0 at (-5.0, 0.0, 0.0) in central hall; Visible Enemy 1 at (-5.0, 5.0, 0.0).
            nt_spawn_player(srv_a.ptr, 0, -5.0, 0.0, 0.0, 0);
            nt_spawn_player(srv_a.ptr, 1, -5.0, 5.0, 0.0, -16384);
            nt_spawn_player(srv_b.ptr, 0, -5.0, 0.0, 0.0, 0);
            nt_spawn_player(srv_b.ptr, 1, -5.0, 5.0, 0.0, -16384);

            // Spawn 15 occluded enemies (2..=16) behind solid concrete outer walls at (±24, ±24)
            for p in 2..=16u32 {
                let ox = if (p & 1) == 0 { -24.0f32 } else { 24.0f32 };
                let oy = if (p & 2) == 0 { -24.0f32 } else { 24.0f32 };
                nt_spawn_player(srv_a.ptr, p, ox, oy, 0.0, 0);
                // In srv_b, drastically mutate positions (still behind outer walls), yaw, armor, HP, vel!
                nt_spawn_player(srv_b.ptr, p, ox + 2.5f32, oy + 1.8f32, 0.0, 21234);
                let slot = ((*srv_b.ptr).telemetry_and_header.server_tick as usize) & 63;
                let st = &mut (*srv_b.ptr).tier_b_poses_plates[slot][p as usize];
                st.vel = [4.0, -3.5, 0.0];
                st.health_q8 = 3200;
                st.plate_intact_mask = 0x0000_000F;
                st.weapon_id = 3;
                st.stance_flags |= NT_BTN_CROUCH;
            }

            let mut mtu_a = [0xAAu8; 1200];
            let mut mtu_b = [0x55u8; 1200];

            let len_a = nt_pack_client_datagram(srv_a.ptr, 0, 0, mtu_a.as_mut_ptr(), 1200);
            let len_b = nt_pack_client_datagram(srv_b.ptr, 0, 0, mtu_b.as_mut_ptr(), 1200);

            assert!(len_a > 16 && (len_a as usize) < NT_MTU_CEILING_BYTES);
            assert_eq!(len_a, len_b);
            assert_eq!(
                mtu_a, mtu_b,
                "CRITICAL ANTI-DMA VIOLATION: occluded actors leaked bits into out_mtu_buf!"
            );

            // Unpack into Client A and Client B arenas and verify 100% of 1,572,864 client RAM bytes match
            // AND all 15 occluded slots are 0x00 zeroed!
            let mut ac_a = [AcousticImpulse::default(); 16];
            let mut ac_b = [AcousticImpulse::default(); 16];
            let mut nac_a = 0u32;
            let mut nac_b = 0u32;

            let vis_a = nt_unpack_client_datagram(
                cli_a.ptr,
                mtu_a.as_ptr(),
                len_a as u32,
                ac_a.as_mut_ptr(),
                16,
                &mut nac_a,
            );
            let vis_b = nt_unpack_client_datagram(
                cli_b.ptr,
                mtu_b.as_ptr(),
                len_b as u32,
                ac_b.as_mut_ptr(),
                16,
                &mut nac_b,
            );
            assert_eq!(vis_a, 2); // ONLY Observer 0 + Visible Enemy 1!
            assert_eq!(vis_b, 2);
            assert_eq!(nac_a, 0);
            assert_eq!(nac_b, 0);

            let ram_a = core::slice::from_raw_parts(cli_a.ptr as *const u8, NT_ARENA_BYTES);
            let ram_b = core::slice::from_raw_parts(cli_b.ptr as *const u8, NT_ARENA_BYTES);
            assert_eq!(ram_a, ram_b, "Client unpacked arena leaked occluded enemy differences!");

            for p in 2..=16u32 {
                let mut pose = PlayerPoseSnapshot::default();
                nt_get_player_pose(cli_a.ptr, p, &mut pose);
                assert_eq!(pose, PlayerPoseSnapshot::default(), "Occluded player {} not scrubbed in client RAM!", p);
            }

            // Now emit a behind-wall acoustic event behind the solid West Wing wall (-13.0, 4.5, 0.0)
            // (blocked by concrete wall x=-10, y in [2.0, 31.0], diffracting through doorway (-10.0, 0.0, 1.3))
            // and verify anonymous portal-diffracted AcousticImpulse (12 B, zero entity_id, zero xyz)!
            let loud_bands = [220u8, 200u8, 160u8, 120u8];
            nt_spawn_player(srv_a.ptr, 7, -13.0, 4.5, 0.0, 0);
            nt_emit_acoustic_event(srv_a.ptr, 7, -13.0, 4.5, 0.0, loud_bands.as_ptr());

            let len_ac = nt_pack_client_datagram(srv_a.ptr, 0, 0, mtu_a.as_mut_ptr(), 1200);
            assert!(len_ac > len_a);
            let vis_ac = nt_unpack_client_datagram(
                cli_a.ptr,
                mtu_a.as_ptr(),
                len_ac as u32,
                ac_a.as_mut_ptr(),
                16,
                &mut nac_a,
            );
            assert_eq!(vis_ac, 2); // Enemy 7 behind wall is STILL excluded from entity table!
            assert_eq!(nac_a, 1);  // Only 1 anonymous 12-byte diffracted AcousticImpulse delivered!
            assert!(ac_a[0].band_energy_q8[0] > ac_a[0].band_energy_q8[3], "High frequency band should attenuate more than low band under portal diffraction!");
        }
    }

    #[test]
    fn test_v02_bug1_modular_delta_lean_qvz_stance_and_64p_mtu() {
        let srv = OwnedArena::new_initialized();
        let cli = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(srv.ptr);
            nt_load_default_raid_map(cli.ptr);

            // Observer 0 at (-5.0, 0.0) and visible Enemy 1 at (-5.0, 4.0)
            nt_spawn_player(srv.ptr, 0, -5.0, 0.0, 0.0, 16384);
            nt_spawn_player(srv.ptr, 1, -5.0, 4.0, 0.0, -16384);

            let mut mtu = [0u8; 1200];
            let mut ac = [AcousticImpulse::default(); 16];
            let mut nac = 0u32;

            // Initial full baseline snapshot at tick 0
            let len0 = nt_pack_client_datagram(srv.ptr, 0, 0, mtu.as_mut_ptr(), 1200);
            assert!(len0 > 16);
            assert_eq!(
                nt_unpack_client_datagram(cli.ptr, mtu.as_ptr(), len0 as u32, ac.as_mut_ptr(), 16, &mut nac),
                2
            );

            // Step 40 ticks toggling lean_l, lean_r, crouch, sprint, ads, vz, and >30cm wide jump
            let mut saw_short_delta = false;
            let mut saw_wide_delta = false;
            for step in 0..40u32 {
                let prev_srv_tick = (*srv.ptr).telemetry_and_header.server_tick;
                let btns = match step % 4 {
                    0 => NT_BTN_LEAN_L | NT_BTN_CROUCH,
                    1 => NT_BTN_LEAN_R | crate::arena::NT_BTN_ADS,
                    2 => NT_BTN_SPRINT,
                    _ => NT_BTN_LEAN_L,
                };
                let cmd1 = ClientInputCmd {
                    cmd_tick: prev_srv_tick + 1,
                    buttons_stance: btns,
                    move_xy_q15: [20000, -15000],
                    yaw_pitch_q15: [-16384 + (step as i16) * 512, -1200 + (step as i16) * 64],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
                nt_ingest_client_cmd(srv.ptr, 1, &cmd1);
                nt_step_kinematics(srv.ptr);

                let cur_tick = (*srv.ptr).telemetry_and_header.server_tick;
                let cur_slot = (cur_tick as usize) & 63;
                // Inject non-zero vertical velocity qvz and, on step 20, a >0.35 m displacement to exercise 0xC0 wide delta
                (*srv.ptr).tier_b_poses_plates[cur_slot][1].vel[2] =
                    if (step & 1) == 0 { 2.25f32 } else { -1.75f32 };
                if step == 20 {
                    (*srv.ptr).tier_b_poses_plates[cur_slot][1].pos[0] += 0.48f32;
                }

                let dlen = nt_pack_client_datagram(srv.ptr, 0, prev_srv_tick, mtu.as_mut_ptr(), 1200);
                assert!(dlen > 16 && (dlen as usize) < NT_MTU_CEILING_BYTES);

                // Inspect Enemy 1 tag (Observer 0 is first record at byte 16; Enemy 1 follows it)
                let obs_tag = mtu[16] & 0xC0;
                let obs_rec_len = if obs_tag == 0x80 { 12usize } else { 14usize };
                let enemy_tag = mtu[16 + obs_rec_len] & 0xC0;
                if enemy_tag == 0x80 {
                    saw_short_delta = true;
                } else if enemy_tag == 0xC0 {
                    saw_wide_delta = true;
                }

                let vis = nt_unpack_client_datagram(
                    cli.ptr,
                    mtu.as_ptr(),
                    dlen as u32,
                    ac.as_mut_ptr(),
                    16,
                    &mut nac,
                );
                assert_eq!(vis, 2);

                // Zero-drift quantized state equality across all 10 quantized fields (lean_q8, qvz_16, stance_u16, etc.)
                let q_srv = (*srv.ptr).client_baseline_snapshots.quantized_ring[cur_slot][1];
                let q_cli = (*cli.ptr).client_baseline_snapshots.quantized_ring[cur_slot][1];
                assert_eq!(q_cli, q_srv, "Modular delta drift at step {}!", step);

                let s_cli = (*cli.ptr).tier_b_poses_plates[cur_slot][1];
                assert_eq!(s_cli.lean_q15, (q_srv.lean_q8 as i16) << 8);
                assert_eq!(s_cli.stance_flags & NT_STANCE_MASK_7BIT, q_srv.stance_u16 & NT_STANCE_MASK_7BIT);
                assert!((s_cli.vel[2] - (q_srv.qvz_16 as f32) / 16.0f32).abs() < 1e-6f32);
            }
            assert!(saw_short_delta, "Expected 12-byte 0x80 short modular delta records");
            assert!(saw_wide_delta, "Expected 14-byte 0xC0 wide modular delta record on >25.4cm jump");

            // Worst-case 64-player full-state + 8 acoustic impulses <= 1,200 B MTU ceiling
            // Spawn all 64 players in unobstructed West Hall (x in [-28, -14], y in [-9, +9])
            let srv64 = OwnedArena::new_initialized();
            nt_load_default_raid_map(srv64.ptr);
            for p in 0..64u32 {
                let col = (p % 8) as f32;
                let row = (p / 8) as f32;
                let rx = -26.5f32 + col * 1.5f32;
                let ry = -7.5f32 + row * 2.0f32;
                nt_spawn_player(srv64.ptr, p, rx, ry, 0.0, 0);
            }
            let bands = [210u8, 180, 140, 100];
            for k in 0..8u32 {
                nt_emit_acoustic_event(srv64.ptr, k + 1, -20.0, 0.0, 0.0, bands.as_ptr());
            }
            let len64 = nt_pack_client_datagram(srv64.ptr, 0, 0, mtu.as_mut_ptr(), 1200);
            assert!(
                len64 > 0 && (len64 as usize) <= 1008 && (len64 as usize) < NT_MTU_CEILING_BYTES,
                "64p + 8 acoustics exceeded MTU: {} B",
                len64
            );
            let cli64 = OwnedArena::new_initialized();
            let vis64 = nt_unpack_client_datagram(
                cli64.ptr,
                mtu.as_ptr(),
                len64 as u32,
                ac.as_mut_ptr(),
                16,
                &mut nac,
            );
            assert_eq!(vis64, 64);
            assert_eq!(nac, 8);
        }
    }

    #[test]
    fn test_v02_bug2_long_range_and_ricochet_projectiles_hit_capsules_no_double_sweep() {
        // (a) CQB hit at <= 25 m resolves instantaneously and spawns 0 straight-through projectiles (no double-sweep)
        let arena_cqb = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena_cqb.ptr);
            // Clear 58m outer-west corridor along x = -18.0
            nt_spawn_player(arena_cqb.ptr, 0, -18.0, -20.0, 0.0, 16384); // Shooter facing +Y
            nt_spawn_player(arena_cqb.ptr, 1, -18.0, -10.0, 0.0, -16384); // CQB victim at 10 m
            nt_spawn_player(arena_cqb.ptr, 2, -18.0, 18.0, 0.0, -16384); // Downrange bystander at 38 m

            let shot = ClientInputCmd {
                cmd_tick: 0,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [16384, -300],
                fire_subtick_u8: 0,
                weapon_slot_u8: 1, // AP rifle (penetrates)
            };
            let mut hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena_cqb.ptr, 0, &shot, &mut hit);
            assert_eq!(hit.victim_id, 1);
            assert_ne!(hit.outcome_flags & NT_HIT_FLAG_HIT, 0);
            assert_eq!(
                (*arena_cqb.ptr).telemetry_and_header.active_projectile_count,
                0,
                "Penetrating CQB hit must NOT spawn a duplicate projectile!"
            );
            for _ in 0..10 {
                nt_step_kinematics(arena_cqb.ptr);
            }
            let slot_end = ((*arena_cqb.ptr).telemetry_and_header.server_tick as usize) & 63;
            assert_eq!(
                (*arena_cqb.ptr).tier_b_poses_plates[slot_end][2].health_q8,
                25600,
                "Downrange bystander should take zero double-sweep damage!"
            );
        }

        // (b) Long-range (> 25 m) shot spawns projectile at 25.0 m handoff and hits enemy capsule at 38 m!
        let arena_lr = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena_lr.ptr);
            nt_spawn_player(arena_lr.ptr, 0, -18.0, -20.0, 0.0, 16384); // Shooter at y = -20
            nt_spawn_player(arena_lr.ptr, 1, -18.0, 18.0, 0.0, -16384); // Target at y = +18 (38 m > 25 m!)

            let lr_shot = ClientInputCmd {
                cmd_tick: 0,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [16384, -150], // Aim at chest accounting for slight 13m ballistic drop
                fire_subtick_u8: 0,
                weapon_slot_u8: 2, // 7.62x51 M61 Sniper (840 m/s)
            };
            let mut lr_hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena_lr.ptr, 0, &lr_shot, &mut lr_hit);
            assert_eq!(lr_hit.victim_id, 0xFFFF_FFFF); // Deferred beyond 25 m CQB horizon
            assert_eq!((*arena_lr.ptr).telemetry_and_header.active_projectile_count, 1);
            // Verify handoff starts at ~25.0 m from muzzle (y ~ -20 + 25 = +5.0), never double-sweeping [0, 25 m]
            let p0 = (*arena_lr.ptr).projectiles_acoustic_slab.projectiles[0];
            assert!((p0.traveled_dist_m - 25.0f32).abs() < 1e-4f32);
            assert!((p0.pos[1] - 5.0f32).abs() < 0.25f32);

            let orig_hp = (*arena_lr.ptr).tier_b_poses_plates[0][1].health_q8;
            let orig_armor: u32 = (*arena_lr.ptr).tier_b_poses_plates[0][1]
                .plate_hp_pools_q8
                .iter()
                .map(|&x| x as u32)
                .sum();

            for _ in 0..6 {
                nt_step_kinematics(arena_lr.ptr);
            }
            let slot_lr = ((*arena_lr.ptr).telemetry_and_header.server_tick as usize) & 63;
            let post_st = (*arena_lr.ptr).tier_b_poses_plates[slot_lr][1];
            let post_armor: u32 = post_st.plate_hp_pools_q8.iter().map(|&x| x as u32).sum();
            assert!(
                post_st.health_q8 < orig_hp && post_armor < orig_armor,
                "Long-range (>25m) projectile failed to damage enemy capsule/plates! hp={} armor={}",
                post_st.health_q8,
                post_armor
            );
            assert_eq!(
                (*arena_lr.ptr).telemetry_and_header.active_projectile_count,
                0,
                "Absorbed/penetrating projectile should deactivate upon capsule impact!"
            );
        }

        // (c) Shallow hard-cover graze at <= 25 m ricochets into an active projectile that hits angled target!
        let arena_ric = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena_ric.ptr);
            // South wall inner face is at y = -11.5 for x in [-9.5, -2.0].
            // Shooter at (-8.5, -10.7, 0.0) fires along (dx=3.2, dy=-0.8) -> hits y=-11.5 at x=-5.3,
            // bounces along (dx=3.2, dy=+0.8) straight through Target 1 at (-2.1, -10.7, 0.0)!
            let ric_rad = nt_atan2_f32(-0.8f32, 3.2f32);
            let ric_yaw = ((ric_rad / core::f32::consts::PI) * 32768.0f32) as i16;
            nt_spawn_player(arena_ric.ptr, 0, -8.5, -10.7, 0.0, ric_yaw);
            nt_spawn_player(arena_ric.ptr, 1, -2.1, -10.7, 0.0, -32768);

            let ric_shot = ClientInputCmd {
                cmd_tick: 0,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [ric_yaw, -200],
                fire_subtick_u8: 0,
                weapon_slot_u8: 1,
            };
            let mut ric_hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena_ric.ptr, 0, &ric_shot, &mut ric_hit);
            assert_ne!(ric_hit.outcome_flags & NT_HIT_FLAG_RICOCHET, 0);
            assert_eq!((*arena_ric.ptr).telemetry_and_header.active_projectile_count, 1);

            nt_step_kinematics(arena_ric.ptr);
            nt_step_kinematics(arena_ric.ptr);
            let slot_r = ((*arena_ric.ptr).telemetry_and_header.server_tick as usize) & 63;
            let st_r = (*arena_ric.ptr).tier_b_poses_plates[slot_r][1];
            let armor_r: u32 = st_r.plate_hp_pools_q8.iter().map(|&x| x as u32).sum();
            assert!(
                st_r.health_q8 < 25600 || armor_r < (12800 + 20480 + 17920 + 10240),
                "Wall-ricocheted projectile failed to hit angled target capsule!"
            );
        }
    }

    #[test]
    fn test_v02_bug3_pure_idempotent_client_reconcile_and_zero_stamina_preservation() {
        let srv = OwnedArena::new_initialized();
        let cli = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(srv.ptr);
            nt_load_default_raid_map(cli.ptr);
            nt_spawn_player(srv.ptr, 0, -18.0, -25.0, 0.0, 16384);
            nt_spawn_player(cli.ptr, 0, -18.0, -25.0, 0.0, 16384);

            // Force stamina_q15 = 0 (complete sprint exhaustion) on server
            let slot0 = ((*srv.ptr).telemetry_and_header.server_tick as usize) & 63;
            (*srv.ptr).tier_b_poses_plates[slot0][0].stamina_q15 = 0;

            let mut auth_zero_stam = PlayerPoseSnapshot::default();
            nt_get_player_pose(srv.ptr, 0, &mut auth_zero_stam);
            assert_ne!(auth_zero_stam.stance_flags & NT_POSE_HAS_STAMINA, 0);
            assert_eq!(auth_zero_stam.stance_flags >> 8, 0, "Packed stamina u8 must be 0!");

            // Build 16 sprint-attempt commands: since stamina == 0 (and 16*55 = 880 <= 1024),
            // server MUST stay at walk speed (never sprinting) across all 16 ticks!
            let mut cmds = [ClientInputCmd::default(); 16];
            for (i, c) in cmds.iter_mut().enumerate() {
                *c = ClientInputCmd {
                    cmd_tick: (i + 1) as u32,
                    buttons_stance: NT_BTN_SPRINT,
                    move_xy_q15: [32767, 0],
                    yaw_pitch_q15: [16384, 0],
                    fire_subtick_u8: 0,
                    weapon_slot_u8: 0,
                };
                nt_ingest_client_cmd(srv.ptr, 0, c);
                nt_step_kinematics(srv.ptr);
            }
            let mut srv_final = PlayerPoseSnapshot::default();
            nt_get_player_pose(srv.ptr, 0, &mut srv_final);

            // Snapshot client's Tier A & Tier B ring bytes BEFORE nt_client_reconcile
            let tier_a_before = (*cli.ptr).tier_a_cylinders;
            let tier_b_before = (*cli.ptr).tier_b_poses_plates;

            let mut pred1 = PlayerPoseSnapshot::default();
            let mut pred2 = PlayerPoseSnapshot::default();
            nt_client_reconcile(cli.ptr, 0, &auth_zero_stam, cmds.as_ptr(), 16, &mut pred1);
            nt_client_reconcile(cli.ptr, 0, &auth_zero_stam, cmds.as_ptr(), 16, &mut pred2);

            // 1. Strict purity on Tier A and Tier B rings + idempotence across repeated calls
            assert_eq!((*cli.ptr).tier_a_cylinders, tier_a_before, "nt_client_reconcile mutated Tier A!");
            assert_eq!((*cli.ptr).tier_b_poses_plates, tier_b_before, "nt_client_reconcile mutated Tier B!");
            assert_eq!(pred1, pred2, "nt_client_reconcile is not idempotent!");

            // 2. Exact 0-ULP lockstep with exhausted server player (did NOT reset 0 stamina to 32767!)
            assert_eq!(pred1.x.to_bits(), srv_final.x.to_bits());
            assert_eq!(pred1.y.to_bits(), srv_final.y.to_bits());
            assert_eq!(pred1.vy.to_bits(), srv_final.vy.to_bits());
            assert!(
                pred1.vy < 4.5f32,
                "Zero-stamina player erroneously sprinted at vy={}!",
                pred1.vy
            );
        }
    }

    #[test]
    fn test_v02_bug4_subtick_shortest_arc_yaw_and_lean_plate_interpolation() {
        // 1. Verify shortest modular i16 arc across ±pi seam: +32000 -> -32000 at alpha=0.5
        //    dyaw = (-32000i16).wrapping_sub(32000) = +1536 -> midpoint = 32768 (-32768, facing -X, NOT 0 facing +X!)
        let mid_yaw = ballistics::lerp_yaw_q15_shortest(32000, -32000, 0.5f32);
        assert!(
            mid_yaw <= -32700 || mid_yaw >= 32700,
            "Shortest-arc yaw across ±pi seam gave {} instead of ±32768!",
            mid_yaw
        );

        let arena = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena.ptr);
            // Shooter 0 at (-8.0, -5.0, 0.0) facing +X (yaw = 0) toward Victim 1 at (-2.0, -5.0, 0.0)
            nt_spawn_player(arena.ptr, 0, -8.0, -5.0, 0.0, 0);
            nt_spawn_player(arena.ptr, 1, -2.0, -5.0, 0.0, 32000);
            nt_step_kinematics(arena.ptr); // tick 1: Victim yaw = +32000, lean_q15 = 0

            // Advance to tick 2 and set Victim yaw = -32000 (flick across ±pi facing Shooter -X!) and full Q/E lean (4096)
            nt_step_kinematics(arena.ptr);
            (*arena.ptr).tier_b_poses_plates[1][1].yaw_q15 = 32000;
            (*arena.ptr).tier_b_poses_plates[1][1].lean_q15 = 0;
            (*arena.ptr).tier_b_poses_plates[2][1].yaw_q15 = -32000;
            (*arena.ptr).tier_b_poses_plates[2][1].lean_q15 = 4096;
            // Widen Tier A broadphase cylinder so leaned sub-tick pose is inside broadphase
            (*arena.ptr).tier_a_cylinders[1][1].rad_height = TierACylinder::pack_rad_height(600, 1800);
            (*arena.ptr).tier_a_cylinders[2][1].rad_height = TierACylinder::pack_rad_height(600, 1800);

            // At alpha = 128/256 = 0.5 between tick 1 and tick 2:
            // - Victim yaw is -32768 (facing -X directly AT Shooter 0 -> front upper thoracic plate zone 6 or 7, NOT back plate 8/9!)
            // - Victim lean_q15 is 2048 -> lean_shift_m = 0.10 m -> 0.65 * 0.10 = +0.065 m in +Y!
            let aim_y_offset = 0.065f32;
            let aim_rad = nt_atan2_f32(aim_y_offset, 6.0f32);
            let aim_yaw = ((aim_rad / core::f32::consts::PI) * 32768.0f32) as i16;
            let fire_cmd = ClientInputCmd {
                cmd_tick: 1,
                buttons_stance: NT_BTN_FIRE,
                move_xy_q15: [0, 0],
                yaw_pitch_q15: [aim_yaw, -500], // Upper chest height u_z ~ 0.68
                fire_subtick_u8: 128,           // alpha = 0.5
                weapon_slot_u8: 1,
            };
            let mut hit = HitRecordOut::default();
            nt_resolve_subtick_fire(arena.ptr, 0, &fire_cmd, &mut hit);
            assert_eq!(hit.victim_id, 1);
            assert_ne!(hit.outcome_flags & NT_HIT_FLAG_HIT, 0);
            assert!(
                hit.plate_zone_id == 6 || hit.plate_zone_id == 7,
                "Expected front thoracic plate (6 or 7) across ±pi seam, got zone {}!",
                hit.plate_zone_id
            );
        }
    }

    #[test]
    fn test_v02_bug5_simd_override_per_arena_isolation() {
        let arena_a = OwnedArena::new_initialized();
        let arena_b = OwnedArena::new_initialized();
        unsafe {
            assert_eq!(nt_set_simd_backend_override(core::ptr::null_mut(), 0), 0);
            assert_eq!(GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed), 0);

            // Setting arena_b to forced scalar MUST NOT mutate GLOBAL_SIMD_OVERRIDE or arena_a!
            assert_eq!(nt_set_simd_backend_override(arena_b.ptr, 1), 0);
            assert_eq!((*arena_b.ptr).telemetry_and_header.simd_mode_override, 1);
            assert_eq!((*arena_a.ptr).telemetry_and_header.simd_mode_override, 0);
            assert_eq!(
                GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed),
                0,
                "Non-null arena override polluted GLOBAL_SIMD_OVERRIDE!"
            );

            // Null arena pointer toggles GLOBAL_SIMD_OVERRIDE cleanly
            assert_eq!(nt_set_simd_backend_override(core::ptr::null_mut(), 1), 0);
            assert_eq!(GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed), 1);
            assert_eq!(nt_set_simd_backend_override(core::ptr::null_mut(), 0), 0);
            assert_eq!(GLOBAL_SIMD_OVERRIDE.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_v02_voxel_dda_vs_pure_bvh_ray_exact_equivalence() {
        let arena = OwnedArena::new_initialized();
        unsafe {
            nt_load_default_raid_map(arena.ptr);
            let map = &(*arena.ptr).static_voxel_bvh;

            let mut rng: u64 = 0x5EED_DDA0_2026_0200;
            let mut hits_verified = 0u32;
            let mut misses_verified = 0u32;

            for case in 0..5_000u32 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let ox = (((rng & 0xFFFF) as f32) / 65535.0f32) * 60.0f32 - 30.0f32;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let oy = (((rng & 0xFFFF) as f32) / 65535.0f32) * 60.0f32 - 30.0f32;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let oz = (((rng & 0xFFFF) as f32) / 65535.0f32) * 3.8f32 + 0.05f32;

                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let (dx, dy, dz) = if case < 200 {
                    // Axis-aligned and grazing cell-boundary stress rays
                    match case % 6 {
                        0 => (1.0f32, 0.0f32, 0.0f32),
                        1 => (-1.0f32, 0.0f32, 0.0f32),
                        2 => (0.0f32, 1.0f32, 0.0f32),
                        3 => (0.0f32, -1.0f32, 0.0f32),
                        4 => (0.0f32, 0.0f32, 1.0f32),
                        _ => (0.99995f32, 0.00999f32, 0.0f32),
                    }
                } else {
                    let rx = (((rng & 0xFFFF) as f32) / 32767.5f32) - 1.0f32;
                    let ry = ((((rng >> 16) & 0xFFFF) as f32) / 32767.5f32) - 1.0f32;
                    let rz = ((((rng >> 32) & 0xFFFF) as f32) / 32767.5f32 - 1.0f32) * 0.35f32;
                    let len = nt_sqrt_f32((rx * rx + ry * ry) + rz * rz).max(1e-4f32);
                    (rx / len, ry / len, rz / len)
                };

                let max_dist = 0.5f32 + (((rng >> 48) as u16 as f32) / 65535.0f32) * 55.0f32;
                let origin = [ox, oy, oz];
                let dir = [dx, dy, dz];
                let inv_dir = map_bvh::safe_inv_dir(&dir);

                let dda_res = map_bvh::raycast_hard_cover(map, &origin, &dir, max_dist);
                let bvh_res = map_bvh::raycast_hard_cover_bvh(map, &origin, &inv_dir, max_dist);

                match (dda_res, bvh_res) {
                    (Some((t1, n1)), Some((t2, n2))) => {
                        assert_eq!(
                            t1.to_bits(),
                            t2.to_bits(),
                            "Voxel DDA vs BVH t_hit mismatch at case {}: {} vs {}",
                            case,
                            t1,
                            t2
                        );
                        assert_eq!(n1, n2, "Voxel DDA vs BVH normal mismatch at case {}", case);
                        hits_verified += 1;
                    }
                    (None, None) => {
                        misses_verified += 1;
                    }
                    _ => panic!(
                        "Voxel DDA vs pure BVH hit/miss disagreement at case {}: origin={:?} dir={:?} max_dist={} dda={:?} bvh={:?}",
                        case, origin, dir, max_dist, dda_res, bvh_res
                    ),
                }
            }
            assert!(hits_verified > 1000 && misses_verified > 500);
        }
    }
}
