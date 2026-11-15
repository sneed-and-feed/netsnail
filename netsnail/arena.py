"""Zero-copy 64-byte-aligned ctypes FFI bridge to netsnail_rs (#![no_std] NetTwinArena)."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
from pathlib import Path
import platform
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# 1. Frozen Compile-Time Arena & Protocol Constants (matches include/net_twin.h)
# ==============================================================================

NT_ARENA_BYTES: int = 1_572_864
NT_ARENA_ALIGN: int = 64
NT_REWIND_RING_BYTES: int = 327_680  # 320 KB = Tier A (64 KB) + Tier B (256 KB)
NT_TICK_HZ: int = 128
NT_RING_TICKS: int = 64
NT_MAX_PLAYERS: int = 64
NT_MAX_PROJECTILES: int = 256
NT_PLATE_ZONES: int = 30
NT_MTU_CEILING_BYTES: int = 1200
NT_WASM_MTU_BUF_BYTES: int = 1536
NT_MAGIC_TAG: int = 0x4E545731

# Aliases for verifier cross-language offset checks
NET_TWIN_ARENA_SIZE_BYTES: int = NT_ARENA_BYTES
NET_TWIN_ARENA_ALIGN_BYTES: int = NT_ARENA_ALIGN

# Exact 8-Partition Byte Offsets in NetTwinArena
NT_OFFSET_TIER_A_CYLINDERS: int = 0
NT_OFFSET_TIER_B_POSES_PLATES: int = 65_536
NT_OFFSET_STATIC_VOXEL_BVH: int = 327_680
NT_OFFSET_CLIENT_JITTER_GATES: int = 589_824
NT_OFFSET_PROJECTILES_ACOUSTIC_SLAB: int = 655_360
NT_OFFSET_CLIENT_BASELINE_SNAPSHOTS: int = 720_896
NT_OFFSET_TRIAD_CASCADE_SCRATCH: int = 1_245_184
NT_OFFSET_TELEMETRY_AND_HEADER: int = 1_507_328

# Exact 8-Partition Byte Sizes in NetTwinArena
NT_SIZE_TIER_A_CYLINDERS: int = 65_536
NT_SIZE_TIER_B_POSES_PLATES: int = 262_144
NT_SIZE_STATIC_VOXEL_BVH: int = 262_144
NT_SIZE_CLIENT_JITTER_GATES: int = 65_536
NT_SIZE_PROJECTILES_ACOUSTIC_SLAB: int = 65_536
NT_SIZE_CLIENT_BASELINE_SNAPSHOTS: int = 524_288
NT_SIZE_TRIAD_CASCADE_SCRATCH: int = 262_144
NT_SIZE_TELEMETRY_AND_HEADER: int = 65_536

NET_TWIN_PARTITION_OFFSETS: Tuple[int, ...] = (
    NT_OFFSET_TIER_A_CYLINDERS,
    NT_OFFSET_TIER_B_POSES_PLATES,
    NT_OFFSET_STATIC_VOXEL_BVH,
    NT_OFFSET_CLIENT_JITTER_GATES,
    NT_OFFSET_PROJECTILES_ACOUSTIC_SLAB,
    NT_OFFSET_CLIENT_BASELINE_SNAPSHOTS,
    NT_OFFSET_TRIAD_CASCADE_SCRATCH,
    NT_OFFSET_TELEMETRY_AND_HEADER,
)

NET_TWIN_PARTITION_SIZES: Tuple[int, ...] = (
    NT_SIZE_TIER_A_CYLINDERS,
    NT_SIZE_TIER_B_POSES_PLATES,
    NT_SIZE_STATIC_VOXEL_BVH,
    NT_SIZE_CLIENT_JITTER_GATES,
    NT_SIZE_PROJECTILES_ACOUSTIC_SLAB,
    NT_SIZE_CLIENT_BASELINE_SNAPSHOTS,
    NT_SIZE_TRIAD_CASCADE_SCRATCH,
    NT_SIZE_TELEMETRY_AND_HEADER,
)

# ClientInputCmd::buttons_stance bitmask flags
NT_BTN_FIRE: int = 1 << 0
NT_BTN_ADS: int = 1 << 1
NT_BTN_CROUCH: int = 1 << 2
NT_BTN_SPRINT: int = 1 << 3
NT_BTN_LEAN_L: int = 1 << 4
NT_BTN_LEAN_R: int = 1 << 5

# HitRecordOut::outcome_flags bitmask flags
NT_HIT_FLAG_HIT: int = 1 << 0                       # 0x01
NT_HIT_FLAG_PENETRATED: int = 1 << 1                # 0x02
NT_HIT_FLAG_RICOCHET: int = 1 << 2                  # 0x04
NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED: int = 1 << 3  # 0x08
NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED: int = 1 << 4 # 0x10

# SIMD Backend Override Modes
NT_SIMD_MODE_AUTO: int = 0
NT_SIMD_MODE_FORCE_SCALAR: int = 1

# Morse Arrival-Jitter Gate Status Codes
NT_GATE_NOMINAL: int = 0
NT_GATE_PENDING_IMPULSE: int = 1
NT_GATE_STEP_MANEUVER: int = 2
NT_GATE_GLINT_REJECTED: int = 3
NT_GATE_LAG_SWITCH_QUARANTINED: int = 4


# ==============================================================================
# 2. Frozen C-ABI Wire & Arena ctypes Structures (Zero Implicit Padding)
# ==============================================================================

class ClientInputCmd(ctypes.Structure):
    """16-byte client input command."""

    _fields_ = [
        ("cmd_tick", ctypes.c_uint32),            # 0..4
        ("buttons_stance", ctypes.c_uint16),      # 4..6
        ("move_xy_q15", ctypes.c_int16 * 2),      # 6..10
        ("yaw_pitch_q15", ctypes.c_int16 * 2),    # 10..14
        ("fire_subtick_u8", ctypes.c_uint8),      # 14..15
        ("weapon_slot_u8", ctypes.c_uint8),      # 15..16
    ]


class HitRecordOut(ctypes.Structure):
    """32-byte server authoritative sub-tick hit & armor evaluation record."""

    _fields_ = [
        ("shooter_id", ctypes.c_uint32),          # 0..4
        ("victim_id", ctypes.c_uint32),           # 4..8
        ("rewind_tick", ctypes.c_uint32),         # 8..12
        ("plate_zone_id", ctypes.c_uint16),       # 12..14
        ("outcome_flags", ctypes.c_uint8),        # 14..15
        ("_pad0", ctypes.c_uint8),                # 15..16
        ("subtick_alpha", ctypes.c_float),        # 16..20
        ("damage_applied", ctypes.c_float),       # 20..24
        ("armor_damage", ctypes.c_float),         # 24..28
        ("hit_dist_m", ctypes.c_float),           # 28..32
    ]


class AcousticImpulse(ctypes.Structure):
    """12-byte anonymous portal-diffracted acoustic impulse (zero entity_id, zero xyz)."""

    _fields_ = [
        ("arrival_tick", ctypes.c_uint32),        # 0..4
        ("azimuth_q12", ctypes.c_int16),          # 4..6
        ("elevation_q12", ctypes.c_int16),        # 6..8
        ("band_energy_q8", ctypes.c_uint8 * 4),   # 8..12
    ]


class PlayerPoseSnapshot(ctypes.Structure):
    """32-byte player kinematic & attitude readback snapshot."""

    _fields_ = [
        ("x", ctypes.c_float),                    # 0..4
        ("y", ctypes.c_float),                    # 4..8
        ("z", ctypes.c_float),                    # 8..12
        ("vx", ctypes.c_float),                   # 12..16
        ("vy", ctypes.c_float),                   # 16..20
        ("vz", ctypes.c_float),                   # 20..24
        ("yaw_q15", ctypes.c_int16),              # 24..26
        ("pitch_q15", ctypes.c_int16),            # 26..28
        ("lean_q15", ctypes.c_int16),             # 28..30
        ("stance_flags", ctypes.c_uint16),        # 30..32
    ]


class TierACylinder(ctypes.Structure):
    """16-byte Tier-A swept broadphase cylinder."""

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z_base", ctypes.c_float),
        ("rad_height", ctypes.c_uint32),
    ]


class PlayerTickState(ctypes.Structure):
    """64-byte Tier-B pose & 30-zone multi-plate armor state."""

    _fields_ = [
        ("pos", ctypes.c_float * 3),              # 0..12
        ("vel", ctypes.c_float * 3),              # 12..24
        ("yaw_q15", ctypes.c_int16),              # 24..26
        ("pitch_q15", ctypes.c_int16),            # 26..28
        ("lean_q15", ctypes.c_int16),             # 28..30
        ("stance_flags", ctypes.c_uint16),        # 30..32
        ("cayley_gen_q15", ctypes.c_int16 * 3),   # 32..38
        ("stamina_q15", ctypes.c_uint16),         # 38..40
        ("health_q8", ctypes.c_uint16),           # 40..42
        ("team_id", ctypes.c_uint8),              # 42..43
        ("weapon_id", ctypes.c_uint8),            # 43..44
        ("plate_intact_mask", ctypes.c_uint32),   # 44..48
        ("plate_material_bits", ctypes.c_uint64), # 48..56
        ("plate_hp_pools_q8", ctypes.c_uint16 * 4),  # 56..64
    ]


class ClientJitterGate(ctypes.Structure):
    """1024-byte per-player 3-point Morse arrival-jitter & lag-switch gate."""

    _fields_ = [
        ("ack_mask_u64", ctypes.c_uint64),              # 0..8
        ("latest_cmd_tick", ctypes.c_uint32),           # 8..12
        ("last_arrival_server_tick", ctypes.c_uint32),  # 12..16
        ("est_rtt_ticks", ctypes.c_uint32),             # 16..20
        ("gate_status", ctypes.c_uint32),               # 20..24
        ("morse_j_prev2", ctypes.c_float),              # 24..28
        ("morse_j_prev1", ctypes.c_float),              # 28..32
        ("morse_j_curr", ctypes.c_float),               # 32..36
        ("morse_pending_j", ctypes.c_float),            # 36..40
        ("morse_impulse_last", ctypes.c_float),         # 40..44
        ("jitter_var_ema", ctypes.c_float),             # 44..48
        ("acklam_pullback_z", ctypes.c_float),          # 48..52
        ("h_clock_burst_score", ctypes.c_float),        # 52..56
        ("quarantine_ticks_remaining", ctypes.c_uint32),# 56..60
        ("clamped_rewind_max_ticks", ctypes.c_uint32),  # 60..64
        ("total_cmds_ingested", ctypes.c_uint32),       # 64..68
        ("total_burst_clumps", ctypes.c_uint32),        # 68..72
        ("total_quarantine_triggers", ctypes.c_uint32), # 72..76
        ("hist_head", ctypes.c_uint32),                 # 76..80
        ("cmd_ring", ClientInputCmd * 32),              # 80..592
        ("arrival_skew_hist", ctypes.c_float * 64),     # 592..848
        ("reserved", ctypes.c_uint8 * 176),             # 848..1024
    ]


class TelemetryAndHeader(ctypes.Structure):
    """65536-byte telemetry & arena control header (offset 1,507,328)."""

    _fields_ = [
        ("server_tick", ctypes.c_uint32),               # 0..4
        ("simd_mode_override", ctypes.c_uint32),        # 4..8
        ("active_player_mask", ctypes.c_uint64),        # 8..16
        ("state_digest_u64", ctypes.c_uint64),          # 16..24
        ("total_steps", ctypes.c_uint32),               # 24..28
        ("total_shots_resolved", ctypes.c_uint32),      # 28..32
        ("total_hits_confirmed", ctypes.c_uint32),      # 32..36
        ("total_occluder_cutoffs", ctypes.c_uint32),    # 36..40
        ("total_jitter_quarantines", ctypes.c_uint32),  # 40..44
        ("total_ricochets", ctypes.c_uint32),           # 44..48
        ("total_penetrations", ctypes.c_uint32),        # 48..52
        ("last_datagram_bytes", ctypes.c_uint32),       # 52..56
        ("last_pvs_visible_count", ctypes.c_uint32),    # 56..60
        ("last_acoustic_impulses", ctypes.c_uint32),    # 60..64
        ("max_cayley_norm_drift", ctypes.c_float),      # 64..68
        ("last_reconcile_ticks", ctypes.c_uint32),      # 68..72
        ("active_projectile_count", ctypes.c_uint32),   # 72..76
        ("magic_tag", ctypes.c_uint32),                 # 76..80
        ("reserved_tail", ctypes.c_uint8 * 65_456),     # 80..65536
    ]


class NetTwinArena(ctypes.Structure):
    """Exact 1,572,864-byte (1.50 MB) static arena layout."""

    _fields_ = [
        ("tier_a_cylinders", (TierACylinder * NT_MAX_PLAYERS) * NT_RING_TICKS),
        ("tier_b_poses_plates", (PlayerTickState * NT_MAX_PLAYERS) * NT_RING_TICKS),
        ("static_voxel_bvh", ctypes.c_uint8 * NT_SIZE_STATIC_VOXEL_BVH),
        ("client_jitter_gates", ClientJitterGate * NT_MAX_PLAYERS),
        ("projectiles_acoustic_slab", ctypes.c_uint8 * NT_SIZE_PROJECTILES_ACOUSTIC_SLAB),
        ("client_baseline_snapshots", ctypes.c_uint8 * NT_SIZE_CLIENT_BASELINE_SNAPSHOTS),
        ("triad_cascade_scratch", ctypes.c_uint8 * NT_SIZE_TRIAD_CASCADE_SCRATCH),
        ("telemetry_and_header", TelemetryAndHeader),
    ]


# Enforce import-time C-ABI layout invariants identical to C11 _Static_asserts
assert ctypes.sizeof(ClientInputCmd) == 16, f"ClientInputCmd size {ctypes.sizeof(ClientInputCmd)} != 16"
assert ctypes.sizeof(HitRecordOut) == 32, f"HitRecordOut size {ctypes.sizeof(HitRecordOut)} != 32"
assert ctypes.sizeof(AcousticImpulse) == 12, f"AcousticImpulse size {ctypes.sizeof(AcousticImpulse)} != 12"
assert ctypes.sizeof(PlayerPoseSnapshot) == 32, f"PlayerPoseSnapshot size {ctypes.sizeof(PlayerPoseSnapshot)} != 32"
assert ctypes.sizeof(TierACylinder) == 16, f"TierACylinder size {ctypes.sizeof(TierACylinder)} != 16"
assert ctypes.sizeof(PlayerTickState) == 64, f"PlayerTickState size {ctypes.sizeof(PlayerTickState)} != 64"
assert ctypes.sizeof(ClientJitterGate) == 1024, f"ClientJitterGate size {ctypes.sizeof(ClientJitterGate)} != 1024"
assert ctypes.sizeof(TelemetryAndHeader) == 65_536, f"TelemetryAndHeader size {ctypes.sizeof(TelemetryAndHeader)} != 65536"
assert ctypes.sizeof(NetTwinArena) == NT_ARENA_BYTES, f"NetTwinArena size {ctypes.sizeof(NetTwinArena)} != 1572864"

assert NetTwinArena.tier_a_cylinders.offset == NT_OFFSET_TIER_A_CYLINDERS == 0
assert NetTwinArena.tier_b_poses_plates.offset == NT_OFFSET_TIER_B_POSES_PLATES == 65_536
assert NetTwinArena.static_voxel_bvh.offset == NT_OFFSET_STATIC_VOXEL_BVH == 327_680
assert NetTwinArena.client_jitter_gates.offset == NT_OFFSET_CLIENT_JITTER_GATES == 589_824
assert NetTwinArena.projectiles_acoustic_slab.offset == NT_OFFSET_PROJECTILES_ACOUSTIC_SLAB == 655_360
assert NetTwinArena.client_baseline_snapshots.offset == NT_OFFSET_CLIENT_BASELINE_SNAPSHOTS == 720_896
assert NetTwinArena.triad_cascade_scratch.offset == NT_OFFSET_TRIAD_CASCADE_SCRATCH == 1_245_184
assert NetTwinArena.telemetry_and_header.offset == NT_OFFSET_TELEMETRY_AND_HEADER == 1_507_328


# ==============================================================================
# 3. Shared Library & WASM32 Discovery + C-ABI Signature Binding
# ==============================================================================

_LIB_CACHE: Optional[ctypes.CDLL] = None
_LIB_PATH_CACHE: Optional[Path] = None


def find_rust_library(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Locate compiled netsnail_rs shared library (.dll / .so / .dylib)."""
    if explicit_path:
        p = Path(explicit_path)
        return p.resolve() if p.is_file() else None

    system = platform.system()
    if system == "Windows":
        lib_name = "netsnail_rs.dll"
    elif system == "Darwin":
        lib_name = "libnetsnail_rs.dylib"
    else:
        lib_name = "libnetsnail_rs.so"

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent

    candidates = [
        repo_root / "target" / "release" / lib_name,
        repo_root / "rust" / "netsnail_rs" / "target" / "release" / lib_name,
        repo_root / "target" / "debug" / lib_name,
        repo_root / "rust" / "netsnail_rs" / "target" / "debug" / lib_name,
        pkg_dir / lib_name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def find_wasm_library(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Locate compiled wasm32-unknown-unknown netsnail_rs.wasm module."""
    if explicit_path:
        p = Path(explicit_path)
        return p.resolve() if p.is_file() else None

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent

    candidates = [
        repo_root / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
        repo_root / "rust" / "netsnail_rs" / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
        pkg_dir / "netsnail_rs.wasm",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _bind_c_abi_signatures(lib: ctypes.CDLL) -> None:
    """Bind exact C-ABI argtypes and restypes matching include/net_twin.h."""
    # Lifecycle, map & player management
    lib.nt_arena_init.argtypes = [ctypes.c_void_p]
    lib.nt_arena_init.restype = ctypes.c_int32

    lib.nt_set_simd_backend_override.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.nt_set_simd_backend_override.restype = ctypes.c_int32

    lib.nt_load_default_raid_map.argtypes = [ctypes.c_void_p]
    lib.nt_load_default_raid_map.restype = ctypes.c_int32

    lib.nt_spawn_player.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int16,
    ]
    lib.nt_spawn_player.restype = ctypes.c_int32

    lib.nt_get_player_pose.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(PlayerPoseSnapshot),
    ]
    lib.nt_get_player_pose.restype = ctypes.c_int32

    if hasattr(lib, "nt_set_player_rtt_ticks"):
        lib.nt_set_player_rtt_ticks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.nt_set_player_rtt_ticks.restype = ctypes.c_int32

    if hasattr(lib, "nt_emit_acoustic_event"):
        lib.nt_emit_acoustic_event.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.nt_emit_acoustic_event.restype = ctypes.c_int32

    # Tick ingestion, swept kinematics, client reconciliation & sub-tick rewind
    lib.nt_ingest_client_cmd.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ClientInputCmd),
    ]
    lib.nt_ingest_client_cmd.restype = ctypes.c_int32

    lib.nt_step_kinematics.argtypes = [ctypes.c_void_p]
    lib.nt_step_kinematics.restype = ctypes.c_int32

    lib.nt_client_reconcile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(PlayerPoseSnapshot),
        ctypes.POINTER(ClientInputCmd),
        ctypes.c_uint32,
        ctypes.POINTER(PlayerPoseSnapshot),
    ]
    lib.nt_client_reconcile.restype = ctypes.c_int32

    lib.nt_resolve_subtick_fire.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ClientInputCmd),
        ctypes.POINTER(HitRecordOut),
    ]
    lib.nt_resolve_subtick_fire.restype = ctypes.c_int32

    # Minkowski Reachability PVS + Anonymous Acoustic + <1200B Delta Bitpacker
    lib.nt_pack_client_datagram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
    ]
    lib.nt_pack_client_datagram.restype = ctypes.c_int32

    lib.nt_unpack_client_datagram.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.POINTER(AcousticImpulse),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.nt_unpack_client_datagram.restype = ctypes.c_int32

    lib.nt_hash_arena_deterministic.argtypes = [ctypes.c_void_p]
    lib.nt_hash_arena_deterministic.restype = ctypes.c_uint64

    # Cross-ISA 0-ULP Tier-0 Numerical Kernels
    lib.nt_sin_q15.argtypes = [ctypes.c_int16]
    lib.nt_sin_q15.restype = ctypes.c_float

    lib.nt_cos_q15.argtypes = [ctypes.c_int16]
    lib.nt_cos_q15.restype = ctypes.c_float

    lib.nt_atan2_f32.argtypes = [ctypes.c_float, ctypes.c_float]
    lib.nt_atan2_f32.restype = ctypes.c_float

    lib.nt_sqrt_f32.argtypes = [ctypes.c_float]
    lib.nt_sqrt_f32.restype = ctypes.c_float

    lib.nt_dot16_pairwise_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.nt_dot16_pairwise_f32.restype = ctypes.c_float

    lib.nt_cayley_rotor_so3_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.nt_cayley_rotor_so3_f32.restype = ctypes.c_int32

    # Static bridge helpers (exported on both native and WASM32)
    for bridge_sym in (
        "nt_wasm_global_arena",
        "nt_wasm_cmd_buf",
        "nt_wasm_mtu_buf",
        "nt_wasm_hit_buf",
        "nt_wasm_acoustic_buf",
    ):
        if hasattr(lib, bridge_sym):
            fn = getattr(lib, bridge_sym)
            fn.argtypes = []
            fn.restype = ctypes.c_void_p


def load_native_library(explicit_path: Optional[str] = None, reload_lib: bool = False) -> ctypes.CDLL:
    """Load and cache the compiled netsnail_rs shared library."""
    global _LIB_CACHE, _LIB_PATH_CACHE
    if _LIB_CACHE is not None and not reload_lib and explicit_path is None:
        return _LIB_CACHE

    lib_path = find_rust_library(explicit_path)
    if lib_path is None:
        raise FileNotFoundError(
            "Compiled netsnail_rs shared library not found. "
            "Run `cargo build --release` first."
        )

    lib = ctypes.CDLL(str(lib_path))
    _bind_c_abi_signatures(lib)
    if explicit_path is None:
        _LIB_CACHE = lib
        _LIB_PATH_CACHE = lib_path
    return lib


def is_native_available() -> bool:
    try:
        load_native_library()
        return True
    except Exception:
        return False


# ==============================================================================
# 4. Hardware x86_64 MXCSR Flush-To-Zero Hostility Context Manager
# ==============================================================================

@contextmanager
def hostile_mxcsr_flush_to_zero() -> Iterator[Dict[str, Any]]:
    """Temporarily sets caller-side x86_64 MXCSR Flush-To-Zero (_DN_FLUSH) if supported.

    Allows red-team tests to verify that netsnail_rs sanitizes MXCSR (FTZ=0, DAZ=0, RN)
    on C-ABI entry and restores the caller's control word upon exit.
    """
    info: Dict[str, Any] = {"supported": False, "before": None, "during": None, "restored": None}
    if platform.system() == "Windows" and platform.machine().lower() in ("amd64", "x86_64"):
        try:
            ucrt = ctypes.CDLL("ucrtbase")
            controlfp_s = ucrt._controlfp_s
            controlfp_s.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_uint32]
            controlfp_s.restype = ctypes.c_int

            MCW_DN = 0x03000000
            DN_FLUSH = 0x01000000

            orig_cw = ctypes.c_uint32(0)
            if controlfp_s(ctypes.byref(orig_cw), 0, 0) == 0:
                new_cw = ctypes.c_uint32(0)
                if controlfp_s(ctypes.byref(new_cw), DN_FLUSH, MCW_DN) == 0:
                    info["supported"] = True
                    info["before"] = int(orig_cw.value)
                    info["during"] = int(new_cw.value)
                    try:
                        yield info
                    finally:
                        after_call = ctypes.c_uint32(0)
                        controlfp_s(ctypes.byref(after_call), 0, 0)
                        info["after_ffi_call"] = int(after_call.value)
                        restored_cw = ctypes.c_uint32(0)
                        controlfp_s(ctypes.byref(restored_cw), orig_cw.value & MCW_DN, MCW_DN)
                        info["restored"] = int(restored_cw.value)
                    return
        except Exception:
            pass
    yield info


# ==============================================================================
# 5. Official Spec-Compliant Vectorized BLAKE3-256 (Zero External Pip Deps)
# ==============================================================================

_B3_IV = np.array(
    [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ],
    dtype=np.uint32,
)

_B3_MSG_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

_B3_CHUNK_START = np.uint32(1 << 0)
_B3_CHUNK_END = np.uint32(1 << 1)
_B3_PARENT = np.uint32(1 << 2)
_B3_ROOT = np.uint32(1 << 3)


def _b3_rotr_vec(x: np.ndarray, r: int) -> np.ndarray:
    return (x >> np.uint32(r)) | (x << np.uint32(32 - r))


def _b3_g_vec(
    s: List[np.ndarray],
    a: int,
    b: int,
    c: int,
    d: int,
    mx: np.ndarray,
    my: np.ndarray,
) -> None:
    s[a] = s[a] + s[b] + mx
    s[d] = _b3_rotr_vec(s[d] ^ s[a], 16)
    s[c] = s[c] + s[d]
    s[b] = _b3_rotr_vec(s[b] ^ s[c], 12)
    s[a] = s[a] + s[b] + my
    s[d] = _b3_rotr_vec(s[d] ^ s[a], 8)
    s[c] = s[c] + s[d]
    s[b] = _b3_rotr_vec(s[b] ^ s[c], 7)


def _b3_compress_batch(
    cv: np.ndarray,          # shape (N, 8) uint32
    block_words: np.ndarray, # shape (N, 16) uint32
    counter_lo: np.ndarray,  # shape (N,) uint32
    counter_hi: np.ndarray,  # shape (N,) uint32
    block_len: np.ndarray,   # shape (N,) uint32
    flags: np.ndarray,       # shape (N,) uint32
) -> np.ndarray:
    """Vectorized 7-round BLAKE3 compression returning 8-word chaining values (N, 8)."""
    n = cv.shape[0]
    s: List[np.ndarray] = [
        cv[:, 0].copy(),
        cv[:, 1].copy(),
        cv[:, 2].copy(),
        cv[:, 3].copy(),
        cv[:, 4].copy(),
        cv[:, 5].copy(),
        cv[:, 6].copy(),
        cv[:, 7].copy(),
        np.full(n, _B3_IV[0], dtype=np.uint32),
        np.full(n, _B3_IV[1], dtype=np.uint32),
        np.full(n, _B3_IV[2], dtype=np.uint32),
        np.full(n, _B3_IV[3], dtype=np.uint32),
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ]
    m: List[np.ndarray] = [block_words[:, i] for i in range(16)]

    for r in range(7):
        _b3_g_vec(s, 0, 4, 8, 12, m[0], m[1])
        _b3_g_vec(s, 1, 5, 9, 13, m[2], m[3])
        _b3_g_vec(s, 2, 6, 10, 14, m[4], m[5])
        _b3_g_vec(s, 3, 7, 11, 15, m[6], m[7])
        _b3_g_vec(s, 0, 5, 10, 15, m[8], m[9])
        _b3_g_vec(s, 1, 6, 11, 12, m[10], m[11])
        _b3_g_vec(s, 2, 7, 8, 13, m[12], m[13])
        _b3_g_vec(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[idx] for idx in _B3_MSG_PERM]

    out = np.empty((n, 8), dtype=np.uint32)
    for i in range(8):
        out[:, i] = s[i] ^ s[i + 8]
    return out


def _b3_reduce_pow2_subtree(cvs: np.ndarray) -> np.ndarray:
    """Reduces a power-of-two array of 8-word chaining values (2^k, 8) down to (1, 8)."""
    cur = cvs
    while cur.shape[0] > 1:
        half = cur.shape[0] // 2
        blocks = cur.reshape(half, 16)
        iv_batch = np.broadcast_to(_B3_IV, (half, 8)).copy()
        zeros = np.zeros(half, dtype=np.uint32)
        blen = np.full(half, 64, dtype=np.uint32)
        flgs = np.full(half, _B3_PARENT, dtype=np.uint32)
        cur = _b3_compress_batch(iv_batch, blocks, zeros, zeros, blen, flgs)
    return cur


def blake3_256_hex(data: bytes | memoryview | np.ndarray) -> str:
    """Computes exact official BLAKE3-256 hex digest (fast vectorized path for >=1024B multiples)."""
    try:
        import blake3 as _blake3_ext  # type: ignore
        return str(_blake3_ext.blake3(data).hexdigest())
    except ImportError:
        pass

    arr = np.frombuffer(data, dtype=np.uint8)
    total_len = int(arr.size)

    if total_len > 1024 and (total_len % 1024 == 0):
        num_chunks = total_len // 1024
        # Reshape into (num_chunks, 16 blocks, 16 uint32 words) little-endian
        words = arr.view("<u4").reshape(num_chunks, 16, 16).astype(np.uint32, copy=False)
        cv = np.broadcast_to(_B3_IV, (num_chunks, 8)).copy()
        t_lo = np.arange(num_chunks, dtype=np.uint32)
        t_hi = np.zeros(num_chunks, dtype=np.uint32)
        blen = np.full(num_chunks, 64, dtype=np.uint32)
        fl_start = np.full(num_chunks, _B3_CHUNK_START, dtype=np.uint32)
        fl_mid = np.zeros(num_chunks, dtype=np.uint32)
        fl_end = np.full(num_chunks, _B3_CHUNK_END, dtype=np.uint32)

        cv = _b3_compress_batch(cv, words[:, 0, :], t_lo, t_hi, blen, fl_start)
        for b in range(1, 15):
            cv = _b3_compress_batch(cv, words[:, b, :], t_lo, t_hi, blen, fl_mid)
        cv = _b3_compress_batch(cv, words[:, 15, :], t_lo, t_hi, blen, fl_end)

        # Official BLAKE3 left-heavy binary tree decomposition by set bits of num_chunks
        subtrees: List[np.ndarray] = []
        idx = 0
        rem = num_chunks
        while rem > 0:
            # Largest power of 2 <= rem
            p2 = 1 << (rem.bit_length() - 1)
            sub_cv = cv[idx : idx + p2]
            if p2 == num_chunks:
                # num_chunks itself is a power of 2: reduce down to last 2 CVs and merge with PARENT|ROOT
                while sub_cv.shape[0] > 2:
                    half = sub_cv.shape[0] // 2
                    sub_cv = _b3_compress_batch(
                        np.broadcast_to(_B3_IV, (half, 8)).copy(),
                        sub_cv.reshape(half, 16),
                        np.zeros(half, dtype=np.uint32),
                        np.zeros(half, dtype=np.uint32),
                        np.full(half, 64, dtype=np.uint32),
                        np.full(half, _B3_PARENT, dtype=np.uint32),
                    )
                root_cv = _b3_compress_batch(
                    _B3_IV.reshape(1, 8).copy(),
                    sub_cv.reshape(1, 16),
                    np.zeros(1, dtype=np.uint32),
                    np.zeros(1, dtype=np.uint32),
                    np.array([64], dtype=np.uint32),
                    np.array([_B3_PARENT | _B3_ROOT], dtype=np.uint32),
                )
                return root_cv.astype("<u4").tobytes().hex()
            subtrees.append(_b3_reduce_pow2_subtree(sub_cv))
            idx += p2
            rem -= p2

        while len(subtrees) > 1:
            right = subtrees.pop()
            left = subtrees.pop()
            is_root = len(subtrees) == 0
            fl = _B3_PARENT | (_B3_ROOT if is_root else np.uint32(0))
            merged = _b3_compress_batch(
                _B3_IV.reshape(1, 8).copy(),
                np.concatenate([left, right], axis=1),
                np.zeros(1, dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
                np.array([64], dtype=np.uint32),
                np.array([fl], dtype=np.uint32),
            )
            subtrees.append(merged)
        return subtrees[0].astype("<u4").tobytes().hex()

    # General <=1024B single-chunk path
    num_blocks = max(1, (total_len + 63) // 64)
    cv = _B3_IV.reshape(1, 8).copy()
    zero1 = np.zeros(1, dtype=np.uint32)
    for b in range(num_blocks):
        start = b * 64
        end = min(total_len, start + 64)
        blk_bytes = bytearray(64)
        if end > start:
            blk_bytes[: end - start] = arr[start:end].tobytes()
        blk_words = np.frombuffer(blk_bytes, dtype="<u4").astype(np.uint32).reshape(1, 16)
        fl = np.uint32(0)
        if b == 0:
            fl |= _B3_CHUNK_START
        if b == num_blocks - 1:
            fl |= _B3_CHUNK_END | _B3_ROOT
        cv = _b3_compress_batch(
            cv,
            blk_words,
            zero1,
            zero1,
            np.array([end - start], dtype=np.uint32),
            np.array([fl], dtype=np.uint32),
        )
    return cv.astype("<u4").tobytes().hex()


# ==============================================================================
# 6. Zero-Copy 64-Byte Aligned NetTwinArena Python Bridge
# ==============================================================================

class NetTwinArenaBridge:
    """Zero-copy 64-byte aligned ctypes & NumPy wrapper over NetTwinArena (1,572,864 B)."""

    def __init__(self, explicit_lib_path: Optional[str] = None, auto_init: bool = True) -> None:
        self._lib = load_native_library(explicit_lib_path)

        # Over-allocate by 64 bytes to guarantee exact 64-byte hardware cache-line alignment
        self._raw_owner = (ctypes.c_uint8 * (NT_ARENA_BYTES + NT_ARENA_ALIGN))()
        raw_addr = ctypes.addressof(self._raw_owner)
        self.aligned_address: int = (raw_addr + (NT_ARENA_ALIGN - 1)) & ~(NT_ARENA_ALIGN - 1)
        assert self.aligned_address % NT_ARENA_ALIGN == 0

        self.ptr = ctypes.c_void_p(self.aligned_address)
        self.struct = NetTwinArena.from_address(self.aligned_address)
        self._byte_array = (ctypes.c_uint8 * NT_ARENA_BYTES).from_address(self.aligned_address)
        self.np_bytes: np.ndarray = np.ctypeslib.as_array(self._byte_array)

        # Pre-allocated reusable C-ABI scratch buffers (zero per-tick Python allocation)
        self._mtu_scratch = (ctypes.c_uint8 * NT_WASM_MTU_BUF_BYTES)()
        self._acoustics_scratch = (AcousticImpulse * 64)()

        if auto_init:
            rc = self.arena_init()
            if rc != 0:
                raise RuntimeError(f"nt_arena_init failed with status {rc}")

    @property
    def size_bytes(self) -> int:
        return NT_ARENA_BYTES

    @property
    def server_tick(self) -> int:
        return int(self.struct.telemetry_and_header.server_tick)

    def memoryview_bytes(self) -> memoryview:
        return memoryview(self._byte_array)

    def arena_init(self) -> int:
        return int(self._lib.nt_arena_init(self.ptr))

    def set_simd_backend_override(self, mode: int) -> int:
        return int(self._lib.nt_set_simd_backend_override(self.ptr, ctypes.c_uint32(mode)))

    def load_default_raid_map(self) -> int:
        return int(self._lib.nt_load_default_raid_map(self.ptr))

    def spawn_player(
        self,
        player_id: int,
        x: float,
        y: float,
        z: float = 0.0,
        yaw_q15: int = 0,
    ) -> int:
        return int(
            self._lib.nt_spawn_player(
                self.ptr,
                ctypes.c_uint32(player_id),
                ctypes.c_float(x),
                ctypes.c_float(y),
                ctypes.c_float(z),
                ctypes.c_int16(yaw_q15),
            )
        )

    def set_player_rtt_ticks(self, player_id: int, rtt_ticks: int) -> int:
        if hasattr(self._lib, "nt_set_player_rtt_ticks"):
            return int(
                self._lib.nt_set_player_rtt_ticks(
                    self.ptr,
                    ctypes.c_uint32(player_id),
                    ctypes.c_uint32(rtt_ticks),
                )
            )
        self.struct.client_jitter_gates[player_id].est_rtt_ticks = rtt_ticks
        return 0

    def emit_acoustic_event(
        self,
        source_player_id: int,
        x: float,
        y: float,
        z: float,
        band_energy_q8: Sequence[int],
    ) -> int:
        arr = (ctypes.c_uint8 * 4)(*[int(v) & 0xFF for v in band_energy_q8[:4]])
        if hasattr(self._lib, "nt_emit_acoustic_event"):
            return int(
                self._lib.nt_emit_acoustic_event(
                    self.ptr,
                    ctypes.c_uint32(source_player_id),
                    ctypes.c_float(x),
                    ctypes.c_float(y),
                    ctypes.c_float(z),
                    arr,
                )
            )
        return -1

    def get_player_pose(self, player_id: int) -> PlayerPoseSnapshot:
        pose = PlayerPoseSnapshot()
        rc = int(self._lib.nt_get_player_pose(self.ptr, ctypes.c_uint32(player_id), ctypes.byref(pose)))
        if rc != 0:
            raise RuntimeError(f"nt_get_player_pose(player_id={player_id}) failed with code {rc}")
        return pose

    def ingest_client_cmd(self, player_id: int, cmd: ClientInputCmd) -> int:
        return int(self._lib.nt_ingest_client_cmd(self.ptr, ctypes.c_uint32(player_id), ctypes.byref(cmd)))

    def step_kinematics(self) -> int:
        return int(self._lib.nt_step_kinematics(self.ptr))

    def client_reconcile(
        self,
        player_id: int,
        auth_pose: PlayerPoseSnapshot,
        pending_cmds: Sequence[ClientInputCmd] | ctypes.Array[ClientInputCmd],
    ) -> PlayerPoseSnapshot:
        if isinstance(pending_cmds, ctypes.Array):
            cmd_arr = pending_cmds
            num_cmds = len(pending_cmds)
        else:
            num_cmds = len(pending_cmds)
            cmd_arr = (ClientInputCmd * max(1, num_cmds))(*pending_cmds)
        out_pose = PlayerPoseSnapshot()
        rc = int(
            self._lib.nt_client_reconcile(
                self.ptr,
                ctypes.c_uint32(player_id),
                ctypes.byref(auth_pose),
                cmd_arr,
                ctypes.c_uint32(num_cmds),
                ctypes.byref(out_pose),
            )
        )
        if rc != 0:
            raise RuntimeError(f"nt_client_reconcile failed with code {rc}")
        return out_pose

    def resolve_subtick_fire(self, shooter_id: int, fire_cmd: ClientInputCmd) -> HitRecordOut:
        hit = HitRecordOut()
        rc = int(
            self._lib.nt_resolve_subtick_fire(
                self.ptr,
                ctypes.c_uint32(shooter_id),
                ctypes.byref(fire_cmd),
                ctypes.byref(hit),
            )
        )
        if rc < 0:
            raise RuntimeError(f"nt_resolve_subtick_fire failed with code {rc}")
        return hit

    def pack_client_datagram(
        self,
        observer_id: int,
        baseline_ack_tick: int,
        max_buf_len: int = NT_MTU_CEILING_BYTES,
    ) -> bytes:
        cap = min(max_buf_len, NT_WASM_MTU_BUF_BYTES)
        n_bytes = int(
            self._lib.nt_pack_client_datagram(
                self.ptr,
                ctypes.c_uint32(observer_id),
                ctypes.c_uint32(baseline_ack_tick),
                self._mtu_scratch,
                ctypes.c_uint32(cap),
            )
        )
        if n_bytes < 0:
            raise RuntimeError(f"nt_pack_client_datagram failed with code {n_bytes}")
        return bytes(memoryview(self._mtu_scratch)[:n_bytes])

    def unpack_client_datagram(
        self,
        datagram: bytes,
        max_acoustics: int = 64,
    ) -> Tuple[int, List[AcousticImpulse]]:
        buf = (ctypes.c_uint8 * max(1, len(datagram))).from_buffer_copy(
            datagram if len(datagram) > 0 else b"\x00"
        )
        n_acoust = ctypes.c_uint32(0)
        cap = min(max_acoustics, 64)
        rc = int(
            self._lib.nt_unpack_client_datagram(
                self.ptr,
                buf,
                ctypes.c_uint32(len(datagram)),
                self._acoustics_scratch,
                ctypes.c_uint32(cap),
                ctypes.byref(n_acoust),
            )
        )
        if rc < 0:
            raise RuntimeError(f"nt_unpack_client_datagram failed with code {rc}")
        impulses: List[AcousticImpulse] = []
        for i in range(int(n_acoust.value)):
            imp = AcousticImpulse()
            ctypes.memmove(ctypes.byref(imp), ctypes.byref(self._acoustics_scratch[i]), ctypes.sizeof(AcousticImpulse))
            impulses.append(imp)
        return rc, impulses

    def hash_arena_deterministic(self) -> int:
        return int(self._lib.nt_hash_arena_deterministic(self.ptr))

    def canonical_arena_blake2b_hex(self, mask_simd_override: bool = True) -> str:
        """Hashes all 1,572,864 arena bytes via BLAKE2b-256 (zeroing only simd_mode_override flag if requested)."""
        h = hashlib.blake2b(digest_size=32)
        mv = self.memoryview_bytes()
        if not mask_simd_override:
            h.update(mv)
            return h.hexdigest()
        # simd_mode_override is at NT_OFFSET_TELEMETRY_AND_HEADER + 4 .. + 8
        off_flag = NT_OFFSET_TELEMETRY_AND_HEADER + 4
        h.update(mv[:off_flag])
        h.update(b"\x00\x00\x00\x00")
        h.update(mv[off_flag + 4 :])
        return h.hexdigest()

    def canonical_arena_blake3_hex(self, mask_simd_override: bool = True) -> str:
        """Hashes all 1,572,864 arena bytes via official BLAKE3-256."""
        if not mask_simd_override:
            return blake3_256_hex(self.np_bytes)
        saved = int(self.struct.telemetry_and_header.simd_mode_override)
        try:
            self.struct.telemetry_and_header.simd_mode_override = 0
            return blake3_256_hex(self.np_bytes)
        finally:
            self.struct.telemetry_and_header.simd_mode_override = saved
