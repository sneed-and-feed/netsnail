"""Comprehensive Unit, Numerical, Causal & Adversarial Red-Team Pytest Suite for NetSnail v0.2.0."""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import netsnail
from netsnail.ffi import (
    NET_TWIN_ARENA_ALIGN_BYTES,
    NET_TWIN_ARENA_SIZE_BYTES,
    NET_TWIN_PARTITION_OFFSETS,
    NET_TWIN_PARTITION_SIZES,
    NT_ARENA_ALIGN,
    NT_ARENA_BYTES,
    NT_BTN_ADS,
    NT_BTN_CROUCH,
    NT_BTN_FIRE,
    NT_BTN_LEAN_L,
    NT_BTN_LEAN_R,
    NT_BTN_SPRINT,
    NT_HIT_FLAG_HIT,
    NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
    NT_HIT_FLAG_PENETRATED,
    NT_HIT_FLAG_RICOCHET,
    NT_MTU_CEILING_BYTES,
    NT_POSE_HAS_STAMINA,
    NT_REWIND_RING_BYTES,
    NT_SIMD_MODE_AUTO,
    NT_SIMD_MODE_FORCE_SCALAR,
    NT_STANCE_MASK_7BIT,
    AcousticImpulse,
    ClientInputCmd,
    ClientJitterGate,
    HitRecordOut,
    NetTwinArena,
    NetTwinArenaBridge,
    PlayerPoseSnapshot,
    PlayerTickState,
    TelemetryAndHeader,
    TierACylinder,
    blake3_256_hex,
    load_native_library,
)
from netsnail.fuzzer import (
    default_chaos_spawns,
    generate_chaos_command_bytes,
    run_cross_isa_chaos_fuzzer,
    verify_mxcsr_ftz_resilience,
)
from netsnail.network_sim import (
    run_scenario_a_honest_wan_and_ge_wifi,
    run_scenario_b_clumsy_lag_switch_red_team,
    run_scenario_c_behind_pillar_occluder_cutoff,
    run_scenario_d_anti_dma_and_doorway_pop_in,
)
from scripts.verify_bcm2712_arena import verify_bcm2712_arena
from scripts.verify_cross_isa_lockstep import verify_cross_isa_lockstep
from scripts.verify_zero_alloc import audit_zero_alloc


def _yaw_q15(rad: float) -> int:
    wrapped = ((rad + math.pi) % (2.0 * math.pi)) - math.pi
    return int(max(-32768, min(32767, round((wrapped / math.pi) * 32767.0))))


# ==============================================================================
# 1. Struct ABI Layout, Cross-Language Offset Parity & BLAKE3 Primitives
# ==============================================================================

def test_01_c_abi_wire_struct_exact_sizes_and_zero_padding() -> None:
    assert ctypes.sizeof(ClientInputCmd) == 16
    assert ctypes.sizeof(HitRecordOut) == 32
    assert ctypes.sizeof(AcousticImpulse) == 12
    assert ctypes.sizeof(PlayerPoseSnapshot) == 32
    assert ctypes.sizeof(TierACylinder) == 16
    assert ctypes.sizeof(PlayerTickState) == 64
    assert ctypes.sizeof(ClientJitterGate) == 1024
    assert ctypes.sizeof(TelemetryAndHeader) == 65_536

    # Exact wire byte offsets
    assert ClientInputCmd.cmd_tick.offset == 0
    assert ClientInputCmd.buttons_stance.offset == 4
    assert ClientInputCmd.move_xy_q15.offset == 6
    assert ClientInputCmd.yaw_pitch_q15.offset == 10
    assert ClientInputCmd.fire_subtick_u8.offset == 14
    assert ClientInputCmd.weapon_slot_u8.offset == 15

    assert HitRecordOut.shooter_id.offset == 0
    assert HitRecordOut.victim_id.offset == 4
    assert HitRecordOut.rewind_tick.offset == 8
    assert HitRecordOut.plate_zone_id.offset == 12
    assert HitRecordOut.outcome_flags.offset == 14
    assert HitRecordOut.subtick_alpha.offset == 16
    assert HitRecordOut.damage_applied.offset == 20
    assert HitRecordOut.armor_damage.offset == 24
    assert HitRecordOut.hit_dist_m.offset == 28

    assert AcousticImpulse.arrival_tick.offset == 0
    assert AcousticImpulse.azimuth_q12.offset == 4
    assert AcousticImpulse.elevation_q12.offset == 6
    assert AcousticImpulse.band_energy_q8.offset == 8


def test_02_net_twin_arena_size_alignment_and_8_partition_offsets() -> None:
    assert ctypes.sizeof(NetTwinArena) == NET_TWIN_ARENA_SIZE_BYTES == 1_572_864
    assert NET_TWIN_ARENA_ALIGN_BYTES == 64
    assert NET_TWIN_PARTITION_OFFSETS == (
        0,
        65_536,
        327_680,
        589_824,
        655_360,
        720_896,
        1_245_184,
        1_507_328,
    )
    assert sum(NET_TWIN_PARTITION_SIZES) == 1_572_864
    assert NET_TWIN_PARTITION_SIZES[0] + NET_TWIN_PARTITION_SIZES[1] == NT_REWIND_RING_BYTES == 327_680
    assert NT_REWIND_RING_BYTES <= 524_288  # Fits inside 512 KB Cortex-A76 private L2
    assert NT_ARENA_BYTES < 2_097_152       # Fits inside 2.0 MB BCM2712 shared L3

    arena = NetTwinArenaBridge(auto_init=True)
    assert arena.aligned_address % NT_ARENA_ALIGN == 0
    assert arena.np_bytes.nbytes == 1_572_864


def test_03_cross_language_header_offset_parity() -> None:
    hdr_path = _REPO_ROOT / "include" / "net_twin.h"
    assert hdr_path.is_file(), "include/net_twin.h missing"
    text = hdr_path.read_text(encoding="utf-8")

    expected_macros = {
        "NT_ARENA_BYTES": 1_572_864,
        "NT_ARENA_ALIGN": 64,
        "NT_OFFSET_TIER_A_CYLINDERS": 0,
        "NT_OFFSET_TIER_B_POSES_PLATES": 65_536,
        "NT_OFFSET_STATIC_VOXEL_BVH": 327_680,
        "NT_OFFSET_CLIENT_JITTER_GATES": 589_824,
        "NT_OFFSET_PROJECTILES_ACOUSTIC_SLAB": 655_360,
        "NT_OFFSET_CLIENT_BASELINE_SNAPSHOTS": 720_896,
        "NT_OFFSET_TRIAD_CASCADE_SCRATCH": 1_245_184,
        "NT_OFFSET_TELEMETRY_AND_HEADER": 1_507_328,
    }
    for macro, val in expected_macros.items():
        m = re.search(rf"#define\s+{macro}\s+(\d+)u?", text)
        assert m is not None, f"Missing {macro} in include/net_twin.h"
        assert int(m.group(1)) == val, f"{macro} mismatch: {m.group(1)} != {val}"


def test_04_blake3_official_test_vectors() -> None:
    # Official RFC/spec BLAKE3-256 test vectors (empty & "abc" & official patterned inputs)
    assert blake3_256_hex(b"") == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    assert blake3_256_hex(b"abc") == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"

    # Official BLAKE3 test_vectors.json: input[i] = i % 251 for len = 1024, 2048, 3072
    pat_1024 = bytes(i % 251 for i in range(1024))
    pat_2048 = bytes(i % 251 for i in range(2048))
    pat_3072 = bytes(i % 251 for i in range(3072))
    assert blake3_256_hex(pat_1024) == "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"
    assert blake3_256_hex(pat_2048) == "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"
    assert blake3_256_hex(pat_3072) == "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"


# ==============================================================================
# 2. Tier-0 Numerical Kernels, Cayley Rotor & Swept-Capsule Kinematics
# ==============================================================================

def test_05_tier0_q15_trig_and_canonical_float_invariants() -> None:
    lib = load_native_library()
    max_pythag_err = 0.0
    for phase in range(-32768, 32768, 64):
        s = float(lib.nt_sin_q15(phase))
        c = float(lib.nt_cos_q15(phase))
        err = abs((s * s + c * c) - 1.0)
        max_pythag_err = max(max_pythag_err, err)
        # Canonical +0.0 check: negative zero (-0.0) must never be emitted
        s_bits = struct.unpack("<I", struct.pack("<f", s))[0]
        c_bits = struct.unpack("<I", struct.pack("<f", c))[0]
        assert s_bits != 0x8000_0000, f"Negative zero emitted by nt_sin_q15({phase})"
        assert c_bits != 0x8000_0000, f"Negative zero emitted by nt_cos_q15({phase})"

    assert max_pythag_err < 2.0e-6
    assert abs(float(lib.nt_sqrt_f32(9.0)) - 3.0) == 0.0
    assert abs(float(lib.nt_atan2_f32(1.0, 1.0)) - (math.pi / 4.0)) < 2.0e-6


def test_06_tier0_dot16_pairwise_simd_vs_scalar_zero_ulp() -> None:
    lib = load_native_library()
    # Adversarial cancellation array sensitive to FMA contraction vs pairwise tree order
    vals_a = np.array(
        [1e7, -1e7, 3.1415927, -2.7182817, 1e-4, -1e-4, 0.1234567, 0.7654321,
         -500.25, 500.25, 1.0000001, -0.9999999, 42.125, -42.125, 1e-6, 2e-6],
        dtype=np.float32,
    )
    vals_b = np.array(
        [1.0, 1.0, 0.5, 0.5, 3.0, 3.0, -2.25, 1.75,
         4.0, 4.0, 8.125, 8.125, 0.0625, 0.0625, 100.0, -50.0],
        dtype=np.float32,
    )

    ptr_a = vals_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_b = vals_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    got = np.float32(lib.nt_dot16_pairwise_f32(ptr_a, ptr_b))

    # Exact balanced pairwise f32 tree in NumPy float32
    p = (vals_a * vals_b).astype(np.float32)
    l1 = (p[0::2] + p[1::2]).astype(np.float32)
    l2 = (l1[0::2] + l1[1::2]).astype(np.float32)
    l3 = (l2[0::2] + l2[1::2]).astype(np.float32)
    expected = np.float32(l3[0] + l3[1])

    got_u32 = int(got.view(np.uint32))
    exp_u32 = int(expected.view(np.uint32))
    assert got_u32 == exp_u32, f"Pairwise dot16 ULP mismatch: got=0x{got_u32:08x} vs expected=0x{exp_u32:08x}"


def test_07_cayley_rotor_so3_orthogonality_and_norm_conservation() -> None:
    lib = load_native_library()
    rng = np.random.default_rng(2026)
    for _ in range(200):
        omega = rng.normal(0.0, 0.8, size=3).astype(np.float32)
        v_in = rng.normal(0.0, 5.0, size=3).astype(np.float32)
        r_out = np.zeros(9, dtype=np.float32)
        v_out = np.zeros(3, dtype=np.float32)

        rc = lib.nt_cayley_rotor_so3_f32(
            omega.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            r_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        assert rc == 0
        R = r_out.reshape(3, 3)
        ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3, dtype=np.float32))))
        assert ortho_err < 2.0e-6, f"SO(3) Cayley orthogonality drift {ortho_err}"

        norm_in = float(np.linalg.norm(v_in))
        norm_out = float(np.linalg.norm(v_out))
        assert abs(norm_out - norm_in) <= 2.5e-6 * max(1.0, norm_in)


def test_08_swept_capsule_collide_and_slide_wall_and_crease_non_penetration() -> None:
    arena = NetTwinArenaBridge(auto_init=True)
    assert arena.load_default_raid_map() == 0

    # Player 0 sprints straight South (-Y) into central concrete pillar [-1.5, 1.5] x [-1.5, 1.5]
    assert arena.spawn_player(0, 0.0, 3.2, 0.0, _yaw_q15(-math.pi / 2.0)) == 0
    # Player 1 sprints diagonally into acute NE wall corner (x=10, y=12)
    assert arena.spawn_player(1, 8.2, 10.2, 0.0, _yaw_q15(math.pi / 4.0)) == 0

    for t in range(1, 160):
        c0 = ClientInputCmd(cmd_tick=t, buttons_stance=NT_BTN_SPRINT)
        c0.move_xy_q15[0] = 32767
        c0.yaw_pitch_q15[0] = _yaw_q15(-math.pi / 2.0)
        arena.ingest_client_cmd(0, c0)

        c1 = ClientInputCmd(cmd_tick=t, buttons_stance=NT_BTN_SPRINT | NT_BTN_LEAN_L)
        c1.move_xy_q15[0] = 32767
        c1.yaw_pitch_q15[0] = _yaw_q15(math.pi / 4.0)
        arena.ingest_client_cmd(1, c1)

        arena.step_kinematics()
        p0 = arena.get_player_pose(0)
        p1 = arena.get_player_pose(1)

        # Player 0 must NEVER penetrate inside central concrete pillar interior (-1.5, 1.5)^2
        assert not (-1.49 < p0.x < 1.49 and -1.49 < p0.y < 1.49), (
            f"Capsule tunneled into central pillar at tick {t}: ({p0.x}, {p0.y})"
        )
        # Player 1 must NEVER tunnel through solid East/North walls at corner (10.0, 12.0)
        assert p1.x < 10.0 and p1.y < 12.0, f"Corner crease penetration at tick {t}: ({p1.x}, {p1.y})"


# ==============================================================================
# 3. Prediction Reconciliation, Sub-Tick Ballistics & Multi-Plate Ricochet
# ==============================================================================

def test_09_client_reconcile_32_tick_lockstep_and_sub_90us_wcet() -> None:
    srv = NetTwinArenaBridge(auto_init=True)
    cli = NetTwinArenaBridge(auto_init=True)
    srv.load_default_raid_map()
    cli.load_default_raid_map()

    srv.spawn_player(0, -4.0, 3.0, 0.0, 0)
    cli.spawn_player(0, -4.0, 3.0, 0.0, 0)
    auth_pose = srv.get_player_pose(0)

    cmds = []
    for i in range(32):
        cmd = ClientInputCmd(cmd_tick=i + 1, buttons_stance=NT_BTN_SPRINT if i < 20 else NT_BTN_CROUCH)
        cmd.move_xy_q15[0] = 26000
        cmd.move_xy_q15[1] = -12000 if (i & 2) else 12000
        cmd.yaw_pitch_q15[0] = _yaw_q15(0.04 * i)
        cmds.append(cmd)
        srv.ingest_client_cmd(0, cmd)
        srv.step_kinematics()

    srv_final = srv.get_player_pose(0)
    cli_pred = cli.client_reconcile(0, auth_pose, cmds)

    # Exact bit-level prediction lockstep between 32 server steps and 1 client_reconcile replay!
    assert cli_pred.x == srv_final.x and cli_pred.y == srv_final.y and cli_pred.z == srv_final.z
    assert cli_pred.vx == srv_final.vx and cli_pred.vy == srv_final.vy

    scen_a = run_scenario_a_honest_wan_and_ge_wifi(num_ticks=256)
    assert scen_a["reconcile_32_tick_us"]["within_90us_budget"], (
        f"32-tick reconcile exceeded 90 us budget: {scen_a['reconcile_32_tick_us']}"
    )


def test_10_subtick_alpha_ballistics_interpolation_accuracy() -> None:
    arena = NetTwinArenaBridge(auto_init=True)
    arena.load_default_raid_map()

    # Shooter at (-6.0, -5.0) facing +Y; Target sprints across X at y = +4.0 (clear of central pillar)
    arena.spawn_player(0, -6.0, -5.0, 0.0, _yaw_q15(math.pi / 2.0))
    arena.spawn_player(1, -6.5, 4.0, 0.0, 0)
    arena.set_player_rtt_ticks(0, 6)
    arena.set_player_rtt_ticks(1, 4)

    for t in range(1, 12):
        c0 = ClientInputCmd(cmd_tick=t)
        c0.yaw_pitch_q15[0] = _yaw_q15(math.pi / 2.0)
        arena.ingest_client_cmd(0, c0)

        c1 = ClientInputCmd(cmd_tick=t, buttons_stance=NT_BTN_SPRINT)
        c1.move_xy_q15[0] = 32767
        c1.yaw_pitch_q15[0] = 0
        arena.ingest_client_cmd(1, c1)
        arena.step_kinematics()

    # Aim directly at analytical mid-tick (alpha = 128/256 = 0.5) pose between tick 9 and tick 10
    s9 = arena.struct.tier_b_poses_plates[9 & 63][1]
    s10 = arena.struct.tier_b_poses_plates[10 & 63][1]
    mid_x = 0.5 * (float(s9.pos[0]) + float(s10.pos[0]))
    mid_y = 0.5 * (float(s9.pos[1]) + float(s10.pos[1]))
    aim_yaw = math.atan2(mid_y - (-5.0), mid_x - (-6.0))

    fire = ClientInputCmd(cmd_tick=9, buttons_stance=NT_BTN_FIRE, fire_subtick_u8=128, weapon_slot_u8=0)
    fire.yaw_pitch_q15[0] = _yaw_q15(aim_yaw)
    hit = arena.resolve_subtick_fire(0, fire)

    assert abs(hit.subtick_alpha - 0.5) < 1e-6
    assert bool(hit.outcome_flags & NT_HIT_FLAG_HIT), f"Expected mid-subtick hit, got flags=0x{hit.outcome_flags:02x}"
    assert hit.victim_id == 1
    assert 0 <= hit.plate_zone_id < 30


def test_11_multi_plate_30_zone_obliquity_ricochet_and_penetration() -> None:
    arena = NetTwinArenaBridge(auto_init=True)
    arena.load_default_raid_map()

    # Shooter at (-6.0, -6.0), Victim stationary at (-6.0, 2.0) (8m range, clear sightline)
    arena.spawn_player(0, -6.0, -6.0, 0.0, _yaw_q15(math.pi / 2.0))
    arena.spawn_player(1, -6.0, 2.0, 0.0, _yaw_q15(-math.pi / 2.0))
    for t in range(1, 5):
        arena.ingest_client_cmd(0, ClientInputCmd(cmd_tick=t, yaw_pitch_q15=(_yaw_q15(math.pi / 2.0), 0)))
        arena.ingest_client_cmd(1, ClientInputCmd(cmd_tick=t, yaw_pitch_q15=(_yaw_q15(-math.pi / 2.0), 0)))
        arena.step_kinematics()

    # Dead-center orthogonal rifle shot -> direct penetration/hit
    center_shot = ClientInputCmd(cmd_tick=4, buttons_stance=NT_BTN_FIRE, fire_subtick_u8=0, weapon_slot_u8=1)
    center_shot.yaw_pitch_q15[0] = _yaw_q15(math.pi / 2.0)
    hit_center = arena.resolve_subtick_fire(0, center_shot)
    assert bool(hit_center.outcome_flags & NT_HIT_FLAG_HIT)

    # Sweep grazing lateral offsets near capsule tangent edge to verify obliquity ricochet + penetration mechanics
    saw_ricochet = False
    saw_penetrated = bool(hit_center.outcome_flags & NT_HIT_FLAG_PENETRATED)
    for lateral_offset in np.linspace(0.0, 0.39, 40):
        grazing_yaw = math.atan2(2.0 - (-6.0), (-6.0 + float(lateral_offset)) - (-6.0))
        for wslot in (0, 1, 2):
            for pitch in (0, -400, -750):
                shot = ClientInputCmd(cmd_tick=4, buttons_stance=NT_BTN_FIRE, fire_subtick_u8=64, weapon_slot_u8=wslot)
                shot.yaw_pitch_q15[0] = _yaw_q15(grazing_yaw)
                shot.yaw_pitch_q15[1] = pitch
                h = arena.resolve_subtick_fire(0, shot)
                if h.outcome_flags & NT_HIT_FLAG_RICOCHET:
                    saw_ricochet = True
                if h.outcome_flags & NT_HIT_FLAG_PENETRATED:
                    saw_penetrated = True
    assert saw_penetrated, "Expected armor penetration on orthogonal/high-pen shots"
    assert saw_ricochet, "Expected shallow-obliquity armor plate ricochet on grazing edge shots"

    # Also test >25m long-range shot spawning active finite-velocity ballistic projectile
    arena_lr = NetTwinArenaBridge(auto_init=True)
    arena_lr.load_default_raid_map()
    arena_lr.spawn_player(0, -6.0, -25.0, 0.0, _yaw_q15(math.pi / 2.0))
    arena_lr.spawn_player(1, -6.0, 15.0, 0.0, _yaw_q15(-math.pi / 2.0))
    arena_lr.step_kinematics()
    lr_shot = ClientInputCmd(cmd_tick=1, buttons_stance=NT_BTN_FIRE, fire_subtick_u8=0, weapon_slot_u8=2)
    lr_shot.yaw_pitch_q15[0] = _yaw_q15(math.pi / 2.0)
    _ = arena_lr.resolve_subtick_fire(0, lr_shot)
    # Either resolved or spawned ballistic projectile tracked in header
    assert (
        arena_lr.struct.telemetry_and_header.active_projectile_count >= 0
        and arena_lr.struct.telemetry_and_header.total_shots_resolved >= 1
    )


# ==============================================================================
# 4. Adversarial Combat & Security Invariants (Occluder, Lag-Switch, Anti-DMA)
# ==============================================================================

def test_12_mutual_occluder_behind_pillar_cutoff_zero_behind_wall_deaths() -> None:
    res = run_scenario_c_behind_pillar_occluder_cutoff(num_trials=20)
    assert res["honest_exposed_control_hit_rate_pct"] == 100.0
    assert res["netsnail_behind_wall_deaths"] == 0
    assert res["netsnail_behind_wall_death_rate_pct"] == 0.0
    assert res["netsnail_occluder_cutoff_rejections"] == 20
    assert res["naive_behind_wall_death_rate_pct"] > 80.0


def test_13_morse_jitter_gate_honest_ping_invariance_vs_clumsy_lag_switch() -> None:
    res = run_scenario_b_clumsy_lag_switch_red_team(num_trials=20)
    assert res["honest_warmup_false_quarantines"] == 0
    assert res["netsnail_quarantine_detection_rate_pct"] >= 95.0
    assert res["netsnail_lag_switch_exploit_rejection_rate_pct"] >= 95.0
    assert res["naive_lag_switch_exploit_rejection_rate_pct"] < 20.0


def test_14_anti_dma_observational_noninterference_zero_leak_bits() -> None:
    res = run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks=64)
    assert res["netsnail_out_of_cone_dma_wire_leak_bits"] == 0
    assert res["netsnail_out_of_cone_dma_ram_leak_bits"] == 0
    assert res["netsnail_out_of_cone_leak_pct"] == 0.0
    assert res["naive_out_of_cone_dma_wire_leak_bits"] > 10_000


def test_15_anonymous_acoustic_portal_diffraction_100pct_id_stripping() -> None:
    res = run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks=16)
    assert res["acoustic_impulse_size_bytes"] == 12
    assert res["acoustic_entity_id_stripping_pct"] == 100.0


def test_16_minkowski_reachability_pvs_zero_doorway_pop_in() -> None:
    res = run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks=16)
    assert res["doorway_swing"]["netsnail_pop_in_frames"] == 0
    assert res["doorway_swing"]["pre_delivery_lead_ticks"] >= 0


def test_17_mtu_1200_byte_ceiling_under_16_player_combat() -> None:
    arena = NetTwinArenaBridge(auto_init=True)
    arena.load_default_raid_map()
    for p in range(16):
        ang = 2.0 * math.pi * p / 16.0
        arena.spawn_player(p, 5.5 * math.cos(ang), 5.5 * math.sin(ang), 0.0, _yaw_q15(ang + math.pi))

    max_len = 0
    for t in range(1, 65):
        for p in range(16):
            c = ClientInputCmd(cmd_tick=t, buttons_stance=NT_BTN_SPRINT)
            c.move_xy_q15[0] = 25000
            c.yaw_pitch_q15[0] = _yaw_q15(0.1 * t + p)
            arena.ingest_client_cmd(p, c)
            if (p + t) % 4 == 0:
                arena.emit_acoustic_event(p, 4.0, 4.0, 0.0, [200, 160, 120, 80])
        arena.step_kinematics()
        for obs in range(4):
            dgram = arena.pack_client_datagram(obs, max(0, arena.server_tick - 6))
            assert 0 < len(dgram) < NT_MTU_CEILING_BYTES, f"MTU exceeded: {len(dgram)} >= {NT_MTU_CEILING_BYTES}"
            max_len = max(max_len, len(dgram))
    assert 0 < max_len < NT_MTU_CEILING_BYTES


# ==============================================================================
# 5. Cross-ISA 0-ULP Determinism, MXCSR Hygiene, BCM2712 Cache & Zero-Alloc
# ==============================================================================

def test_18_mxcsr_flush_to_zero_ffi_resilience() -> None:
    report = verify_mxcsr_ftz_resilience(num_ticks=256, num_players=8)
    assert report["tick_digest_mismatches"] == 0
    assert report["caller_control_word_preserved"] is True
    assert report["zero_ulp_resilient"] is True


def test_19_cross_isa_x86_simd_vs_scalar_vs_wasm32_zero_ulp_chaos() -> None:
    # Execute >= 100,000 live player-tick steps (16 players * 6,400 ticks = 102,400 player-ticks)
    live_res = run_cross_isa_chaos_fuzzer(
        num_ticks=6_400,
        num_players=16,
        full_arena_hash_every_tick=False,
        output_path=None,
    )
    ps = live_res["parity_summary"]
    assert live_res["total_player_ticks"] >= 100_000
    assert ps["mismatched_ticks_simd_vs_scalar"] == 0
    assert ps["mismatched_ticks_simd_vs_wasm32"] == 0
    assert ps["digest_x86_simd_blake3_256"] == ps["digest_scalar_blake3_256"] == ps["digest_wasm32_blake3_256"]
    assert ps["zero_ulp_cross_isa_verified"] is True

    # Also verify archived 160,000 player-tick receipt if present
    receipt_file = _REPO_ROOT / "results" / "cross_isa_blake3_receipts.json"
    if receipt_file.is_file():
        archived = json.loads(receipt_file.read_text(encoding="utf-8"))
        assert archived["total_player_ticks"] >= 100_000
        assert archived["parity_summary"]["zero_ulp_cross_isa_verified"] is True


def test_20_bcm2712_cache_residency_and_zero_alloc_binary_audit(tmp_path: Path) -> None:
    za = audit_zero_alloc(require_both=True)
    assert za["native"]["passed"] is True, f"Native zero-alloc failed: {za['native']}"
    assert za["wasm32"]["passed"] is True, f"WASM32 zero-alloc failed: {za['wasm32']}"
    assert za["overall_passed"] is True

    cache_out = tmp_path / "bcm2712_test.json"
    bcm = verify_bcm2712_arena(num_references=50_000_000, output_path=str(cache_out))
    assert bcm["summary"]["l3_capacity_evictions"] == 0
    assert bcm["summary"]["lld_miss_rate_pct"] < 0.100
    assert bcm["summary"]["overall_verification_status"] == "PASS"


# ==============================================================================
# v0.2.0 Adversarial Regression & Property Tests (Bugs 1-5 + WASM32 +simd128)
# ==============================================================================

def test_v02_delta_lean_vz_stance_multihop_fidelity() -> None:
    assert netsnail.__version__ == "0.2.0"
    assert NT_STANCE_MASK_7BIT == 0x007F
    assert NT_POSE_HAS_STAMINA == (1 << 7)

    srv = NetTwinArenaBridge(auto_init=True)
    srv_shadow = NetTwinArenaBridge(auto_init=True)
    cli = NetTwinArenaBridge(auto_init=True)
    srv.load_default_raid_map()
    srv_shadow.load_default_raid_map()
    cli.load_default_raid_map()

    # Observer 0 and visible Enemy 1 inside PVS; Enemy 9 strictly occluded behind West Wing concrete wall
    for a in (srv, srv_shadow):
        a.spawn_player(0, -5.0, 0.0, 0.0, 16384)
        a.spawn_player(1, -5.0, 4.0, 0.0, -16384)
        a.spawn_player(9, -15.0, 10.0, 0.0, 0)
        a.set_player_rtt_ticks(0, 8)

    # Deliver initial baseline at tick 0
    pkt0 = srv.pack_client_datagram(0, 0)
    pkt0_sh = srv_shadow.pack_client_datagram(0, 0)
    assert pkt0 == pkt0_sh
    vis0, _ = cli.unpack_client_datagram(pkt0)
    assert vis0 == 2
    last_acked_tick = 0

    saw_short_0x80 = False
    saw_wide_0xc0 = False
    saw_multihop_gap = False

    # Step 64 ticks with simulated bursty packet drops (non-trivial ack_mask_u64 baseline hops!)
    for step in range(64):
        btns = [
            NT_BTN_LEAN_L | NT_BTN_CROUCH,
            NT_BTN_LEAN_R | NT_BTN_ADS,
            NT_BTN_SPRINT,
            NT_BTN_LEAN_L,
            NT_BTN_LEAN_R | NT_BTN_CROUCH,
        ][step % 5]

        cmd1 = ClientInputCmd()
        cmd1.cmd_tick = step + 1
        cmd1.buttons_stance = btns
        cmd1.move_xy_q15[0] = 18000 if (step % 2 == 0) else -14000
        cmd1.move_xy_q15[1] = -12000 if (step % 3 == 0) else 12000
        cmd1.yaw_pitch_q15[0] = int(((-16384 + step * 733) & 0xFFFF) - 32768)
        cmd1.yaw_pitch_q15[1] = -1200 + (step % 16) * 150

        # Mutate occluded Enemy 9 ONLY in srv_shadow (wild lean, sprint/crouch, yaw, position)
        cmd9 = ClientInputCmd()
        cmd9.cmd_tick = step + 1
        cmd9.buttons_stance = NT_BTN_LEAN_R | NT_BTN_SPRINT
        cmd9.move_xy_q15[0] = 32767
        cmd9.move_xy_q15[1] = -20000
        cmd9.yaw_pitch_q15[0] = int(((step * 3907) & 0xFFFF) - 32768)

        srv.ingest_client_cmd(1, cmd1)
        srv_shadow.ingest_client_cmd(1, cmd1)
        srv_shadow.ingest_client_cmd(9, cmd9)

        srv.step_kinematics()
        srv_shadow.step_kinematics()

        cur_tick = srv.server_tick
        cur_slot = cur_tick & 63
        vz_target = 2.25 if (step & 1) == 0 else -1.75
        for a in (srv, srv_shadow):
            a.struct.tier_b_poses_plates[cur_slot][1].vel[2] = vz_target
            if step == 31:
                # > 0.256 m displacement forces 14-byte wide modular delta record (tag 0xC0 | pid)
                a.struct.tier_b_poses_plates[cur_slot][1].pos[0] += 0.48
        srv_shadow.struct.tier_b_poses_plates[cur_slot][9].vel[2] = -3.5

        pkt = srv.pack_client_datagram(0, last_acked_tick)
        pkt_sh = srv_shadow.pack_client_datagram(0, last_acked_tick)
        assert pkt == pkt_sh, f"Out-of-cone occluded mutation leaked wire bits at step {step}!"
        assert 16 < len(pkt) < NT_MTU_CEILING_BYTES

        # Drop steps {2,3,4}, {11,12}, {25,26,27,28}, {44,45} to test multi-tick baseline hops
        dropped = step in {2, 3, 4, 11, 12, 25, 26, 27, 28, 44, 45}
        if dropped:
            continue

        if cur_tick - last_acked_tick > 1:
            saw_multihop_gap = True

        obs_tag = pkt[16] & 0xC0
        obs_len = 12 if obs_tag == 0x80 else 14
        enemy_tag = pkt[16 + obs_len] & 0xC0
        if enemy_tag == 0x80:
            saw_short_0x80 = True
        elif enemy_tag == 0xC0:
            saw_wide_0xc0 = True

        vis, _ = cli.unpack_client_datagram(pkt)
        assert vis == 2
        last_acked_tick = cur_tick

        s_srv = srv.struct.tier_b_poses_plates[cur_slot][1]
        s_cli = cli.struct.tier_b_poses_plates[cur_slot][1]

        exp_lean_q15 = int(np.int8(s_srv.lean_q15 >> 8)) << 8
        exp_vz = round(float(s_srv.vel[2]) * 16.0) / 16.0
        exp_stance_7b = int(s_srv.stance_flags) & NT_STANCE_MASK_7BIT

        assert int(s_cli.lean_q15) == exp_lean_q15, f"Lean delta mismatch at step {step}"
        assert abs(float(s_cli.vel[2]) - exp_vz) <= (1.0 / 16.0 + 1e-5), f"vz delta mismatch at step {step}"
        assert (int(s_cli.stance_flags) & NT_STANCE_MASK_7BIT) == exp_stance_7b, f"Stance mismatch at step {step}"

    assert saw_multihop_gap, "Expected multi-tick baseline hop across simulated packet drops"
    assert saw_short_0x80, "Expected 12 B short modular delta (0x80) records"
    assert saw_wide_0xc0, "Expected 14 B wide modular delta (0xC0) records"

    # Worst-case 64-player full-state + 8 acoustic impulses <= 1,200 B MTU
    srv64 = NetTwinArenaBridge(auto_init=True)
    srv64.load_default_raid_map()
    for p in range(64):
        rx = -26.5 + float(p % 8) * 1.5
        ry = -7.5 + float(p // 8) * 2.0
        srv64.spawn_player(p, rx, ry, 0.0, 0)
    for k in range(8):
        srv64.emit_acoustic_event(k + 1, -20.0, 0.0, 0.0, (210, 180, 140, 100))
    pkt64 = srv64.pack_client_datagram(0, 0)
    assert len(pkt64) <= 1008 <= NT_MTU_CEILING_BYTES
    cli64 = NetTwinArenaBridge(auto_init=True)
    vis64, ac64 = cli64.unpack_client_datagram(pkt64)
    assert vis64 == 64 and len(ac64) == 8


def test_v02_long_range_and_ricochet_ballistic_projectiles() -> None:
    # (a) CQB (<= 25 m) hit resolves instantaneously without double-sweeping downrange bystander
    arena_cqb = NetTwinArenaBridge(auto_init=True)
    arena_cqb.load_default_raid_map()
    arena_cqb.spawn_player(0, -18.0, -20.0, 0.0, 16384)
    arena_cqb.spawn_player(1, -18.0, -10.0, 0.0, -16384)  # 10 m CQB victim
    arena_cqb.spawn_player(2, -18.0, 18.0, 0.0, -16384)   # 38 m downrange bystander

    cqb_cmd = ClientInputCmd()
    cqb_cmd.cmd_tick = 0
    cqb_cmd.buttons_stance = NT_BTN_FIRE
    cqb_cmd.yaw_pitch_q15[0] = 16384
    cqb_cmd.yaw_pitch_q15[1] = -300
    cqb_cmd.weapon_slot_u8 = 1  # 7.62x51 AP

    hit_cqb = arena_cqb.resolve_subtick_fire(0, cqb_cmd)
    assert hit_cqb.victim_id == 1
    assert (hit_cqb.outcome_flags & NT_HIT_FLAG_HIT) != 0
    assert arena_cqb.struct.telemetry_and_header.active_projectile_count == 0
    for _ in range(10):
        arena_cqb.step_kinematics()
    slot_cqb = arena_cqb.server_tick & 63
    assert arena_cqb.struct.tier_b_poses_plates[slot_cqb][2].health_q8 == 25600

    # (b) Long-range (> 25 m, 45.0 m) unobstructed shot spawns projectile at 25 m handoff and degrades target plates
    arena_lr = NetTwinArenaBridge(auto_init=True)
    arena_lr.load_default_raid_map()
    arena_lr.spawn_player(0, -18.0, -24.0, 0.0, 16384)
    arena_lr.spawn_player(1, -18.0, 21.0, 0.0, -16384)    # 45.0 m (> 25.0 m)

    orig_hp = int(arena_lr.struct.tier_b_poses_plates[0][1].health_q8)
    orig_plates = sum(int(x) for x in arena_lr.struct.tier_b_poses_plates[0][1].plate_hp_pools_q8)

    lr_cmd = ClientInputCmd()
    lr_cmd.cmd_tick = 0
    lr_cmd.buttons_stance = NT_BTN_FIRE
    lr_cmd.yaw_pitch_q15[0] = 16384
    lr_cmd.yaw_pitch_q15[1] = -100
    lr_cmd.weapon_slot_u8 = 1  # 840 m/s 7.62x51 AP

    hit_lr = arena_lr.resolve_subtick_fire(0, lr_cmd)
    assert hit_lr.victim_id == 0xFFFFFFFF  # Deferred past 25 m CQB horizon
    assert arena_lr.struct.telemetry_and_header.active_projectile_count == 1

    for _ in range(8):
        arena_lr.step_kinematics()
    slot_lr = arena_lr.server_tick & 63
    post_st = arena_lr.struct.tier_b_poses_plates[slot_lr][1]
    post_plates = sum(int(x) for x in post_st.plate_hp_pools_q8)
    assert int(post_st.health_q8) < orig_hp and post_plates < orig_plates
    assert arena_lr.struct.telemetry_and_header.active_projectile_count == 0

    # (c) Shallow-angle hard-cover ricochet spawns live projectile capable of striking angled capsule
    arena_ric = NetTwinArenaBridge(auto_init=True)
    arena_ric.load_default_raid_map()
    lib = load_native_library()
    ric_rad = float(lib.nt_atan2_f32(ctypes.c_float(-0.8), ctypes.c_float(3.2)))
    ric_yaw = int(round((ric_rad / math.pi) * 32768.0))
    arena_ric.spawn_player(0, -8.5, -10.7, 0.0, ric_yaw)
    arena_ric.spawn_player(1, -2.1, -10.7, 0.0, -32768)

    ric_cmd = ClientInputCmd()
    ric_cmd.cmd_tick = 0
    ric_cmd.buttons_stance = NT_BTN_FIRE
    ric_cmd.yaw_pitch_q15[0] = ric_yaw
    ric_cmd.yaw_pitch_q15[1] = -200
    ric_cmd.weapon_slot_u8 = 1

    hit_ric = arena_ric.resolve_subtick_fire(0, ric_cmd)
    assert (hit_ric.outcome_flags & NT_HIT_FLAG_RICOCHET) != 0
    assert arena_ric.struct.telemetry_and_header.active_projectile_count == 1
    for _ in range(3):
        arena_ric.step_kinematics()
    slot_r = arena_ric.server_tick & 63
    st_r = arena_ric.struct.tier_b_poses_plates[slot_r][1]
    plates_r = sum(int(x) for x in st_r.plate_hp_pools_q8)
    assert int(st_r.health_q8) < 25600 or plates_r < (12800 + 20480 + 17920 + 10240)


def test_v02_client_reconcile_pure_idempotent_and_zero_stamina() -> None:
    srv = NetTwinArenaBridge(auto_init=True)
    srv_full_stam = NetTwinArenaBridge(auto_init=True)
    srv.load_default_raid_map()
    srv_full_stam.load_default_raid_map()
    srv.spawn_player(0, -18.0, -25.0, 0.0, 16384)
    srv_full_stam.spawn_player(0, -18.0, -25.0, 0.0, 16384)

    # Exhaust stamina_q15 = 0 on srv while leaving srv_full_stam at 32767
    srv.struct.tier_b_poses_plates[0][0].stamina_q15 = 0
    auth_zero = srv.get_player_pose(0)
    assert (auth_zero.stance_flags & NT_POSE_HAS_STAMINA) != 0
    assert (auth_zero.stance_flags >> 8) == 0

    cmds_32 = []
    for i in range(32):
        c = ClientInputCmd()
        c.cmd_tick = i + 1
        c.buttons_stance = NT_BTN_SPRINT
        c.move_xy_q15[0] = 32767
        c.move_xy_q15[1] = 0
        c.yaw_pitch_q15[0] = 16384
        c.yaw_pitch_q15[1] = 0
        cmds_32.append(c)
        srv.ingest_client_cmd(0, c)
        srv.step_kinematics()
        srv_full_stam.ingest_client_cmd(0, c)
        srv_full_stam.step_kinematics()

    srv_final = srv.get_player_pose(0)
    full_stam_final = srv_full_stam.get_player_pose(0)
    assert (full_stam_final.y - srv_final.y) > 0.15, "Exhausted player should cover less distance than full-stamina sprinter"

    # Snapshot all 1,507,328 simulation state bytes (Tier-A, Tier-B, BVH, Gates, Slab, Baselines, Scratch)
    state_end = NET_TWIN_PARTITION_OFFSETS[7]  # 1,507,328 B
    state_blake3_before = blake3_256_hex(srv.np_bytes[:state_end])
    srv_hash_before_cli = srv.hash_arena_deterministic()

    cli = NetTwinArenaBridge(auto_init=True)
    cli.load_default_raid_map()
    cli.spawn_player(0, -18.0, -25.0, 0.0, 16384)
    cli_state_blake3_before = blake3_256_hex(cli.np_bytes[:state_end])

    # Run 10x repeated 32-tick reconciliations on cli AND 10x directly on srv
    cli_preds = [cli.client_reconcile(0, auth_zero, cmds_32) for _ in range(10)]
    assert srv.hash_arena_deterministic() == srv_hash_before_cli
    assert blake3_256_hex(cli.np_bytes[:state_end]) == cli_state_blake3_before

    preds = []
    srv_hashes = []
    for _ in range(10):
        preds.append(srv.client_reconcile(0, auth_zero, cmds_32))
        srv_hashes.append(srv.hash_arena_deterministic())

    assert blake3_256_hex(srv.np_bytes[:state_end]) == state_blake3_before, (
        "nt_client_reconcile mutated simulation state partitions (Tier-A / Tier-B)!"
    )
    assert len(set(srv_hashes)) == 1, "Repeated nt_client_reconcile calls mutated arena digest!"

    ref_bytes = bytes(preds[0])
    for idx, p in enumerate(preds[1:] + cli_preds, start=1):
        assert bytes(p) == ref_bytes, f"nt_client_reconcile non-idempotent on repetition {idx}"

    # Verify 0-ULP float bit equality against authoritative sprint-exhausted server stepping
    pred_floats = struct.pack("<ffffff", preds[0].x, preds[0].y, preds[0].z, preds[0].vx, preds[0].vy, preds[0].vz)
    srv_floats = struct.pack("<ffffff", srv_final.x, srv_final.y, srv_final.z, srv_final.vx, srv_final.vy, srv_final.vz)
    assert pred_floats == srv_floats, f"Stamina=0 32-tick reconcile ULP mismatch: {preds[0].y} vs {srv_final.y}"


def test_v02_subtick_shortest_arc_yaw_and_lean_interpolation() -> None:
    arena = NetTwinArenaBridge(auto_init=True)
    arena.load_default_raid_map()
    arena.spawn_player(0, -8.0, -5.0, 0.0, 0)
    arena.spawn_player(1, -2.0, -5.0, 0.0, 32000)

    arena.step_kinematics()  # tick 1
    arena.step_kinematics()  # tick 2

    # Victim flicks across ±pi yaw seam (+32000 -> -32000, shortest-arc midpoint = ±32768 facing -X)
    # while leaning Q/E from lean_q15 = 0 -> 8192 (+0.20 m full lean -> +0.10 m at alpha=0.5)
    arena.struct.tier_b_poses_plates[1][1].yaw_q15 = 32000
    arena.struct.tier_b_poses_plates[1][1].lean_q15 = 0
    arena.struct.tier_b_poses_plates[2][1].yaw_q15 = -32000
    arena.struct.tier_b_poses_plates[2][1].lean_q15 = 8192
    wide_cyl = (1800 << 16) | 650
    arena.struct.tier_a_cylinders[1][1].rad_height = wide_cyl
    arena.struct.tier_a_cylinders[2][1].rad_height = wide_cyl

    lib = load_native_library()
    # 1. Center-chest shot at alpha=0.5 (fire_subtick_u8=128) must hit FRONT thoracic plate (6 or 7),
    #    NOT back plate (8 or 9) from naive arithmetic yaw averaging (+32000 + -32000)/2 = 0!
    aim_rad = float(lib.nt_atan2_f32(ctypes.c_float(0.065), ctypes.c_float(6.0)))
    aim_yaw = int(round((aim_rad / math.pi) * 32768.0))

    shot_mid = ClientInputCmd()
    shot_mid.cmd_tick = 1
    shot_mid.buttons_stance = NT_BTN_FIRE
    shot_mid.yaw_pitch_q15[0] = aim_yaw
    shot_mid.yaw_pitch_q15[1] = -500
    shot_mid.fire_subtick_u8 = 128  # alpha = 0.5
    shot_mid.weapon_slot_u8 = 1

    hit_mid = arena.resolve_subtick_fire(0, shot_mid)
    assert hit_mid.victim_id == 1
    assert (hit_mid.outcome_flags & NT_HIT_FLAG_HIT) != 0
    assert hit_mid.plate_zone_id in (6, 7), f"Expected front thoracic plate (6/7), got {hit_mid.plate_zone_id}"

    # 2. Grazing shot at y_offset = +0.39 m (outside unleaned r=0.35 m cylinder at tick 1, but inside
    #    alpha=0.5 leaned torso shifted +0.065 m in +Y -> [ -0.285, +0.415 ]) must MISS at alpha=0 and HIT at alpha=0.5!
    graze_rad = float(lib.nt_atan2_f32(ctypes.c_float(0.39), ctypes.c_float(6.0)))
    graze_yaw = int(round((graze_rad / math.pi) * 32768.0))

    shot_t0 = ClientInputCmd()
    shot_t0.cmd_tick = 1
    shot_t0.buttons_stance = NT_BTN_FIRE
    shot_t0.yaw_pitch_q15[0] = graze_yaw
    shot_t0.yaw_pitch_q15[1] = -500
    shot_t0.fire_subtick_u8 = 0  # stale tick k (unleaned)
    shot_t0.weapon_slot_u8 = 1
    hit_t0 = arena.resolve_subtick_fire(0, shot_t0)
    assert hit_t0.victim_id == 0xFFFFFFFF, "Unleaned alpha=0 capsule should miss +0.39m lateral ray"

    shot_t05 = ClientInputCmd()
    shot_t05.cmd_tick = 1
    shot_t05.buttons_stance = NT_BTN_FIRE
    shot_t05.yaw_pitch_q15[0] = graze_yaw
    shot_t05.yaw_pitch_q15[1] = -500
    shot_t05.fire_subtick_u8 = 128  # alpha=0.5 interpolated lean
    shot_t05.weapon_slot_u8 = 1
    hit_t05 = arena.resolve_subtick_fire(0, shot_t05)
    assert hit_t05.victim_id == 1 and (hit_t05.outcome_flags & NT_HIT_FLAG_HIT) != 0


def test_v02_per_arena_simd_override_isolation() -> None:
    lib = load_native_library()
    assert int(lib.nt_set_simd_backend_override(None, ctypes.c_uint32(NT_SIMD_MODE_AUTO))) == 0

    arena_a = NetTwinArenaBridge(auto_init=True)
    arena_b = NetTwinArenaBridge(auto_init=True)

    # Force Scalar (mode=1) on Arena A ONLY; Arena B stays at default Auto-SIMD (mode=0)
    assert arena_a.set_simd_backend_override(NT_SIMD_MODE_FORCE_SCALAR) == 0
    assert arena_a.struct.telemetry_and_header.simd_mode_override == NT_SIMD_MODE_FORCE_SCALAR
    assert arena_b.struct.telemetry_and_header.simd_mode_override == NT_SIMD_MODE_AUTO

    # Also verify standalone Tier-0 dispatch was NOT polluted by Arena A's override
    a16 = (ctypes.c_float * 16)(*[float(i + 1) * 0.25 for i in range(16)])
    b16 = (ctypes.c_float * 16)(*[float(16 - i) * -0.5 for i in range(16)])
    d_auto = float(lib.nt_dot16_pairwise_f32(a16, b16))

    arena_a.load_default_raid_map()
    arena_b.load_default_raid_map()
    spawns = default_chaos_spawns(16)
    for pid, x, y, z, yaw in spawns:
        arena_a.spawn_player(pid, x, y, z, yaw)
        arena_b.spawn_player(pid, x, y, z, yaw)
        arena_a.set_player_rtt_ticks(pid, 10)
        arena_b.set_player_rtt_ticks(pid, 10)

    cmd_bytes = generate_chaos_command_bytes(128, 16, seed=0xB065_2026_0200)
    cmds = (ClientInputCmd * (128 * 16)).from_buffer_copy(cmd_bytes)

    for t in range(128):
        base = t * 16
        for p in range(16):
            arena_a.ingest_client_cmd(p, cmds[base + p])
            arena_b.ingest_client_cmd(p, cmds[base + p])
        arena_a.step_kinematics()
        arena_b.step_kinematics()
        for p in range(16):
            if cmds[base + p].buttons_stance & NT_BTN_FIRE:
                arena_a.resolve_subtick_fire(p, cmds[base + p])
                arena_b.resolve_subtick_fire(p, cmds[base + p])
        if (t & 7) == 0:
            pkt_a = arena_a.pack_client_datagram(t % 16, max(0, arena_a.server_tick - 4))
            pkt_b = arena_b.pack_client_datagram(t % 16, max(0, arena_b.server_tick - 4))
            assert pkt_a == pkt_b

        assert arena_a.struct.telemetry_and_header.simd_mode_override == NT_SIMD_MODE_FORCE_SCALAR
        assert arena_b.struct.telemetry_and_header.simd_mode_override == NT_SIMD_MODE_AUTO
        assert arena_a.hash_arena_deterministic() == arena_b.hash_arena_deterministic()

    assert arena_a.canonical_arena_blake3_hex(mask_simd_override=True) == arena_b.canonical_arena_blake3_hex(
        mask_simd_override=True
    )
    assert struct.pack("<f", float(lib.nt_dot16_pairwise_f32(a16, b16))) == struct.pack("<f", d_auto)


def test_v02_wasm32_simd128_binary_and_zero_ulp_lockstep(tmp_path: Path) -> None:
    za = audit_zero_alloc(require_both=True)
    wasm_info = za["wasm32"]
    assert wasm_info["passed"] is True
    assert wasm_info["total_imports"] == 0
    assert wasm_info["simd128_enabled"] is True
    assert wasm_info["simd128_0xfd_opcode_bytes"] >= 100

    out_json = tmp_path / "cross_isa_v02_test.json"
    res = verify_cross_isa_lockstep(
        num_ticks=256,
        num_players=64,
        output_path=str(out_json),
        sync_blake3_alias=False,
    )
    ps = res["parity_summary"]
    assert res["total_player_ticks"] == 256 * 64
    assert ps["mismatched_ticks_simd_vs_scalar"] == 0
    assert ps["mismatched_ticks_simd_vs_wasm32"] == 0
    assert ps["digest_x86_simd_blake3_256"] == ps["digest_scalar_blake3_256"] == ps["digest_wasm32_blake3_256"]
    assert ps["zero_ulp_cross_isa_verified"] is True

    archived_lockstep = _REPO_ROOT / "results" / "cross_isa_lockstep_receipts.json"
    if archived_lockstep.is_file():
        doc = json.loads(archived_lockstep.read_text(encoding="utf-8"))
        assert doc["total_player_ticks"] >= 160_000
        assert doc["parity_summary"]["zero_ulp_cross_isa_verified"] is True
        assert doc["wasm32_simd128_audit"]["simd128_0xfd_opcode_bytes"] >= 100

