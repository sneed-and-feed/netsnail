#ifndef NET_TWIN_H
#define NET_TWIN_H

/*
 * net_twin.h — C-ABI Header for NetSnail v0.2.0 (netsnail_rs)
 * Deterministic 128 Hz Multiplayer Causal Twin, Sub-Tick 30-Plate Rewind,
 * Morse Jitter Gate, 1-Bit Voxel DDA + Static BVH, and Minkowski Reachability
 * Anti-DMA Serializer.
 *
 * Zero dynamic heap allocation (#![no_std]). Exact 1.50 MB (1,572,864 B)
 * 64-byte cache-line aligned static arena.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Compile-Time Arena & Protocol Constants
 * ============================================================================ */

#define NT_ARENA_BYTES                 1572864u
#define NT_ARENA_ALIGN                 64u
#define NT_TICK_HZ                     128u
#define NT_RING_TICKS                  64u
#define NT_MAX_PLAYERS                 64u
#define NT_MAX_PROJECTILES             256u
#define NT_PLATE_ZONES                 30u
#define NT_MTU_CEILING_BYTES           1200u
#define NT_WASM_MTU_BUF_BYTES          1536u
#define NT_MAGIC_TAG                   0x4E545731u

/* Exact 8-Partition Byte Offsets in NetTwinArena */
#define NT_OFFSET_TIER_A_CYLINDERS          0u
#define NT_OFFSET_TIER_B_POSES_PLATES       65536u
#define NT_OFFSET_STATIC_VOXEL_BVH          327680u
#define NT_OFFSET_CLIENT_JITTER_GATES       589824u
#define NT_OFFSET_PROJECTILES_ACOUSTIC_SLAB 655360u
#define NT_OFFSET_CLIENT_BASELINE_SNAPSHOTS 720896u
#define NT_OFFSET_TRIAD_CASCADE_SCRATCH     1245184u
#define NT_OFFSET_TELEMETRY_AND_HEADER      1507328u

/* Exact 8-Partition Byte Sizes in NetTwinArena */
#define NT_SIZE_TIER_A_CYLINDERS            65536u
#define NT_SIZE_TIER_B_POSES_PLATES         262144u
#define NT_SIZE_STATIC_VOXEL_BVH            262144u
#define NT_SIZE_CLIENT_JITTER_GATES         65536u
#define NT_SIZE_PROJECTILES_ACOUSTIC_SLAB   65536u
#define NT_SIZE_CLIENT_BASELINE_SNAPSHOTS   524288u
#define NT_SIZE_TRIAD_CASCADE_SCRATCH       262144u
#define NT_SIZE_TELEMETRY_AND_HEADER        65536u

/* ClientInputCmd::buttons_stance & PlayerPoseSnapshot::stance_flags bitmasks */
#define NT_BTN_FIRE                    (1u << 0)
#define NT_BTN_ADS                     (1u << 1)
#define NT_BTN_CROUCH                  (1u << 2)
#define NT_BTN_SPRINT                  (1u << 3)
#define NT_BTN_LEAN_L                  (1u << 4)
#define NT_BTN_LEAN_R                  (1u << 5)
#define NT_STANCE_MASK_7BIT            0x007Fu
#define NT_POSE_HAS_STAMINA            (1u << 7) /* high byte carries stamina_q15 >> 7 */

/* HitRecordOut::outcome_flags bitmask flags */
#define NT_HIT_FLAG_HIT                        (1u << 0) /* 0x01 */
#define NT_HIT_FLAG_PENETRATED                 (1u << 1) /* 0x02 */
#define NT_HIT_FLAG_RICOCHET                   (1u << 2) /* 0x04 */
#define NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED   (1u << 3) /* 0x08 */
#define NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED  (1u << 4) /* 0x10 */

/* SIMD Backend Override Modes */
#define NT_SIMD_MODE_AUTO              0u
#define NT_SIMD_MODE_FORCE_SCALAR      1u

/* Morse Arrival-Jitter Gate Status Codes */
#define NT_GATE_NOMINAL                0
#define NT_GATE_PENDING_IMPULSE        1
#define NT_GATE_STEP_MANEUVER          2
#define NT_GATE_GLINT_REJECTED         3
#define NT_GATE_LAG_SWITCH_QUARANTINED 4

#if defined(_MSC_VER)
#define NT_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define NT_ALIGN64 __attribute__((aligned(64)))
#else
#define NT_ALIGN64
#endif

/* ============================================================================
 * Frozen C-ABI Wire & Snapshot Structs (Zero Implicit Padding)
 * ============================================================================ */

/* 16-byte client input command */
typedef struct ClientInputCmd {
    uint32_t cmd_tick;         /* 0..4   */
    uint16_t buttons_stance;   /* 4..6   : bit0=fire,1=ads,2=crouch,3=sprint,4=lean_l,5=lean_r */
    int16_t  move_xy_q15[2];   /* 6..10  : forward/strafe in [-32767, +32767] */
    int16_t  yaw_pitch_q15[2]; /* 10..14 : modular yaw (-pi..pi) & clamped pitch */
    uint8_t  fire_subtick_u8;  /* 14..15 : sub-tick alpha = fire_subtick_u8 / 256.0f */
    uint8_t  weapon_slot_u8;   /* 15..16 : weapon ballistic profile index */
} ClientInputCmd;

/* 32-byte server authoritative sub-tick hit & armor evaluation record */
typedef struct HitRecordOut {
    uint32_t shooter_id;       /* 0..4   */
    uint32_t victim_id;        /* 4..8   */
    uint32_t rewind_tick;      /* 8..12  */
    uint16_t plate_zone_id;    /* 12..14 : 0..29 anatomical/armor zone (0xFFFF if miss) */
    uint8_t  outcome_flags;    /* 14..15 : NT_HIT_FLAG_* bitmask */
    uint8_t  _pad0;            /* 15..16 : explicit zero pad */
    float    subtick_alpha;    /* 16..20 */
    float    damage_applied;   /* 20..24 */
    float    armor_damage;     /* 24..28 */
    float    hit_dist_m;       /* 28..32 */
} HitRecordOut;

/* 12-byte anonymous portal-diffracted acoustic impulse (zero entity_id, zero xyz) */
typedef struct AcousticImpulse {
    uint32_t arrival_tick;     /* 0..4   */
    int16_t  azimuth_q12;      /* 4..6   : [-pi, pi] quantized in Q3.12 */
    int16_t  elevation_q12;    /* 6..8   : [-pi/2, pi/2] quantized in Q3.12 */
    uint8_t  band_energy_q8[4];/* 8..12  : 4-band octave energy (125Hz, 500Hz, 2kHz, 8kHz) */
} AcousticImpulse;

/* 32-byte clean player kinematic & attitude readback snapshot */
typedef struct PlayerPoseSnapshot {
    float    x;                /* 0..4   */
    float    y;                /* 4..8   */
    float    z;                /* 8..12  */
    float    vx;               /* 12..16 */
    float    vy;               /* 16..20 */
    float    vz;               /* 20..24 */
    int16_t  yaw_q15;          /* 24..26 */
    int16_t  pitch_q15;        /* 26..28 */
    int16_t  lean_q15;         /* 28..30 */
    uint16_t stance_flags;     /* 30..32 */
} PlayerPoseSnapshot;

/* ============================================================================
 * Internal L2 Ring & Arena Partition Structs
 * ============================================================================ */

/* 16-byte Tier-A swept broadphase cylinder */
typedef struct TierACylinder {
    float    x;
    float    y;
    float    z_base;
    uint32_t rad_height;       /* low u16: radius_mm, high u16: height_mm */
} TierACylinder;

/* 64-byte cache-line-aligned Tier-B pose & 30-zone multi-plate armor state */
typedef struct NT_ALIGN64 PlayerTickState {
    float    pos[3];               /* 0..12  */
    float    vel[3];               /* 12..24 */
    int16_t  yaw_q15;              /* 24..26 */
    int16_t  pitch_q15;            /* 26..28 */
    int16_t  lean_q15;             /* 28..30 */
    uint16_t stance_flags;         /* 30..32 */
    int16_t  cayley_gen_q15[3];    /* 32..38 */
    uint16_t stamina_q15;          /* 38..40 */
    uint16_t health_q8;            /* 40..42 */
    uint8_t  team_id;              /* 42..43 */
    uint8_t  weapon_id;            /* 43..44 */
    uint32_t plate_intact_mask;    /* 44..48 : bits 0..29 active armor plates */
    uint64_t plate_material_bits;  /* 48..56 : 2 bits * 30 zones material class */
    uint16_t plate_hp_pools_q8[4]; /* 56..64 : helmet, front, back/side, limb pools */
} PlayerTickState;

/* 1024-byte per-player 3-point Morse arrival-jitter & lag-switch gate */
typedef struct NT_ALIGN64 ClientJitterGate {
    uint64_t       ack_mask_u64;              /* 0..8     */
    uint32_t       latest_cmd_tick;           /* 8..12    */
    uint32_t       last_arrival_server_tick;  /* 12..16   */
    uint32_t       est_rtt_ticks;             /* 16..20   */
    uint32_t       gate_status;               /* 20..24   */
    float          morse_j_prev2;             /* 24..28   */
    float          morse_j_prev1;             /* 28..32   */
    float          morse_j_curr;              /* 32..36   */
    float          morse_pending_j;           /* 36..40   */
    float          morse_impulse_last;        /* 40..44   */
    float          jitter_var_ema;            /* 44..48   */
    float          acklam_pullback_z;         /* 48..52   */
    float          h_clock_burst_score;       /* 52..56   */
    uint32_t       quarantine_ticks_remaining;/* 56..60   */
    uint32_t       clamped_rewind_max_ticks;  /* 60..64   */
    uint32_t       total_cmds_ingested;       /* 64..68   */
    uint32_t       total_burst_clumps;        /* 68..72   */
    uint32_t       total_quarantine_triggers; /* 72..76   */
    uint32_t       hist_head;                 /* 76..80   */
    ClientInputCmd cmd_ring[32];              /* 80..592  */
    float          arrival_skew_hist[64];     /* 592..848 */
    uint8_t        reserved[176];             /* 848..1024*/
} ClientJitterGate;

/* 65536-byte telemetry & arena control header (offset 1,507,328) */
typedef struct NT_ALIGN64 TelemetryAndHeader {
    uint32_t server_tick;               /* 0..4    */
    uint32_t simd_mode_override;        /* 4..8    : 0=auto SIMD, 1=force scalar */
    uint64_t active_player_mask;        /* 8..16   */
    uint64_t state_digest_u64;          /* 16..24  */
    uint32_t total_steps;               /* 24..28  */
    uint32_t total_shots_resolved;      /* 28..32  */
    uint32_t total_hits_confirmed;      /* 32..36  */
    uint32_t total_occluder_cutoffs;    /* 36..40  */
    uint32_t total_jitter_quarantines;  /* 40..44  */
    uint32_t total_ricochets;           /* 44..48  */
    uint32_t total_penetrations;        /* 48..52  */
    uint32_t last_datagram_bytes;       /* 52..56  */
    uint32_t last_pvs_visible_count;    /* 56..60  */
    uint32_t last_acoustic_impulses;    /* 60..64  */
    float    max_cayley_norm_drift;     /* 64..68  */
    uint32_t last_reconcile_ticks;      /* 68..72  */
    uint32_t active_projectile_count;   /* 72..76  */
    uint32_t magic_tag;                 /* 76..80  */
    uint8_t  reserved_tail[65456];      /* 80..65536 */
} TelemetryAndHeader;

/* Exact 1,572,864-byte (1.50 MB) static arena */
typedef struct NT_ALIGN64 NetTwinArena {
    TierACylinder      tier_a_cylinders[NT_RING_TICKS][NT_MAX_PLAYERS];   /* 0..65536           */
    PlayerTickState    tier_b_poses_plates[NT_RING_TICKS][NT_MAX_PLAYERS];/* 65536..327680      */
    uint8_t            static_voxel_bvh[NT_SIZE_STATIC_VOXEL_BVH];        /* 327680..589824     */
    ClientJitterGate   client_jitter_gates[NT_MAX_PLAYERS];               /* 589824..655360     */
    uint8_t            projectiles_acoustic_slab[NT_SIZE_PROJECTILES_ACOUSTIC_SLAB]; /* 655360..720896 */
    uint8_t            client_baseline_snapshots[NT_SIZE_CLIENT_BASELINE_SNAPSHOTS]; /* 720896..1245184 */
    uint8_t            triad_cascade_scratch[NT_SIZE_TRIAD_CASCADE_SCRATCH];         /* 1245184..1507328 */
    TelemetryAndHeader telemetry_and_header;                              /* 1507328..1572864   */
} NetTwinArena;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(ClientInputCmd) == 16, "ClientInputCmd must be 16 bytes");
_Static_assert(sizeof(HitRecordOut) == 32, "HitRecordOut must be 32 bytes");
_Static_assert(sizeof(AcousticImpulse) == 12, "AcousticImpulse must be 12 bytes");
_Static_assert(sizeof(PlayerPoseSnapshot) == 32, "PlayerPoseSnapshot must be 32 bytes");
_Static_assert(sizeof(TierACylinder) == 16, "TierACylinder must be 16 bytes");
_Static_assert(sizeof(PlayerTickState) == 64, "PlayerTickState must be 64 bytes");
_Static_assert(sizeof(ClientJitterGate) == 1024, "ClientJitterGate must be 1024 bytes");
_Static_assert(sizeof(TelemetryAndHeader) == 65536, "TelemetryAndHeader must be 65536 bytes");
_Static_assert(sizeof(NetTwinArena) == 1572864, "NetTwinArena must be 1,572,864 bytes");
_Static_assert(offsetof(NetTwinArena, tier_a_cylinders) == 0, "offset 0 mismatch");
_Static_assert(offsetof(NetTwinArena, tier_b_poses_plates) == 65536, "offset 65536 mismatch");
_Static_assert(offsetof(NetTwinArena, static_voxel_bvh) == 327680, "offset 327680 mismatch");
_Static_assert(offsetof(NetTwinArena, client_jitter_gates) == 589824, "offset 589824 mismatch");
_Static_assert(offsetof(NetTwinArena, projectiles_acoustic_slab) == 655360, "offset 655360 mismatch");
_Static_assert(offsetof(NetTwinArena, client_baseline_snapshots) == 720896, "offset 720896 mismatch");
_Static_assert(offsetof(NetTwinArena, triad_cascade_scratch) == 1245184, "offset 1245184 mismatch");
_Static_assert(offsetof(NetTwinArena, telemetry_and_header) == 1507328, "offset 1507328 mismatch");
#endif

/* ============================================================================
 * Exported C-ABI Functions (Native & wasm32-unknown-unknown)
 * ============================================================================ */

/* Arena Lifecycle, Map & Player Management */
int32_t  nt_arena_init(NetTwinArena *arena);
int32_t  nt_set_simd_backend_override(NetTwinArena *arena, uint32_t mode);
int32_t  nt_load_default_raid_map(NetTwinArena *arena);
int32_t  nt_spawn_player(
    NetTwinArena *arena,
    uint32_t player_id,
    float x,
    float y,
    float z,
    int16_t yaw_q15
);
int32_t  nt_get_player_pose(
    const NetTwinArena *arena,
    uint32_t player_id,
    PlayerPoseSnapshot *out_pose
);
int32_t  nt_set_player_rtt_ticks(
    NetTwinArena *arena,
    uint32_t player_id,
    uint32_t rtt_ticks
);
int32_t  nt_emit_acoustic_event(
    NetTwinArena *arena,
    uint32_t source_player_id,
    float x,
    float y,
    float z,
    const uint8_t band_energy_q8[4]
);

/* Tick Ingestion, Swept Kinematics, Client Reconciliation & Sub-Tick Rewind */
int32_t  nt_ingest_client_cmd(
    NetTwinArena *arena,
    uint32_t player_id,
    const ClientInputCmd *cmd
);
int32_t  nt_step_kinematics(NetTwinArena *arena);
int32_t  nt_client_reconcile(
    NetTwinArena *arena,
    uint32_t player_id,
    const PlayerPoseSnapshot *auth_pose,
    const ClientInputCmd *pending_cmds,
    uint32_t num_cmds,
    PlayerPoseSnapshot *out_predicted_pose
);
int32_t  nt_resolve_subtick_fire(
    NetTwinArena *arena,
    uint32_t shooter_id,
    const ClientInputCmd *fire_cmd,
    HitRecordOut *out_hit
);

/* Minkowski Reachability PVS + Anonymous Acoustic + <1200B Delta Bitpacker */
int32_t  nt_pack_client_datagram(
    NetTwinArena *arena,
    uint32_t observer_id,
    uint32_t baseline_ack_tick,
    uint8_t *out_mtu_buf,
    uint32_t max_buf_len
);
int32_t  nt_unpack_client_datagram(
    NetTwinArena *client_arena,
    const uint8_t *in_mtu_buf,
    uint32_t buf_len,
    AcousticImpulse *out_acoustics,
    uint32_t max_acoustics,
    uint32_t *out_num_acoustics
);
uint64_t nt_hash_arena_deterministic(const NetTwinArena *arena);

/* Cross-ISA 0-ULP Tier-0 Numerical Kernels */
float    nt_sin_q15(int16_t phase_q15);
float    nt_cos_q15(int16_t phase_q15);
float    nt_atan2_f32(float y, float x);
float    nt_sqrt_f32(float x);
float    nt_dot16_pairwise_f32(const float *a, const float *b);
int32_t  nt_cayley_rotor_so3_f32(
    const float *omega_3,
    const float *v_in_3,
    float *r_out_9,
    float *v_out_3
);

/* Zero-Alloc Static Bridge Helpers (WASM32 & Native) */
NetTwinArena    *nt_wasm_global_arena(void);
NetTwinArena    *nt_wasm_client_arena(void);
ClientInputCmd  *nt_wasm_cmd_buf(void);
uint8_t         *nt_wasm_mtu_buf(void);
HitRecordOut    *nt_wasm_hit_buf(void);
AcousticImpulse *nt_wasm_acoustic_buf(void);

#ifdef __cplusplus
}
#endif

#endif /* NET_TWIN_H */
