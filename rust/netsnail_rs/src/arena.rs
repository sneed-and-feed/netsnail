//! Exact `#![no_std]` static `NetTwinArena` (`1,572,864 B` = `1.50 MB`, `align(64)`)
//! and frozen C-ABI wire/snapshot data structures.

use core::mem::{align_of, offset_of, size_of};

pub const NT_ARENA_BYTES: usize = 1_572_864;
pub const NT_ARENA_ALIGN: usize = 64;
pub const NT_TICK_HZ: u32 = 128;
pub const NT_DT_SEC: f32 = 1.0f32 / 128.0f32; // Exact IEEE-754 power-of-two (2^-7)
pub const NT_RING_TICKS: usize = 64;
pub const NT_MAX_PLAYERS: usize = 64;
pub const NT_MAX_PROJECTILES: usize = 256;
pub const NT_MAX_BRUSHES: usize = 2048;
pub const NT_MAX_BVH_NODES: usize = 4096;
pub const NT_MAX_PORTALS: usize = 512;
pub const NT_MAX_ACOUSTIC_EMITTERS: usize = 512;
pub const NT_PLATE_ZONES: usize = 30;
pub const NT_MTU_CEILING_BYTES: usize = 1200;
pub const NT_WASM_MTU_BUF_BYTES: usize = 1536;
pub const NT_MAGIC_TAG: u32 = 0x4E54_5731;

// Input button & stance flags
pub const NT_BTN_FIRE: u16 = 1 << 0;
pub const NT_BTN_ADS: u16 = 1 << 1;
pub const NT_BTN_CROUCH: u16 = 1 << 2;
pub const NT_BTN_SPRINT: u16 = 1 << 3;
pub const NT_BTN_LEAN_L: u16 = 1 << 4;
pub const NT_BTN_LEAN_R: u16 = 1 << 5;

// Internal stance flags stored in PlayerTickState::stance_flags
pub const NT_STANCE_ACTIVE: u16 = 1 << 0;
pub const NT_STANCE_ADS: u16 = 1 << 1;
pub const NT_STANCE_CROUCH: u16 = 1 << 2;
pub const NT_STANCE_SPRINT: u16 = 1 << 3;
pub const NT_STANCE_LEAN_L: u16 = 1 << 4;
pub const NT_STANCE_LEAN_R: u16 = 1 << 5;
pub const NT_STANCE_GROUNDED: u16 = 1 << 6;
pub const NT_STANCE_MASK_7BIT: u16 = 0x007F;
pub const NT_POSE_HAS_STAMINA: u16 = 1 << 7;

// HitRecordOut::outcome_flags
pub const NT_HIT_FLAG_HIT: u8 = 1 << 0;
pub const NT_HIT_FLAG_PENETRATED: u8 = 1 << 1;
pub const NT_HIT_FLAG_RICOCHET: u8 = 1 << 2;
pub const NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED: u8 = 1 << 3;
pub const NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED: u8 = 1 << 4;

// Morse Arrival-Jitter Gate Status
pub const NT_GATE_NOMINAL: u32 = 0;
pub const NT_GATE_PENDING_IMPULSE: u32 = 1;
pub const NT_GATE_STEP_MANEUVER: u32 = 2;
pub const NT_GATE_GLINT_REJECTED: u32 = 3;
pub const NT_GATE_LAG_SWITCH_QUARANTINED: u32 = 4;

// ============================================================================
// Frozen C-ABI Wire & Snapshot Structs
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientInputCmd {
    pub cmd_tick: u32,           // 0..4
    pub buttons_stance: u16,     // 4..6
    pub move_xy_q15: [i16; 2],   // 6..10
    pub yaw_pitch_q15: [i16; 2], // 10..14
    pub fire_subtick_u8: u8,     // 14..15
    pub weapon_slot_u8: u8,      // 15..16
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HitRecordOut {
    pub shooter_id: u32,    // 0..4
    pub victim_id: u32,     // 4..8
    pub rewind_tick: u32,   // 8..12
    pub plate_zone_id: u16, // 12..14
    pub outcome_flags: u8,  // 14..15
    pub _pad0: u8,          // 15..16
    pub subtick_alpha: f32, // 16..20
    pub damage_applied: f32,// 20..24
    pub armor_damage: f32,  // 24..28
    pub hit_dist_m: f32,    // 28..32
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcousticImpulse {
    pub arrival_tick: u32,       // 0..4
    pub azimuth_q12: i16,        // 4..6
    pub elevation_q12: i16,      // 6..8
    pub band_energy_q8: [u8; 4], // 8..12 (zero entity_id, zero xyz!)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerPoseSnapshot {
    pub x: f32,            // 0..4
    pub y: f32,            // 4..8
    pub z: f32,            // 8..12
    pub vx: f32,           // 12..16
    pub vy: f32,           // 16..20
    pub vz: f32,           // 20..24
    pub yaw_q15: i16,      // 24..26
    pub pitch_q15: i16,    // 26..28
    pub lean_q15: i16,     // 28..30
    pub stance_flags: u16, // 30..32
}

// ============================================================================
// L2 Rewind Ring & Arena Sub-Partitions
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TierACylinder {
    pub x: f32,
    pub y: f32,
    pub z_base: f32,
    pub rad_height: u32, // low u16: radius_mm, high u16: height_mm
}

impl TierACylinder {
    #[inline(always)]
    pub const fn pack_rad_height(radius_mm: u16, height_mm: u16) -> u32 {
        ((height_mm as u32) << 16) | (radius_mm as u32)
    }

    #[inline(always)]
    pub const fn radius_m(self) -> f32 {
        ((self.rad_height & 0xFFFF) as f32) * 0.001f32
    }

    #[inline(always)]
    pub const fn height_m(self) -> f32 {
        ((self.rad_height >> 16) as f32) * 0.001f32
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerTickState {
    pub pos: [f32; 3],               // 0..12
    pub vel: [f32; 3],               // 12..24
    pub yaw_q15: i16,                // 24..26
    pub pitch_q15: i16,              // 26..28
    pub lean_q15: i16,               // 28..30
    pub stance_flags: u16,           // 30..32
    pub cayley_gen_q15: [i16; 3],    // 32..38
    pub stamina_q15: u16,            // 38..40
    pub health_q8: u16,              // 40..42
    pub team_id: u8,                 // 42..43
    pub weapon_id: u8,               // 43..44
    pub plate_intact_mask: u32,      // 44..48 (bits 0..29 = 30 armor zones)
    pub plate_material_bits: u64,    // 48..56 (2 bits * 30 zones material tier)
    pub plate_hp_pools_q8: [u16; 4], // 56..64 (helmet, front, back/side, limb pools)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StaticBrushAabb {
    pub min: [f32; 3],
    pub flags: u32, // bit0 = solid hard cover
    pub max: [f32; 3],
    pub material_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BvhNode {
    pub min: [f32; 3],
    pub left_or_first: u32,
    pub max: [f32; 3],
    pub count: u32, // 0 = internal node (children at left_or_first, left_or_first + 1), >0 = leaf
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PortalAperture {
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    pub diffraction_loss_q8: u16,
    pub active: u16,
    pub normal_axis: u32, // 0 = X-normal doorway, 1 = Y-normal doorway
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct StaticMapHeader {
    pub num_brushes: u32,
    pub num_bvh_nodes: u32,
    pub num_portals: u32,
    pub initialized: u32,
    pub world_min: [f32; 3],
    pub cell_size_m: f32,
    pub inv_cell_size_m: f32,
    pub grid_dims: [u32; 3], // [128, 128, 16]
    pub brush_indices: [u16; NT_MAX_BRUSHES], // 4096 B permutation array for BVH leaves
    pub reserved: [u8; 12240],                // Pad to 16,384 B
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct StaticVoxelBvh {
    pub voxel_bits_u64: [u64; 4096],              // 32,768 B (128 * 128 * 16 1-bit grid)
    pub brushes: [StaticBrushAabb; NT_MAX_BRUSHES],// 65,536 B
    pub bvh_nodes: [BvhNode; NT_MAX_BVH_NODES],   // 131,072 B
    pub portals: [PortalAperture; NT_MAX_PORTALS],// 16,384 B
    pub header: StaticMapHeader,                  // 16,384 B
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ClientJitterGate {
    pub ack_mask_u64: u64,               // 0..8
    pub latest_cmd_tick: u32,            // 8..12
    pub last_arrival_server_tick: u32,   // 12..16
    pub est_rtt_ticks: u32,              // 16..20
    pub gate_status: u32,                // 20..24
    pub morse_j_prev2: f32,              // 24..28
    pub morse_j_prev1: f32,              // 28..32
    pub morse_j_curr: f32,               // 32..36
    pub morse_pending_j: f32,            // 36..40
    pub morse_impulse_last: f32,         // 40..44
    pub jitter_var_ema: f32,             // 44..48
    pub acklam_pullback_z: f32,          // 48..52
    pub h_clock_burst_score: f32,        // 52..56
    pub quarantine_ticks_remaining: u32, // 56..60
    pub clamped_rewind_max_ticks: u32,   // 60..64
    pub total_cmds_ingested: u32,        // 64..68
    pub total_burst_clumps: u32,         // 68..72
    pub total_quarantine_triggers: u32,  // 72..76
    pub hist_head: u32,                  // 76..80
    pub cmd_ring: [ClientInputCmd; 32],  // 80..592
    pub arrival_skew_hist: [f32; 64],    // 592..848
    pub reserved: [u8; 176],             // 848..1024
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct BallisticProjectile {
    pub pos: [f32; 3],         // 0..12
    pub vel: [f32; 3],         // 12..24
    pub drag_coeff: f32,       // 24..28
    pub mass_kg: f32,          // 28..32
    pub pen_power: f32,        // 32..36
    pub base_damage: f32,      // 36..40
    pub spawn_tick: u32,       // 40..44
    pub shooter_id: u16,       // 44..46
    pub weapon_id: u8,         // 46..47
    pub active: u8,            // 47..48
    pub traveled_dist_m: f32,  // 48..52
    pub ricochet_count: u32,   // 52..56
    pub _pad: [u8; 8],         // 56..64
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ServerAcousticEmitter {
    pub pos: [f32; 3],           // 0..12
    pub emit_tick: u32,          // 12..16
    pub source_player_id: u32,   // 16..20
    pub max_radius_m: f32,       // 20..24
    pub band_energy_q8: [u8; 4], // 24..28
    pub active: u32,             // 28..32
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ProjectilesAcousticSlab {
    pub projectiles: [BallisticProjectile; NT_MAX_PROJECTILES],     // 16,384 B
    pub emitters: [ServerAcousticEmitter; NT_MAX_ACOUSTIC_EMITTERS],// 16,384 B
    pub projectile_head: u32,
    pub emitter_head: u32,
    pub emitter_count: u32,
    pub _pad_hdr: [u8; 52],                                         // 64 B
    pub diffraction_scratch: [u8; 32704],                           // 32,704 B
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantizedEntityState {
    pub qx_2mm: i16,       // 0..2   : x in 2mm units ([-65.536m, +65.534m])
    pub qy_2mm: i16,       // 2..4   : y in 2mm units
    pub qz_2mm: i16,       // 4..6   : z in 2mm units
    pub qvx_16: i8,        // 6..7   : vx * 16
    pub qvy_16: i8,        // 7..8   : vy * 16
    pub qvz_16: i8,        // 8..9   : vz * 16
    pub yaw_q8: i8,        // 9..10  : yaw_q15 >> 8
    pub pitch_q8: i8,      // 10..11 : pitch_q15 >> 8
    pub lean_q8: i8,       // 11..12 : lean_q15 >> 8
    pub stance_u16: u16,   // 12..14
    pub valid: u16,        // 14..16
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ClientBaselineSnapshots {
    pub quantized_ring: [[QuantizedEntityState; NT_MAX_PLAYERS]; NT_RING_TICKS], // 65,536 B
    pub observer_pvs_ring: [[u64; NT_RING_TICKS]; NT_MAX_PLAYERS],               // 32,768 B
    pub observer_entity_cache: [[QuantizedEntityState; NT_MAX_PLAYERS]; NT_MAX_PLAYERS], // 65,536 B
    pub unpacked_acoustics: [AcousticImpulse; 256],                              // 3,072 B
    pub unpacked_acoustic_count: u32,
    pub unpacked_observer_id: u32,
    pub _pad_unp: [u8; 56],                                                      // 64 B
    pub reserved_slab: [u8; 357_312],                                            // 357,312 B
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct TriadCascadeScratch {
    pub triad_tensor_w: [f32; 4096],       // 16,384 B (so(16) skew tie-breaker tensor)
    pub cascade_w_rec: [f32; 4096],        // 16,384 B
    pub rotor_so16_a: [f32; 256],          // 1,024 B
    pub rotor_so16_r: [f32; 256],          // 1,024 B
    pub contact_planes: [[f32; 4]; 256],   // 4,096 B
    pub scratch_bytes: [u8; 223_232],      // 223,232 B
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct TelemetryAndHeader {
    pub server_tick: u32,              // 0..4
    pub simd_mode_override: u32,       // 4..8
    pub active_player_mask: u64,       // 8..16
    pub state_digest_u64: u64,         // 16..24
    pub total_steps: u32,              // 24..28
    pub total_shots_resolved: u32,     // 28..32
    pub total_hits_confirmed: u32,     // 32..36
    pub total_occluder_cutoffs: u32,   // 36..40
    pub total_jitter_quarantines: u32, // 40..44
    pub total_ricochets: u32,          // 44..48
    pub total_penetrations: u32,       // 48..52
    pub last_datagram_bytes: u32,      // 52..56
    pub last_pvs_visible_count: u32,   // 56..60
    pub last_acoustic_impulses: u32,   // 60..64
    pub max_cayley_norm_drift: f32,    // 64..68
    pub last_reconcile_ticks: u32,     // 68..72
    pub active_projectile_count: u32,  // 72..76
    pub magic_tag: u32,                // 76..80
    pub reserved_tail: [u8; 65456],    // 80..65536
}

/// Static 1.50 MB (`1,572,864 B`) 64-byte cache-line aligned deterministic arena.
#[repr(C, align(64))]
pub struct NetTwinArena {
    pub tier_a_cylinders: [[TierACylinder; NT_MAX_PLAYERS]; NT_RING_TICKS],   // Offset 0       (65,536 B)
    pub tier_b_poses_plates: [[PlayerTickState; NT_MAX_PLAYERS]; NT_RING_TICKS], // Offset 65,536  (262,144 B)
    pub static_voxel_bvh: StaticVoxelBvh,                                     // Offset 327,680 (262,144 B)
    pub client_jitter_gates: [ClientJitterGate; NT_MAX_PLAYERS],              // Offset 589,824 (65,536 B)
    pub projectiles_acoustic_slab: ProjectilesAcousticSlab,                   // Offset 655,360 (65,536 B)
    pub client_baseline_snapshots: ClientBaselineSnapshots,                   // Offset 720,896 (524,288 B)
    pub triad_cascade_scratch: TriadCascadeScratch,                           // Offset 1,245,184 (262,144 B)
    pub telemetry_and_header: TelemetryAndHeader,                             // Offset 1,507,328 (65,536 B)
}

// ============================================================================
// Mandatory Compile-Time Binary Layout & Alignment Assertions
// ============================================================================

const _: () = assert!(size_of::<ClientInputCmd>() == 16);
const _: () = assert!(align_of::<ClientInputCmd>() == 4);
const _: () = assert!(offset_of!(ClientInputCmd, cmd_tick) == 0);
const _: () = assert!(offset_of!(ClientInputCmd, buttons_stance) == 4);
const _: () = assert!(offset_of!(ClientInputCmd, move_xy_q15) == 6);
const _: () = assert!(offset_of!(ClientInputCmd, yaw_pitch_q15) == 10);
const _: () = assert!(offset_of!(ClientInputCmd, fire_subtick_u8) == 14);
const _: () = assert!(offset_of!(ClientInputCmd, weapon_slot_u8) == 15);

const _: () = assert!(size_of::<HitRecordOut>() == 32);
const _: () = assert!(align_of::<HitRecordOut>() == 4);
const _: () = assert!(offset_of!(HitRecordOut, shooter_id) == 0);
const _: () = assert!(offset_of!(HitRecordOut, victim_id) == 4);
const _: () = assert!(offset_of!(HitRecordOut, rewind_tick) == 8);
const _: () = assert!(offset_of!(HitRecordOut, plate_zone_id) == 12);
const _: () = assert!(offset_of!(HitRecordOut, outcome_flags) == 14);
const _: () = assert!(offset_of!(HitRecordOut, _pad0) == 15);
const _: () = assert!(offset_of!(HitRecordOut, subtick_alpha) == 16);
const _: () = assert!(offset_of!(HitRecordOut, damage_applied) == 20);
const _: () = assert!(offset_of!(HitRecordOut, armor_damage) == 24);
const _: () = assert!(offset_of!(HitRecordOut, hit_dist_m) == 28);

const _: () = assert!(size_of::<AcousticImpulse>() == 12);
const _: () = assert!(align_of::<AcousticImpulse>() == 4);
const _: () = assert!(offset_of!(AcousticImpulse, arrival_tick) == 0);
const _: () = assert!(offset_of!(AcousticImpulse, azimuth_q12) == 4);
const _: () = assert!(offset_of!(AcousticImpulse, elevation_q12) == 6);
const _: () = assert!(offset_of!(AcousticImpulse, band_energy_q8) == 8);

const _: () = assert!(size_of::<PlayerPoseSnapshot>() == 32);
const _: () = assert!(align_of::<PlayerPoseSnapshot>() == 4);
const _: () = assert!(offset_of!(PlayerPoseSnapshot, x) == 0);
const _: () = assert!(offset_of!(PlayerPoseSnapshot, vx) == 12);
const _: () = assert!(offset_of!(PlayerPoseSnapshot, yaw_q15) == 24);
const _: () = assert!(offset_of!(PlayerPoseSnapshot, stance_flags) == 30);

const _: () = assert!(size_of::<TierACylinder>() == 16);
const _: () = assert!(size_of::<PlayerTickState>() == 64);
const _: () = assert!(align_of::<PlayerTickState>() == 64);
const _: () = assert!(size_of::<StaticVoxelBvh>() == 262_144);
const _: () = assert!(size_of::<ClientJitterGate>() == 1024);
const _: () = assert!(size_of::<BallisticProjectile>() == 64);
const _: () = assert!(size_of::<ProjectilesAcousticSlab>() == 65_536);
const _: () = assert!(size_of::<QuantizedEntityState>() == 16);
const _: () = assert!(size_of::<ClientBaselineSnapshots>() == 524_288);
const _: () = assert!(size_of::<TriadCascadeScratch>() == 262_144);
const _: () = assert!(size_of::<TelemetryAndHeader>() == 65_536);

const _: () = assert!(size_of::<NetTwinArena>() == NT_ARENA_BYTES);
const _: () = assert!(size_of::<NetTwinArena>() == 24 * 65_536);
const _: () = assert!(align_of::<NetTwinArena>() == NT_ARENA_ALIGN);
const _: () = assert!(offset_of!(NetTwinArena, tier_a_cylinders) == 0);
const _: () = assert!(offset_of!(NetTwinArena, tier_b_poses_plates) == 65_536);
const _: () = assert!(offset_of!(NetTwinArena, static_voxel_bvh) == 327_680);
const _: () = assert!(offset_of!(NetTwinArena, client_jitter_gates) == 589_824);
const _: () = assert!(offset_of!(NetTwinArena, projectiles_acoustic_slab) == 655_360);
const _: () = assert!(offset_of!(NetTwinArena, client_baseline_snapshots) == 720_896);
const _: () = assert!(offset_of!(NetTwinArena, triad_cascade_scratch) == 1_245_184);
const _: () = assert!(offset_of!(NetTwinArena, telemetry_and_header) == 1_507_328);

/// Computes a deterministic 64-bit FNV-1a + Stafford Mix13 digest over canonical `NetTwinArena`
/// bytes, skipping `simd_mode_override` (`1507332..1507336`) and `state_digest_u64` (`1507344..1507352`)
/// so `digest(SIMD) == digest(Scalar) == digest(WASM32)` to 0 bits.
pub fn hash_arena_canonical(arena: &NetTwinArena) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let base_ptr = (arena as *const NetTwinArena) as *const u8;
    let skip_simd_start = offset_of!(NetTwinArena, telemetry_and_header)
        + offset_of!(TelemetryAndHeader, simd_mode_override);
    let skip_digest_start = offset_of!(NetTwinArena, telemetry_and_header)
        + offset_of!(TelemetryAndHeader, state_digest_u64);

    let mut h: u64 = FNV_OFFSET;

    // Hash 8 bytes per step in little-endian canonical order across the 3 contiguous regions
    let regions: [(usize, usize); 3] = [
        (0, skip_simd_start),
        (skip_simd_start + 4, skip_digest_start),
        (skip_digest_start + 8, NT_ARENA_BYTES),
    ];

    for &(start, end) in &regions {
        let mut idx = start;
        while idx + 8 <= end {
            let word = unsafe {
                let b = core::slice::from_raw_parts(base_ptr.add(idx), 8);
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            };
            h ^= word;
            h = h.wrapping_mul(FNV_PRIME);
            idx += 8;
        }
        while idx < end {
            let byte = unsafe { *base_ptr.add(idx) } as u64;
            h ^= byte;
            h = h.wrapping_mul(FNV_PRIME);
            idx += 1;
        }
    }

    // Final SplitMix64 / Stafford Mix13 avalanche
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// Returns `true` when portable scalar execution is forced for `arena`, checking
/// `arena.telemetry_and_header.simd_mode_override` first and falling back to
/// `GLOBAL_SIMD_OVERRIDE` only when per-arena override is `0`.
#[inline(always)]
pub fn arena_force_scalar(arena: &NetTwinArena) -> bool {
    if arena.telemetry_and_header.simd_mode_override != 0 {
        true
    } else {
        crate::math::GLOBAL_SIMD_OVERRIDE.load(core::sync::atomic::Ordering::Relaxed) != 0
    }
}

/// Initializes an already allocated `NetTwinArena` in-place with zero stack temporaries.
///
/// # Safety
/// `arena` must point to a valid writable `NetTwinArena` of `1,572,864` bytes aligned to `64` bytes.
pub unsafe fn init_arena_in_place(arena: *mut NetTwinArena) {
    core::ptr::write_bytes(arena as *mut u8, 0, NT_ARENA_BYTES);
    let a = &mut *arena;

    // Initialize SO(16) rotor R to identity and canonical skew tie-breaker weights
    for i in 0..16 {
        a.triad_cascade_scratch.rotor_so16_r[i * 16 + i] = 1.0f32;
    }
    for m in 0..16 {
        let next = (m + 1) & 15;
        a.triad_cascade_scratch.triad_tensor_w[m * 256 + m * 16 + next] = 0.25f32;
        a.triad_cascade_scratch.triad_tensor_w[m * 256 + next * 16 + m] = -0.25f32;
    }

    // Initialize per-player Morse arrival-jitter gates with realistic defaults
    for p in 0..NT_MAX_PLAYERS {
        let g = &mut a.client_jitter_gates[p];
        g.est_rtt_ticks = 6; // ~46.9 ms initial nominal RTT prior
        g.gate_status = NT_GATE_NOMINAL;
        g.jitter_var_ema = 1.0f32;
        g.clamped_rewind_max_ticks = 32; // up to 250 ms nominal rewind credit
    }

    a.telemetry_and_header.magic_tag = NT_MAGIC_TAG;
    a.telemetry_and_header.simd_mode_override = 0;
    a.telemetry_and_header.server_tick = 0;
    a.telemetry_and_header.state_digest_u64 = hash_arena_canonical(a);
}
