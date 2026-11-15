//! Static 1-bit Hard-Cover Voxel Grid (`128x128x16`), Static AABB Brush BVH,
//! Mutual Occluder Raycaster, Minkowski Reachability PVS, and Portal Acoustic Diffraction.

use crate::arena::{
    BvhNode, PortalAperture, StaticBrushAabb, StaticVoxelBvh, NT_MAX_BRUSHES, NT_MAX_BVH_NODES,
    NT_MAX_PORTALS,
};
use crate::math::{abs_f32, atan2_f32_deterministic, canon_f32, sqrt_f32_deterministic};

pub const VOXEL_NX: usize = 128;
pub const VOXEL_NY: usize = 128;
pub const VOXEL_NZ: usize = 16;
pub const VOXEL_CELL_M: f32 = 0.5f32;
pub const VOXEL_INV_CELL_M: f32 = 2.0f32;
pub const WORLD_MIN_X: f32 = -32.0f32;
pub const WORLD_MIN_Y: f32 = -32.0f32;
pub const WORLD_MIN_Z: f32 = 0.0f32;

#[inline]
pub fn set_voxel_bit(map: &mut StaticVoxelBvh, ix: usize, iy: usize, iz: usize) {
    if ix < VOXEL_NX && iy < VOXEL_NY && iz < VOXEL_NZ {
        let bit_index = (iz * VOXEL_NY + iy) * VOXEL_NX + ix;
        let word = bit_index >> 6;
        let shift = bit_index & 63;
        map.voxel_bits_u64[word] |= 1u64 << shift;
    }
}

#[inline]
pub fn get_voxel_bit(map: &StaticVoxelBvh, ix: i32, iy: i32, iz: i32) -> bool {
    if ix < 0
        || iy < 0
        || iz < 0
        || (ix as usize) >= VOXEL_NX
        || (iy as usize) >= VOXEL_NY
        || (iz as usize) >= VOXEL_NZ
    {
        return false;
    }
    let bit_index = ((iz as usize) * VOXEL_NY + (iy as usize)) * VOXEL_NX + (ix as usize);
    let word = bit_index >> 6;
    let shift = bit_index & 63;
    ((map.voxel_bits_u64[word] >> shift) & 1) != 0
}

pub fn add_brush_aabb(
    map: &mut StaticVoxelBvh,
    min: [f32; 3],
    max: [f32; 3],
    material_id: u32,
) -> bool {
    let idx = map.header.num_brushes as usize;
    if idx >= NT_MAX_BRUSHES {
        return false;
    }
    let clean_min = [canon_f32(min[0]), canon_f32(min[1]), canon_f32(min[2])];
    let clean_max = [canon_f32(max[0]), canon_f32(max[1]), canon_f32(max[2])];

    map.brushes[idx] = StaticBrushAabb {
        min: clean_min,
        flags: 1,
        max: clean_max,
        material_id,
    };
    map.header.brush_indices[idx] = idx as u16;
    map.header.num_brushes += 1;

    // Rasterize hard-cover brush into the 128 x 128 x 16 1-bit voxel grid
    let x0 = (((clean_min[0] - WORLD_MIN_X) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NX - 1) as i32) as usize;
    let x1 = (((clean_max[0] - WORLD_MIN_X - 1e-4) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NX - 1) as i32) as usize;
    let y0 = (((clean_min[1] - WORLD_MIN_Y) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NY - 1) as i32) as usize;
    let y1 = (((clean_max[1] - WORLD_MIN_Y - 1e-4) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NY - 1) as i32) as usize;
    let z0 = (((clean_min[2] - WORLD_MIN_Z) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NZ - 1) as i32) as usize;
    let z1 = (((clean_max[2] - WORLD_MIN_Z - 1e-4) * VOXEL_INV_CELL_M) as i32).clamp(0, (VOXEL_NZ - 1) as i32) as usize;

    if clean_max[0] > clean_min[0] && clean_max[1] > clean_min[1] && clean_max[2] > clean_min[2] {
        for iz in z0..=z1 {
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    set_voxel_bit(map, ix, iy, iz);
                }
            }
        }
    }
    true
}

pub fn add_portal_aperture(
    map: &mut StaticVoxelBvh,
    center: [f32; 3],
    half_extents: [f32; 3],
    normal_axis: u32,
    diffraction_loss_q8: u16,
) -> bool {
    let idx = map.header.num_portals as usize;
    if idx >= NT_MAX_PORTALS {
        return false;
    }
    map.portals[idx] = PortalAperture {
        center: [canon_f32(center[0]), canon_f32(center[1]), canon_f32(center[2])],
        half_extents: [
            canon_f32(half_extents[0]),
            canon_f32(half_extents[1]),
            canon_f32(half_extents[2]),
        ],
        diffraction_loss_q8,
        active: 1,
        normal_axis,
    };
    map.header.num_portals += 1;
    true
}

/// Zero-allocation in-place Shell sort on `brush_indices[start..end]` by brush centroid along `axis`.
fn sort_brush_slice_by_axis(
    brushes: &[StaticBrushAabb; NT_MAX_BRUSHES],
    indices: &mut [u16],
    axis: usize,
) {
    let len = indices.len();
    let mut gap = len / 2;
    while gap > 0 {
        for i in gap..len {
            let mut j = i;
            while j >= gap {
                let ia = indices[j - gap] as usize;
                let ib = indices[j] as usize;
                let ca = brushes[ia].min[axis] + brushes[ia].max[axis];
                let cb = brushes[ib].min[axis] + brushes[ib].max[axis];
                if ca > cb || (ca == cb && ia > ib) {
                    indices.swap(j - gap, j);
                    j -= gap;
                } else {
                    break;
                }
            }
        }
        gap /= 2;
    }
}

/// Builds a flat static median-split BVH over `map.brushes[0..num_brushes]` with zero heap allocation.
pub fn rebuild_static_bvh(map: &mut StaticVoxelBvh) {
    let total = map.header.num_brushes as usize;
    if total == 0 {
        map.header.num_bvh_nodes = 0;
        return;
    }

    // Iterative build work-item: (node_idx, first_brush_offset, count)
    let mut work_stack = [(0u32, 0u32, 0u32); 64];
    let mut sp = 0usize;

    map.header.num_bvh_nodes = 1;
    work_stack[0] = (0, 0, total as u32);
    sp += 1;

    while sp > 0 {
        sp -= 1;
        let (node_idx, first, count) = work_stack[sp];
        let start = first as usize;
        let end = start + (count as usize);

        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        for k in start..end {
            let b = &map.brushes[map.header.brush_indices[k] as usize];
            for ax in 0..3 {
                if b.min[ax] < bmin[ax] {
                    bmin[ax] = b.min[ax];
                }
                if b.max[ax] > bmax[ax] {
                    bmax[ax] = b.max[ax];
                }
            }
        }

        let can_split = count > 2
            && (map.header.num_bvh_nodes as usize + 2) <= NT_MAX_BVH_NODES
            && (sp + 2) <= work_stack.len();

        if !can_split {
            map.bvh_nodes[node_idx as usize] = BvhNode {
                min: [canon_f32(bmin[0]), canon_f32(bmin[1]), canon_f32(bmin[2])],
                left_or_first: first,
                max: [canon_f32(bmax[0]), canon_f32(bmax[1]), canon_f32(bmax[2])],
                count,
            };
        } else {
            let ext_x = bmax[0] - bmin[0];
            let ext_y = bmax[1] - bmin[1];
            let ext_z = bmax[2] - bmin[2];
            let axis = if ext_x >= ext_y && ext_x >= ext_z {
                0
            } else if ext_y >= ext_z {
                1
            } else {
                2
            };

            sort_brush_slice_by_axis(
                &map.brushes,
                &mut map.header.brush_indices[start..end],
                axis,
            );

            let mid_count = count / 2;
            let left_child = map.header.num_bvh_nodes;
            let right_child = left_child + 1;
            map.header.num_bvh_nodes += 2;

            map.bvh_nodes[node_idx as usize] = BvhNode {
                min: [canon_f32(bmin[0]), canon_f32(bmin[1]), canon_f32(bmin[2])],
                left_or_first: left_child,
                max: [canon_f32(bmax[0]), canon_f32(bmax[1]), canon_f32(bmax[2])],
                count: 0, // internal node
            };

            work_stack[sp] = (right_child, first + mid_count, count - mid_count);
            sp += 1;
            work_stack[sp] = (left_child, first, mid_count);
            sp += 1;
        }
    }
}

/// Populates `StaticVoxelBvh` with the default tactical multi-room raid hallway + central pillar map.
pub fn populate_default_raid_map(map: &mut StaticVoxelBvh) {
    map.voxel_bits_u64.fill(0);
    map.header.num_brushes = 0;
    map.header.num_bvh_nodes = 0;
    map.header.num_portals = 0;
    map.header.world_min = [WORLD_MIN_X, WORLD_MIN_Y, WORLD_MIN_Z];
    map.header.cell_size_m = VOXEL_CELL_M;
    map.header.inv_cell_size_m = VOXEL_INV_CELL_M;
    map.header.grid_dims = [VOXEL_NX as u32, VOXEL_NY as u32, VOXEL_NZ as u32];

    // 1. Perimeter outer reinforced concrete walls ([-31, 31] x [-31, 31] x [0, 4])
    add_brush_aabb(map, [-32.0, -32.0, 0.0], [-31.0, 32.0, 4.0], 1);
    add_brush_aabb(map, [31.0, -32.0, 0.0], [32.0, 32.0, 4.0], 1);
    add_brush_aabb(map, [-31.0, -32.0, 0.0], [31.0, -31.0, 4.0], 1);
    add_brush_aabb(map, [-31.0, 31.0, 0.0], [31.0, 32.0, 4.0], 1);

    // 2. Central Structural Concrete Pillar at origin ([-1.5, +1.5] x [-1.5, +1.5] x [0.0, 3.5])
    add_brush_aabb(map, [-1.5, -1.5, 0.0], [1.5, 1.5, 3.5], 2);

    // 3. West Wing Partition Wall (x in [-10.5, -9.5]) with central doorway y in [-2.0, +2.0]
    add_brush_aabb(map, [-10.5, -31.0, 0.0], [-9.5, -2.0, 4.0], 1);
    add_brush_aabb(map, [-10.5, 2.0, 0.0], [-9.5, 31.0, 4.0], 1);
    add_brush_aabb(map, [-10.5, -2.0, 2.6], [-9.5, 2.0, 4.0], 1);

    // 4. East Wing Partition Wall (x in [9.5, 10.5]) with central doorway y in [-2.0, +2.0]
    add_brush_aabb(map, [9.5, -31.0, 0.0], [10.5, -2.0, 4.0], 1);
    add_brush_aabb(map, [9.5, 2.0, 0.0], [10.5, 31.0, 4.0], 1);
    add_brush_aabb(map, [9.5, -2.0, 2.6], [10.5, 2.0, 4.0], 1);

    // 5. South Wing Wall (y in [-12.5, -11.5]) with central doorway x in [-2.0, +2.0]
    //    and outer-wing doorways at x in [-19.5, -16.5] and [16.5, 19.5]
    add_brush_aabb(map, [-9.5, -12.5, 0.0], [-2.0, -11.5, 4.0], 1);
    add_brush_aabb(map, [2.0, -12.5, 0.0], [9.5, -11.5, 4.0], 1);
    add_brush_aabb(map, [-2.0, -12.5, 2.6], [2.0, -11.5, 4.0], 1);
    add_brush_aabb(map, [-31.0, -12.5, 0.0], [-19.5, -11.5, 4.0], 1);
    add_brush_aabb(map, [-16.5, -12.5, 0.0], [-10.5, -11.5, 4.0], 1);
    add_brush_aabb(map, [10.5, -12.5, 0.0], [16.5, -11.5, 4.0], 1);
    add_brush_aabb(map, [19.5, -12.5, 0.0], [31.0, -11.5, 4.0], 1);

    // 6. North Wing Wall (y in [11.5, 12.5]) symmetric to South Wing
    add_brush_aabb(map, [-9.5, 11.5, 0.0], [-2.0, 12.5, 4.0], 1);
    add_brush_aabb(map, [2.0, 11.5, 0.0], [9.5, 12.5, 4.0], 1);
    add_brush_aabb(map, [-2.0, 11.5, 2.6], [2.0, 12.5, 4.0], 1);
    add_brush_aabb(map, [-31.0, 11.5, 0.0], [-19.5, 12.5, 4.0], 1);
    add_brush_aabb(map, [-16.5, 11.5, 0.0], [-10.5, 12.5, 4.0], 1);
    add_brush_aabb(map, [10.5, 11.5, 0.0], [16.5, 12.5, 4.0], 1);
    add_brush_aabb(map, [19.5, 11.5, 0.0], [31.0, 12.5, 4.0], 1);

    // 7. Tactical Portals (Doorways + Central Pillar Diffraction Corners)
    add_portal_aperture(map, [-10.0, 0.0, 1.3], [0.6, 2.0, 1.3], 0, 96);
    add_portal_aperture(map, [10.0, 0.0, 1.3], [0.6, 2.0, 1.3], 0, 96);
    add_portal_aperture(map, [0.0, -12.0, 1.3], [2.0, 0.6, 1.3], 1, 96);
    add_portal_aperture(map, [0.0, 12.0, 1.3], [2.0, 0.6, 1.3], 1, 96);
    add_portal_aperture(map, [-18.0, -12.0, 1.3], [1.5, 0.6, 1.3], 1, 112);
    add_portal_aperture(map, [18.0, -12.0, 1.3], [1.5, 0.6, 1.3], 1, 112);
    add_portal_aperture(map, [-18.0, 12.0, 1.3], [1.5, 0.6, 1.3], 1, 112);
    add_portal_aperture(map, [18.0, 12.0, 1.3], [1.5, 0.6, 1.3], 1, 112);
    // Pillar knife-edge diffraction corners
    add_portal_aperture(map, [-1.7, -1.7, 1.3], [0.3, 0.3, 1.3], 0, 64);
    add_portal_aperture(map, [-1.7, 1.7, 1.3], [0.3, 0.3, 1.3], 0, 64);
    add_portal_aperture(map, [1.7, -1.7, 1.3], [0.3, 0.3, 1.3], 0, 64);
    add_portal_aperture(map, [1.7, 1.7, 1.3], [0.3, 0.3, 1.3], 0, 64);

    rebuild_static_bvh(map);
    map.header.initialized = 1;
}

#[inline]
fn ray_aabb_slab(
    origin: &[f32; 3],
    inv_dir: &[f32; 3],
    bmin: &[f32; 3],
    bmax: &[f32; 3],
    max_t: f32,
) -> Option<(f32, usize, f32)> {
    let mut tmin = 0.0f32;
    let mut tmax = max_t;
    let mut hit_axis = 0usize;
    let mut hit_sign = -1.0f32;

    for ax in 0..3 {
        let inv = inv_dir[ax];
        let t1 = (bmin[ax] - origin[ax]) * inv;
        let t2 = (bmax[ax] - origin[ax]) * inv;
        let (t_near, t_far, sgn) = if t1 <= t2 {
            (t1, t2, -1.0f32)
        } else {
            (t2, t1, 1.0f32)
        };
        if t_near > tmin {
            tmin = t_near;
            hit_axis = ax;
            hit_sign = sgn;
        }
        if t_far < tmax {
            tmax = t_far;
        }
        if tmin > tmax {
            return None;
        }
    }
    Some((canon_f32(tmin), hit_axis, hit_sign))
}

#[inline]
fn safe_inv_dir(dir: &[f32; 3]) -> [f32; 3] {
    let inv_one = |d: f32| -> f32 {
        if abs_f32(d) < 1e-8f32 {
            if d.to_bits() & 0x8000_0000 != 0 {
                -1e8f32
            } else {
                1e8f32
            }
        } else {
            canon_f32(1.0f32 / d)
        }
    };
    [inv_one(dir[0]), inv_one(dir[1]), inv_one(dir[2])]
}

/// Raycasts against `StaticVoxelBvh` up to `max_dist`, returning `Some((t_hit, normal))` on hit.
pub fn raycast_hard_cover(
    map: &StaticVoxelBvh,
    origin: &[f32; 3],
    dir: &[f32; 3],
    max_dist: f32,
) -> Option<(f32, [f32; 3])> {
    if map.header.num_bvh_nodes == 0 || max_dist <= 0.0 {
        return None;
    }
    let inv_dir = safe_inv_dir(dir);
    let mut best_t = max_dist;
    let mut best_normal = [0.0f32; 3];
    let mut hit_any = false;

    let mut stack = [0u32; 64];
    let mut sp = 1usize;
    stack[0] = 0;

    while sp > 0 {
        sp -= 1;
        let nidx = stack[sp] as usize;
        let node = &map.bvh_nodes[nidx];
        if ray_aabb_slab(origin, &inv_dir, &node.min, &node.max, best_t).is_none() {
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
                if let Some((t_hit, ax, sgn)) =
                    ray_aabb_slab(origin, &inv_dir, &b.min, &b.max, best_t)
                {
                    if t_hit < best_t {
                        best_t = t_hit;
                        best_normal = [0.0f32; 3];
                        best_normal[ax] = sgn;
                        hit_any = true;
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

    if hit_any {
        Some((best_t, best_normal))
    } else {
        None
    }
}

/// Checks if the open segment `(a, b)` is completely clear of hard cover.
#[inline]
pub fn has_line_of_sight(
    map: &StaticVoxelBvh,
    a: &[f32; 3],
    b: &[f32; 3],
    force_scalar: bool,
) -> bool {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let d2 = (dx * dx + dy * dy) + dz * dz;
    if d2 <= 1e-6f32 {
        return true;
    }
    let dist = sqrt_f32_deterministic(d2, force_scalar);
    let inv = 1.0f32 / dist;
    let dir = [dx * inv, dy * inv, dz * inv];
    let max_t = (dist - 0.02f32).max(0.0f32);
    raycast_hard_cover(map, a, &dir, max_t).is_none()
}

/// Returns `true` iff ALL 5 canonical sample points of the target's capsule are blocked by
/// `StaticVoxelBvh` hard cover from `muzzle_xyz`.
pub fn is_capsule_100pct_occluded(
    map: &StaticVoxelBvh,
    muzzle_xyz: &[f32; 3],
    target_pos: &[f32; 3],
    radius_m: f32,
    height_m: f32,
    force_scalar: bool,
) -> bool {
    if map.header.num_bvh_nodes == 0 {
        return false;
    }
    // Transverse horizontal vector perpendicular to line from muzzle to target
    let vx = target_pos[0] - muzzle_xyz[0];
    let vy = target_pos[1] - muzzle_xyz[1];
    let horiz_d2 = vx * vx + vy * vy;
    let (px, py) = if horiz_d2 > 1e-6f32 {
        let inv_h = 1.0f32 / sqrt_f32_deterministic(horiz_d2, force_scalar);
        (-vy * inv_h * radius_m * 0.92f32, vx * inv_h * radius_m * 0.92f32)
    } else {
        (radius_m * 0.92f32, 0.0f32)
    };

    let samples: [[f32; 3]; 5] = [
        [target_pos[0], target_pos[1], target_pos[2] + height_m * 0.55f32], // chest center
        [target_pos[0], target_pos[1], target_pos[2] + height_m * 0.90f32], // head crown
        [target_pos[0], target_pos[1], target_pos[2] + height_m * 0.20f32], // pelvis
        [
            target_pos[0] + px,
            target_pos[1] + py,
            target_pos[2] + height_m * 0.65f32,
        ], // left shoulder
        [
            target_pos[0] - px,
            target_pos[1] - py,
            target_pos[2] + height_m * 0.65f32,
        ], // right shoulder
    ];

    for sample in &samples {
        if has_line_of_sight(map, muzzle_xyz, sample, force_scalar) {
            return false;
        }
    }
    true
}

/// Minkowski Reachability PVS Visibility Predicate:
/// Returns `true` iff the closed reachability ball `B(target_pos, reach_radius_m + capsule_radius)`
/// intersects the observer's unoccluded visibility / portal aperture region from `obs_eye`.
pub fn is_reachability_ball_visible(
    map: &StaticVoxelBvh,
    obs_eye: &[f32; 3],
    target_pos: &[f32; 3],
    reach_radius_m: f32,
    force_scalar: bool,
) -> bool {
    if map.header.num_bvh_nodes == 0 {
        return true;
    }

    let r_tot = (reach_radius_m + 0.36f32).max(0.36f32);

    // 1. Direct center rays (chest & head)
    let chest = [target_pos[0], target_pos[1], target_pos[2] + 1.05f32];
    if has_line_of_sight(map, obs_eye, &chest, force_scalar) {
        return true;
    }
    let head = [target_pos[0], target_pos[1], target_pos[2] + 1.62f32];
    if has_line_of_sight(map, obs_eye, &head, force_scalar) {
        return true;
    }

    // 2. 8-point Minkowski boundary disk at chest & head heights (catches corner/pillar peeks before swing!)
    const DIAG: f32 = 0.707_106_77f32;
    const OFFSETS_8: [[f32; 2]; 8] = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [DIAG, DIAG],
        [DIAG, -DIAG],
        [-DIAG, DIAG],
        [-DIAG, -DIAG],
    ];
    for off in &OFFSETS_8 {
        let bx = target_pos[0] + off[0] * r_tot;
        let by = target_pos[1] + off[1] * r_tot;
        let p_chest = [bx, by, target_pos[2] + 1.15f32];
        if has_line_of_sight(map, obs_eye, &p_chest, force_scalar) {
            return true;
        }
    }

    // 3. Doorway Portal Aperture Pre-Delivery Check:
    // If any visible portal aperture lies within `r_tot` of `target_pos` (and unoccluded to target),
    // pre-deliver so doorway swings have strictly zero pop-in frames!
    let n_portals = map.header.num_portals as usize;
    let r_tot_sq = r_tot * r_tot;
    for p_idx in 0..n_portals {
        let portal = &map.portals[p_idx];
        if portal.active == 0 {
            continue;
        }
        let cx = target_pos[0].clamp(
            portal.center[0] - portal.half_extents[0],
            portal.center[0] + portal.half_extents[0],
        );
        let cy = target_pos[1].clamp(
            portal.center[1] - portal.half_extents[1],
            portal.center[1] + portal.half_extents[1],
        );
        let cz = (target_pos[2] + 1.25f32).clamp(
            portal.center[2] - portal.half_extents[2],
            portal.center[2] + portal.half_extents[2],
        );
        let dx = cx - target_pos[0];
        let dy = cy - target_pos[1];
        if (dx * dx + dy * dy) <= r_tot_sq {
            let portal_pt = [cx, cy, cz];
            if has_line_of_sight(map, obs_eye, &portal_pt, force_scalar)
                && has_line_of_sight(map, &portal_pt, &chest, force_scalar)
            {
                return true;
            }
        }
    }

    false
}

/// Computes anonymous portal-diffracted 4-band acoustic arrival at `obs_eye` (with observer `obs_yaw_q15`).
/// Returns `Some((azimuth_q12, elevation_q12, band_energy_q8))` if audible, or `None` if beyond horizon / sealed.
/// Direction of occluded sound points strictly from `obs_eye` to the diffracting portal opening (zero xyz leak!).
pub fn compute_anonymous_acoustic_arrival(
    map: &StaticVoxelBvh,
    obs_eye: &[f32; 3],
    obs_yaw_q15: i16,
    src_pos: &[f32; 3],
    max_radius_m: f32,
    raw_bands: &[u8; 4],
    force_scalar: bool,
) -> Option<(i16, i16, [u8; 4])> {
    if raw_bands[0] == 0 && raw_bands[1] == 0 && raw_bands[2] == 0 && raw_bands[3] == 0 {
        return None;
    }

    let dx_direct = src_pos[0] - obs_eye[0];
    let dy_direct = src_pos[1] - obs_eye[1];
    let dz_direct = (src_pos[2] + 0.9f32) - obs_eye[2];
    let dist_sq_direct = (dx_direct * dx_direct + dy_direct * dy_direct) + dz_direct * dz_direct;
    if dist_sq_direct > max_radius_m * max_radius_m {
        return None;
    }

    let src_elev = [src_pos[0], src_pos[1], src_pos[2] + 0.9f32];
    let (apparent_dir, total_dist, occluded_loss_q8) =
        if has_line_of_sight(map, obs_eye, &src_elev, force_scalar) {
            let d = sqrt_f32_deterministic(dist_sq_direct, force_scalar).max(0.25f32);
            ([dx_direct, dy_direct, dz_direct], d, 0u16)
        } else {
            // Find shortest two-hop portal diffraction path: src_elev -> portal_k -> obs_eye
            let n_portals = map.header.num_portals as usize;
            let mut best_path_dist = max_radius_m + 1.0f32;
            let mut best_portal_vec = [0.0f32; 3];
            let mut best_loss = 0u16;
            let mut found = false;

            for k in 0..n_portals {
                let p = &map.portals[k];
                if p.active == 0 {
                    continue;
                }
                if !has_line_of_sight(map, obs_eye, &p.center, force_scalar) {
                    continue;
                }
                if !has_line_of_sight(map, &p.center, &src_elev, force_scalar) {
                    continue;
                }
                let d1x = p.center[0] - src_elev[0];
                let d1y = p.center[1] - src_elev[1];
                let d1z = p.center[2] - src_elev[2];
                let d1 = sqrt_f32_deterministic((d1x * d1x + d1y * d1y) + d1z * d1z, force_scalar);

                let d2x = p.center[0] - obs_eye[0];
                let d2y = p.center[1] - obs_eye[1];
                let d2z = p.center[2] - obs_eye[2];
                let d2 = sqrt_f32_deterministic((d2x * d2x + d2y * d2y) + d2z * d2z, force_scalar);

                let tot = d1 + d2;
                if tot <= max_radius_m && tot < best_path_dist {
                    best_path_dist = tot;
                    best_portal_vec = [d2x, d2y, d2z];
                    best_loss = p.diffraction_loss_q8;
                    found = true;
                }
            }

            if !found {
                return None;
            }
            (best_portal_vec, best_path_dist.max(0.5f32), best_loss)
        };

    // Distance inverse-square rolloff + frequency-dependent knife-edge portal diffraction low-pass
    let dist_atten = 1.0f32 / (1.0f32 + 0.045f32 * total_dist * total_dist);
    let diff_frac = (occluded_loss_q8 as f32) * (1.0f32 / 256.0f32);
    // Low frequencies (125 Hz) diffract strongly around portals; 8 kHz attenuates sharply
    let band_pass: [f32; 4] = [
        (1.0f32 - 0.25f32 * diff_frac).clamp(0.0f32, 1.0f32),
        (1.0f32 - 0.50f32 * diff_frac).clamp(0.0f32, 1.0f32),
        (1.0f32 - 0.78f32 * diff_frac).clamp(0.0f32, 1.0f32),
        (1.0f32 - 0.92f32 * diff_frac).clamp(0.0f32, 1.0f32),
    ];

    let mut out_bands = [0u8; 4];
    let mut non_zero = false;
    for b in 0..4 {
        let val = ((raw_bands[b] as f32) * dist_atten * band_pass[b] + 0.5f32) as i32;
        let clamped = val.clamp(0, 255) as u8;
        out_bands[b] = clamped;
        if clamped > 0 {
            non_zero = true;
        }
    }
    if !non_zero {
        return None;
    }

    // Relative azimuth & elevation in observer's head frame (quantized to Q3.12: 4096 = 1.0 rad)
    let world_az = atan2_f32_deterministic(apparent_dir[1], apparent_dir[0]);
    let obs_yaw_rad = (obs_yaw_q15 as f32) * (core::f32::consts::PI / 32768.0f32);
    let mut rel_az = world_az - obs_yaw_rad;
    if rel_az > core::f32::consts::PI {
        rel_az -= core::f32::consts::TAU;
    } else if rel_az < -core::f32::consts::PI {
        rel_az += core::f32::consts::TAU;
    }

    let horiz = sqrt_f32_deterministic(
        apparent_dir[0] * apparent_dir[0] + apparent_dir[1] * apparent_dir[1],
        force_scalar,
    );
    let elev = atan2_f32_deterministic(apparent_dir[2], horiz.max(1e-4f32));

    let az_q12 = ((rel_az * 4096.0f32) as i32).clamp(-32768, 32767) as i16;
    let el_q12 = ((elev * 4096.0f32) as i32).clamp(-32768, 32767) as i16;

    Some((az_q12, el_q12, out_bands))
}
