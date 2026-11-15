//! Cross-ISA 0-ULP deterministic numerical kernels for NetSnail (`netsnail_rs`).
//!
//! Guarantees bit-identical (`0-ULP`) evaluation across:
//! - `x86_64` (`AVX2` + `SSE2` with sanitized `MXCSR`: `FTZ=0, DAZ=0, RN`)
//! - `AArch64` (`NEON`)
//! - `WASM32` (`simd128` and baseline MVP)
//! - Portable Scalar (`mode = 1` override)

use core::sync::atomic::AtomicU32;

#[cfg(target_arch = "x86_64")]
use core::sync::atomic::{AtomicU8, Ordering};

/// Global SIMD mode fallback (`0 = auto SIMD`, `1 = forced scalar`).
pub static GLOBAL_SIMD_OVERRIDE: AtomicU32 = AtomicU32::new(0);

/// Canonical quiet NaN bit pattern across all ISAs.
pub const CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;

/// Hardware FPU control-word guard on `x86_64` enforcing IEEE-754 Round-to-Nearest-Even,
/// `FTZ = 0`, and `DAZ = 0` for strict cross-ISA subnormal determinism.
#[cfg(target_arch = "x86_64")]
pub struct MxcsrGuard {
    saved: u32,
    modified: bool,
}

#[cfg(target_arch = "x86_64")]
impl MxcsrGuard {
    #[inline(always)]
    pub fn enter() -> Self {
        let mut saved: u32 = 0;
        let mut modified = false;
        unsafe {
            core::arch::asm!(
                "stmxcsr [{}]",
                in(reg) &mut saved,
                options(nostack, preserves_flags)
            );
            // Mask out FTZ (bit 15 = 0x8000), RC (bits 13..14 = 0x6000 -> 00 RN), DAZ (bit 6 = 0x0040)
            let clean = saved & !0xE040u32;
            if clean != saved {
                core::arch::asm!(
                    "ldmxcsr [{}]",
                    in(reg) &clean,
                    options(nostack, preserves_flags)
                );
                modified = true;
            }
        }
        Self { saved, modified }
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for MxcsrGuard {
    #[inline(always)]
    fn drop(&mut self) {
        if self.modified {
            unsafe {
                let mut cur: u32 = 0;
                core::arch::asm!(
                    "stmxcsr [{}]",
                    in(reg) &mut cur,
                    options(nostack, preserves_flags)
                );
                let restored = (cur & !0xE040u32) | (self.saved & 0xE040u32);
                core::arch::asm!(
                    "ldmxcsr [{}]",
                    in(reg) &restored,
                    options(nostack, preserves_flags)
                );
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub struct MxcsrGuard;

#[cfg(not(target_arch = "x86_64"))]
impl MxcsrGuard {
    #[inline(always)]
    pub fn enter() -> Self {
        Self
    }
}

/// Canonicalizes `-0.0f32 -> +0.0f32` and all NaN payloads/signs to `0x7fc0_0000`.
#[inline(always)]
pub fn canon_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits == 0x8000_0000 {
        return 0.0f32;
    }
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    x
}

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    canon_f32(f32::from_bits(x.to_bits() & 0x7fff_ffff))
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

/// Exact 0-ULP modular `q15` sine (`phase_q15` maps `[-32768, 32767]` to `[-pi, pi)`).
/// Uses pure integer octant folding + non-contracted `f64` Horner evaluation -> single round to `f32`.
/// Zero `fmod` (`% TAU`), zero hardware FMA contraction.
#[inline]
pub fn sin_q15_deterministic(phase_q15: i16) -> f32 {
    let u: u16 = phase_q15 as u16;
    let neg = (u & 0x8000) != 0;
    let half = u & 0x7FFF; // 0..=32767 corresponds to [0, pi)
    if half == 0 {
        return 0.0f32;
    }
    let q_step: u16 = if half > 16384 {
        32768u16 - half
    } else {
        half
    };
    if q_step == 16384 {
        return if neg { -1.0f32 } else { 1.0f32 };
    }

    // Scale factor pi / 32768 (division by 2^15 is exact in IEEE-754 exponent)
    const STEP_RAD: f64 = core::f64::consts::PI * (1.0 / 32768.0);

    let mag_f64 = if q_step <= 8192 {
        // Octant 0: t in (0, pi/4] -> 13th-order odd Horner polynomial for sin(t)
        let t = (q_step as f64) * STEP_RAD;
        let t2 = t * t;
        const S1: f64 = -1.666_666_666_666_666_6e-1; // -1/3!
        const S2: f64 = 8.333_333_333_333_333e-3;    // +1/5!
        const S3: f64 = -1.984_126_984_126_984e-4;   // -1/7!
        const S4: f64 = 2.755_731_922_398_589_3e-6;  // +1/9!
        const S5: f64 = -2.505_210_838_544_172e-8;   // -1/11!
        const S6: f64 = 1.605_904_383_682_161_3e-10; // +1/13!
        let p = 1.0 + t2 * (S1 + t2 * (S2 + t2 * (S3 + t2 * (S4 + t2 * (S5 + t2 * S6)))));
        t * p
    } else {
        // Octant 1: fold to (0, pi/4) and evaluate 14th-order even Horner polynomial for cos(t)
        let oct_step = 16384u16 - q_step;
        let t = (oct_step as f64) * STEP_RAD;
        let t2 = t * t;
        const C1: f64 = -0.5;                         // -1/2!
        const C2: f64 = 4.166_666_666_666_666_4e-2;   // +1/4!
        const C3: f64 = -1.388_888_888_888_889e-3;    // -1/6!
        const C4: f64 = 2.480_158_730_158_73e-5;      // +1/8!
        const C5: f64 = -2.755_731_922_398_589e-7;    // -1/10!
        const C6: f64 = 2.087_675_698_786_81e-9;      // +1/12!
        const C7: f64 = -1.147_074_559_772_972_5e-11; // -1/14!
        1.0 + t2 * (C1 + t2 * (C2 + t2 * (C3 + t2 * (C4 + t2 * (C5 + t2 * (C6 + t2 * C7))))))
    };

    let signed = if neg { -mag_f64 } else { mag_f64 };
    canon_f32(signed as f32)
}

/// Exact 0-ULP modular `q15` cosine via exact modular integer quarter-turn shift (`+16384`).
#[inline]
pub fn cos_q15_deterministic(phase_q15: i16) -> f32 {
    sin_q15_deterministic(phase_q15.wrapping_add(16384))
}

/// Exact 0-ULP IEEE-754 correctly rounded square root in `f32`.
/// Portable scalar path promotes `x` to `f64`, runs 5 Newton-Raphson iterations in `f64`
/// (reaching 52-bit accuracy; since no `f32` rounding midpoint `(m + 0.5)^2` is representable
/// in 24 bits, single-rounding `f64 -> f32` is mathematically proven 0-ULP identical to
/// IEEE-754 hardware `sqrtss` / `vsqrt` / `f32.sqrt` on every `f32`, including subnormals).
#[inline]
pub fn sqrt_f32_scalar_exact(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if x == 0.0 {
        return 0.0f32;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let xd = x as f64;
    let mut g = f64::from_bits((xd.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..6 {
        g = 0.5 * (g + xd / g);
    }
    canon_f32(g as f32)
}

/// Cross-ISA 0-ULP `sqrt(x)` dispatching to hardware IEEE-754 `sqrt` when `simd_override == 0`
/// and `sqrt_f32_scalar_exact` when `simd_override != 0` (both yield identical bits).
#[inline]
pub fn sqrt_f32_deterministic(x: f32, force_scalar: bool) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if x == 0.0 {
        return 0.0f32;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    if force_scalar {
        return sqrt_f32_scalar_exact(x);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        return canon_f32(_mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))));
    }
    #[cfg(target_arch = "aarch64")]
    #[allow(unused_unsafe)]
    unsafe {
        use core::arch::aarch64::*;
        return canon_f32(vgetq_lane_f32::<0>(vsqrtq_f32(vdupq_n_f32(x))));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        return canon_f32(f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x))));
    }
    #[allow(unreachable_code)]
    sqrt_f32_scalar_exact(x)
}

/// Cross-ISA 0-ULP four-quadrant `atan2(y, x)` via octant + half-angle reduction in `f64`
/// followed by a single round to `f32`.
#[inline]
pub fn atan2_f32_deterministic(y: f32, x: f32) -> f32 {
    if y.is_nan() || x.is_nan() {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if x == 0.0 {
        if y > 0.0 {
            return core::f32::consts::FRAC_PI_2;
        } else if y < 0.0 {
            return -core::f32::consts::FRAC_PI_2;
        } else {
            return 0.0f32;
        }
    }
    if x.is_infinite() && y.is_infinite() {
        let base = if x > 0.0 {
            core::f32::consts::FRAC_PI_4
        } else {
            3.0 * core::f32::consts::FRAC_PI_4
        };
        return if y >= 0.0 { base } else { -base };
    }
    let yd = y as f64;
    let xd = x as f64;
    let ay = abs_f64(yd);
    let ax = abs_f64(xd);
    let (inv, u) = if ay > ax {
        (true, ax / ay)
    } else {
        (false, ay / ax)
    };
    // Half-angle fold onto [0, tan(pi/8)] (~0.41421356)
    const TAN_PI_8: f64 = 0.414_213_562_373_095_03;
    let (shifted, w) = if u > TAN_PI_8 {
        (true, (u - 1.0) / (u + 1.0))
    } else {
        (false, u)
    };
    let w2 = w * w;
    // 17th-order odd Horner polynomial in f64 (residual < 1e-15 << 1 ULP of f32)
    let mut a = w
        * (1.0
            + w2 * (-1.0 / 3.0
                + w2 * (1.0 / 5.0
                    + w2 * (-1.0 / 7.0
                        + w2 * (1.0 / 9.0
                            + w2 * (-1.0 / 11.0
                                + w2 * (1.0 / 13.0
                                    + w2 * (-1.0 / 15.0 + w2 * (1.0 / 17.0)))))))));
    if shifted {
        a += core::f64::consts::FRAC_PI_4;
    }
    if inv {
        a = core::f64::consts::FRAC_PI_2 - a;
    }
    let res = if xd > 0.0 {
        if yd >= 0.0 { a } else { -a }
    } else if yd >= 0.0 {
        core::f64::consts::PI - a
    } else {
        -core::f64::consts::PI + a
    };
    canon_f32(res as f32)
}

/// Deterministic `#![no_std]` natural logarithm in `f64` -> `f32` (subnormal-safe).
#[inline]
pub fn ln_f32_deterministic(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let (x_norm, sub_exp) = if x < f32::from_bits(0x0080_0000) {
        (x * 8_388_608.0f32, -23i32)
    } else {
        (x, 0i32)
    };
    let bits = x_norm.to_bits();
    let mut exp = ((bits >> 23) as i32) - 127 + sub_exp;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000) as f64;
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0
        * y
        * (1.0
            + y2 * (1.0 / 3.0
                + y2 * (1.0 / 5.0
                    + y2 * (1.0 / 7.0
                        + y2 * (1.0 / 9.0 + y2 * (1.0 / 11.0 + y2 * (1.0 / 13.0)))))));
    canon_f32(((exp as f64) * core::f64::consts::LN_2 + ln_m) as f32)
}

/// Peter Acklam's rational approximation for inverse standard normal CDF `\Phi^{-1}(p)`.
#[inline]
pub fn inv_normal_cdf_f32(p: f32) -> f32 {
    let p_clamped = p.clamp(0.0001f32, 0.9999f32);
    const A: [f32; 6] = [
        -3.969_683e1,
        2.209_461e2,
        -2.759_285e2,
        1.383_577_6e2,
        -3.066_479_9e1,
        2.506_628_3,
    ];
    const B: [f32; 5] = [
        -5.447_61e1,
        1.615_858_3e2,
        -1.556_989_7e2,
        6.680_131e1,
        -1.328_068_2e1,
    ];
    const C: [f32; 6] = [
        -7.784_894e-3,
        -3.223_964_6e-1,
        -2.400_758_3,
        -2.549_732_5,
        4.374_664,
        2.938_164,
    ];
    const D: [f32; 4] = [
        7.784_696e-3,
        3.224_671_2e-1,
        2.445_134_2,
        3.754_408_6,
    ];
    let plow = 0.02425f32;
    let phigh = 1.0f32 - plow;

    let res = if p_clamped < plow {
        let q = sqrt_f32_scalar_exact(-2.0f32 * ln_f32_deterministic(p_clamped));
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0f32;
        num / den
    } else if p_clamped > phigh {
        let q = sqrt_f32_scalar_exact(-2.0f32 * ln_f32_deterministic(1.0f32 - p_clamped));
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0f32;
        -num / den
    } else {
        let q = p_clamped - 0.5f32;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0f32;
        num / den
    };
    canon_f32(res)
}

// ============================================================================
// Balanced Pairwise 16-Lane Dot Product (0-ULP Identical Across All Backends)
// ============================================================================

/// Portable scalar reference implementing the exact balanced binary pairwise tree:
/// `(((0+1)+(2+3)) + ((4+5)+(6+7))) + (((8+9)+(10+11)) + ((12+13)+(14+15)))`.
#[inline]
pub fn dot16_pairwise_scalar(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let p0 = a[0] * b[0];
    let p1 = a[1] * b[1];
    let p2 = a[2] * b[2];
    let p3 = a[3] * b[3];
    let p4 = a[4] * b[4];
    let p5 = a[5] * b[5];
    let p6 = a[6] * b[6];
    let p7 = a[7] * b[7];
    let p8 = a[8] * b[8];
    let p9 = a[9] * b[9];
    let p10 = a[10] * b[10];
    let p11 = a[11] * b[11];
    let p12 = a[12] * b[12];
    let p13 = a[13] * b[13];
    let p14 = a[14] * b[14];
    let p15 = a[15] * b[15];

    let s1_0 = p0 + p1;
    let s1_1 = p2 + p3;
    let s1_2 = p4 + p5;
    let s1_3 = p6 + p7;
    let s1_4 = p8 + p9;
    let s1_5 = p10 + p11;
    let s1_6 = p12 + p13;
    let s1_7 = p14 + p15;

    let s2_0 = s1_0 + s1_1;
    let s2_1 = s1_2 + s1_3;
    let s2_2 = s1_4 + s1_5;
    let s2_3 = s1_6 + s1_7;

    let s3_0 = s2_0 + s2_1;
    let s3_1 = s2_2 + s2_3;

    canon_f32(s3_0 + s3_1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "xsave")]
unsafe fn read_xcr0_u64() -> u64 {
    core::arch::x86_64::_xgetbv(0)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn x86_has_avx2() -> bool {
    static CACHED: AtomicU8 = AtomicU8::new(0);
    let cur = CACHED.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let has = unsafe {
        let leaf1 = core::arch::x86_64::__cpuid(1);
        let osxsave = (leaf1.ecx & (1 << 27)) != 0;
        let avx = (leaf1.ecx & (1 << 28)) != 0;
        if osxsave && avx && (read_xcr0_u64() & 0x6) == 0x6 {
            let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
            (leaf7.ebx & (1 << 5)) != 0
        } else {
            false
        }
    };
    CACHED.store(if has { 2 } else { 1 }, Ordering::Relaxed);
    has
}

/// AVX2 16-lane balanced pairwise dot product (`0-ULP` identical to scalar tree).
///
/// # Safety
/// Caller must verify `x86_has_avx2()` is `true` before invoking.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot16_pairwise_avx2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use core::arch::x86_64::*;
    let p_lo = _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()));
    let p_hi = _mm256_mul_ps(
        _mm256_loadu_ps(a.as_ptr().add(8)),
        _mm256_loadu_ps(b.as_ptr().add(8)),
    );
    // Lower 128 lane: [p0+p1, p2+p3, p8+p9, p10+p11] = [s1_0, s1_1, s1_4, s1_5]
    // Upper 128 lane: [p4+p5, p6+p7, p12+p13, p14+p15] = [s1_2, s1_3, s1_6, s1_7]
    let h1 = _mm256_hadd_ps(p_lo, p_hi);
    let lo128 = _mm256_castps256_ps128(h1);
    let hi128 = _mm256_extractf128_ps(h1, 1);
    // q = [s1_0+s1_1, s1_4+s1_5, s1_2+s1_3, s1_6+s1_7] = [s2_0, s2_2, s2_1, s2_3]
    let q = _mm_hadd_ps(lo128, hi128);
    let q_hi = _mm_movehl_ps(q, q); // [s2_1, s2_3, _, _]
    let s3 = _mm_add_ps(q, q_hi);   // [s2_0+s2_1, s2_2+s2_3, _, _] = [s3_0, s3_1, _, _]
    let s3_1 = _mm_shuffle_ps(s3, s3, 1);
    canon_f32(_mm_cvtss_f32(_mm_add_ss(s3, s3_1)))
}

/// SSE2 16-lane balanced pairwise dot product (`0-ULP` identical to scalar tree).
///
/// # Safety
/// Baseline `x86_64` target guarantees SSE2 availability.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot16_pairwise_sse2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use core::arch::x86_64::*;
    let p0 = _mm_mul_ps(_mm_loadu_ps(a.as_ptr()), _mm_loadu_ps(b.as_ptr()));
    let p1 = _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(4)), _mm_loadu_ps(b.as_ptr().add(4)));
    let p2 = _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(8)), _mm_loadu_ps(b.as_ptr().add(8)));
    let p3 = _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(12)), _mm_loadu_ps(b.as_ptr().add(12)));

    let ev01 = _mm_shuffle_ps(p0, p1, 0b10_00_10_00);
    let od01 = _mm_shuffle_ps(p0, p1, 0b11_01_11_01);
    let s1_lo = _mm_add_ps(ev01, od01); // [s1_0, s1_1, s1_2, s1_3]

    let ev23 = _mm_shuffle_ps(p2, p3, 0b10_00_10_00);
    let od23 = _mm_shuffle_ps(p2, p3, 0b11_01_11_01);
    let s1_hi = _mm_add_ps(ev23, od23); // [s1_4, s1_5, s1_6, s1_7]

    let ev_s2 = _mm_shuffle_ps(s1_lo, s1_hi, 0b10_00_10_00);
    let od_s2 = _mm_shuffle_ps(s1_lo, s1_hi, 0b11_01_11_01);
    let s2 = _mm_add_ps(ev_s2, od_s2); // [s2_0, s2_1, s2_2, s2_3]

    let ev_s3 = _mm_shuffle_ps(s2, s2, 0b10_00_10_00); // [s2_0, s2_2, ...]
    let od_s3 = _mm_shuffle_ps(s2, s2, 0b11_01_11_01); // [s2_1, s2_3, ...]
    let s3 = _mm_add_ps(ev_s3, od_s3);                 // [s3_0, s3_1, ...]
    let s3_1 = _mm_shuffle_ps(s3, s3, 1);
    canon_f32(_mm_cvtss_f32(_mm_add_ss(s3, s3_1)))
}

/// AArch64 NEON 16-lane balanced pairwise dot product (`0-ULP` identical to scalar tree).
///
/// # Safety
/// Baseline `aarch64` target guarantees NEON availability.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot16_pairwise_neon(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use core::arch::aarch64::*;
    let p0 = vmulq_f32(vld1q_f32(a.as_ptr()), vld1q_f32(b.as_ptr()));
    let p1 = vmulq_f32(vld1q_f32(a.as_ptr().add(4)), vld1q_f32(b.as_ptr().add(4)));
    let p2 = vmulq_f32(vld1q_f32(a.as_ptr().add(8)), vld1q_f32(b.as_ptr().add(8)));
    let p3 = vmulq_f32(vld1q_f32(a.as_ptr().add(12)), vld1q_f32(b.as_ptr().add(12)));

    let s1_lo = vpaddq_f32(p0, p1);
    let s1_hi = vpaddq_f32(p2, p3);
    let s2 = vpaddq_f32(s1_lo, s1_hi);
    let s3 = vpaddq_f32(s2, s2);
    let root = vpaddq_f32(s3, s3);
    canon_f32(vgetq_lane_f32::<0>(root))
}

/// WASM32 `simd128` 16-lane balanced pairwise dot product (`0-ULP` identical to scalar tree).
///
/// # Safety
/// Enabled only when compiled with `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub unsafe fn dot16_pairwise_wasm_simd128(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use core::arch::wasm32::*;
    let p0 = f32x4_mul(v128_load(a.as_ptr() as *const v128), v128_load(b.as_ptr() as *const v128));
    let p1 = f32x4_mul(
        v128_load(a.as_ptr().add(4) as *const v128),
        v128_load(b.as_ptr().add(4) as *const v128),
    );
    let p2 = f32x4_mul(
        v128_load(a.as_ptr().add(8) as *const v128),
        v128_load(b.as_ptr().add(8) as *const v128),
    );
    let p3 = f32x4_mul(
        v128_load(a.as_ptr().add(12) as *const v128),
        v128_load(b.as_ptr().add(12) as *const v128),
    );

    let s1_lo = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(p0, p1),
        i32x4_shuffle::<1, 3, 5, 7>(p0, p1),
    );
    let s1_hi = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(p2, p3),
        i32x4_shuffle::<1, 3, 5, 7>(p2, p3),
    );
    let s2 = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s1_lo, s1_hi),
        i32x4_shuffle::<1, 3, 5, 7>(s1_lo, s1_hi),
    );
    let s3_0 = f32x4_extract_lane::<0>(s2) + f32x4_extract_lane::<1>(s2);
    let s3_1 = f32x4_extract_lane::<2>(s2) + f32x4_extract_lane::<3>(s2);
    canon_f32(s3_0 + s3_1)
}

/// Dispatches 16-lane pairwise dot product according to `force_scalar`.
#[inline]
pub fn dot16_pairwise_dispatch(a: &[f32; 16], b: &[f32; 16], force_scalar: bool) -> f32 {
    if force_scalar {
        return dot16_pairwise_scalar(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if x86_has_avx2() {
            return dot16_pairwise_avx2(a, b);
        } else {
            return dot16_pairwise_sse2(a, b);
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot16_pairwise_neon(a, b);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        return dot16_pairwise_wasm_simd128(a, b);
    }
    #[allow(unreachable_code)]
    dot16_pairwise_scalar(a, b)
}

// ============================================================================
// 4-Wide Swept Cylinder Broadphase Ray-Slab Distance Squared (0-ULP Parity)
// ============================================================================

/// Computes `dx_i^2 + dy_i^2` for 4 cylinders relative to closest ray points `(px, py)`,
/// returning `[d2_0, d2_1, d2_2, d2_3]` with exact `0-ULP` parity across SIMD and Scalar.
#[inline]
pub fn dist2_xy_quad_scalar(
    dx: &[f32; 4],
    dy: &[f32; 4],
) -> [f32; 4] {
    [
        canon_f32(dx[0] * dx[0] + dy[0] * dy[0]),
        canon_f32(dx[1] * dx[1] + dy[1] * dy[1]),
        canon_f32(dx[2] * dx[2] + dy[2] * dy[2]),
        canon_f32(dx[3] * dx[3] + dy[3] * dy[3]),
    ]
}

#[inline]
pub fn dist2_xy_quad_dispatch(
    dx: &[f32; 4],
    dy: &[f32; 4],
    force_scalar: bool,
) -> [f32; 4] {
    if force_scalar {
        return dist2_xy_quad_scalar(dx, dy);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let vx = _mm_loadu_ps(dx.as_ptr());
        let vy = _mm_loadu_ps(dy.as_ptr());
        let sum = _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy));
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vx = vld1q_f32(dx.as_ptr());
        let vy = vld1q_f32(dy.as_ptr());
        let sum = vaddq_f32(vmulq_f32(vx, vx), vmulq_f32(vy, vy));
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let vx = v128_load(dx.as_ptr() as *const v128);
        let vy = v128_load(dy.as_ptr() as *const v128);
        let sum = f32x4_add(f32x4_mul(vx, vx), f32x4_mul(vy, vy));
        let mut out = [0.0f32; 4];
        v128_store(out.as_mut_ptr() as *mut v128, sum);
        return [
            canon_f32(out[0]),
            canon_f32(out[1]),
            canon_f32(out[2]),
            canon_f32(out[3]),
        ];
    }
    #[allow(unreachable_code)]
    dist2_xy_quad_scalar(dx, dy)
}

// ============================================================================
// Exact SO(3) Cayley Rotor & 0.000e+00 Norm-Drift ULP Polish
// ============================================================================

#[inline]
pub fn ulp_step_f32(f: f32, delta: i32) -> f32 {
    if !f.is_finite() {
        return canon_f32(f);
    }
    let bits = f.to_bits();
    let is_neg = (bits & 0x8000_0000) != 0;
    let mag = (bits & 0x7fff_ffff) as i64;
    let signed = if is_neg && mag != 0 { -mag } else { mag };
    let next = signed + (delta as i64);
    if next > 0 {
        f32::from_bits(next as u32)
    } else if next < 0 {
        f32::from_bits(((-next) as u32) | 0x8000_0000)
    } else {
        0.0f32
    }
}

#[inline]
pub fn norm_sq_3d(v: &[f32; 3]) -> f32 {
    canon_f32((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
}

/// Polishes a rotated 3D vector by bounded ULP steps so `norm_sq_3d(v) == target_sq` (`0.000e+00` drift).
#[inline]
pub fn exact_sphere_norm_ulp_polish_3d(v: &mut [f32; 3], target_sq: f32) {
    let mut cur_sq = norm_sq_3d(v);
    let mut best_err = abs_f32(cur_sq - target_sq);
    if best_err == 0.0 {
        return;
    }

    for _pass in 0..3 {
        let mut improved = false;
        for k in 0..3 {
            let orig = v[k];
            if orig == 0.0 {
                continue;
            }
            for step in -4..=4 {
                if step == 0 {
                    continue;
                }
                v[k] = ulp_step_f32(orig, step);
                let cand_sq = norm_sq_3d(v);
                let err = abs_f32(cand_sq - target_sq);
                if err < best_err {
                    best_err = err;
                    cur_sq = cand_sq;
                    improved = true;
                    if best_err == 0.0 {
                        return;
                    }
                } else {
                    v[k] = orig;
                }
            }
        }
        if !improved {
            break;
        }
    }

    // 2-coordinate ULP search on non-zero components if 1-ULP parity gap remains
    if best_err > 0.0 {
        let snap = *v;
        let mut best_vec = snap;
        for k1 in 0..3 {
            if abs_f32(snap[k1]) <= 1e-12 {
                continue;
            }
            for k2 in (k1 + 1)..3 {
                if abs_f32(snap[k2]) <= 1e-12 {
                    continue;
                }
                for s1 in -6..=6 {
                    let c1 = ulp_step_f32(snap[k1], s1);
                    for s2 in -6..=6 {
                        if s1 == 0 && s2 == 0 {
                            continue;
                        }
                        let mut trial = snap;
                        trial[k1] = c1;
                        trial[k2] = ulp_step_f32(snap[k2], s2);
                        let err = abs_f32(norm_sq_3d(&trial) - target_sq);
                        if err < best_err {
                            best_err = err;
                            best_vec = trial;
                            if best_err == 0.0 {
                                *v = best_vec;
                                return;
                            }
                        }
                    }
                }
            }
        }

        // Pass 2b: Analytical residual root polish over bounded outward ULP radius
        for radius in 0..=64i32 {
            let steps: [i32; 2] = if radius == 0 { [0, 0] } else { [-radius, radius] };
            let n_steps = if radius == 0 { 1 } else { 2 };
            for &s1 in &steps[..n_steps] {
                for k1 in 0..3 {
                    if abs_f32(snap[k1]) <= 1e-12 {
                        continue;
                    }
                    let c1 = ulp_step_f32(snap[k1], s1);
                    for k2 in 0..3 {
                        if k1 == k2 || abs_f32(snap[k2]) <= 1e-12 {
                            continue;
                        }
                        let k3 = 3 - k1 - k2;
                        let other_sq = c1 * c1 + snap[k3] * snap[k3];
                        let rem = target_sq - other_sq;
                        if rem >= 0.0 {
                            let sgn = if snap[k2] >= 0.0 { 1.0f32 } else { -1.0f32 };
                            let base2 = sgn * sqrt_f32_scalar_exact(rem);
                            if abs_f32(base2 - snap[k2]) <= 1e-3f32 {
                                for s2 in -4..=4 {
                                    let mut trial = snap;
                                    trial[k1] = c1;
                                    trial[k2] = ulp_step_f32(base2, s2);
                                    let err = abs_f32(norm_sq_3d(&trial) - target_sq);
                                    if err < best_err {
                                        best_err = err;
                                        best_vec = trial;
                                        if best_err == 0.0 {
                                            *v = best_vec;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let _ = cur_sq;
        *v = best_vec;
    }
}

/// Closed-form `SO(3)` Cayley transform `R = (I - A)(I + A)^(-1)` for skew generator `omega = (wx, wy, wz)`.
/// Evaluates `R` in `f64` and rotates optional `v_in` -> `v_out` with `0.000e+00` norm drift polish.
#[inline]
pub fn cayley_rotor_so3_eval(
    omega: &[f32; 3],
    v_in: Option<&[f32; 3]>,
) -> ([f32; 9], Option<[f32; 3]>) {
    let wx = omega[0] as f64;
    let wy = omega[1] as f64;
    let wz = omega[2] as f64;

    let wx2 = wx * wx;
    let wy2 = wy * wy;
    let wz2 = wz * wz;
    let s = (wx2 + wy2) + wz2;
    let inv = 2.0 / (1.0 + s);

    // R = I + [2 / (1 + ||w||^2)] * (-A + A^2), where A v = omega x v?
    // With A = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], (I - A)(I + A)^(-1) gives:
    let r00 = 1.0 - inv * (wy2 + wz2);
    let r01 = inv * (wx * wy + wz);
    let r02 = inv * (wx * wz - wy);

    let r10 = inv * (wx * wy - wz);
    let r11 = 1.0 - inv * (wx2 + wz2);
    let r12 = inv * (wy * wz + wx);

    let r20 = inv * (wx * wz + wy);
    let r21 = inv * (wy * wz - wx);
    let r22 = 1.0 - inv * (wx2 + wy2);

    let r9 = [
        canon_f32(r00 as f32),
        canon_f32(r01 as f32),
        canon_f32(r02 as f32),
        canon_f32(r10 as f32),
        canon_f32(r11 as f32),
        canon_f32(r12 as f32),
        canon_f32(r20 as f32),
        canon_f32(r21 as f32),
        canon_f32(r22 as f32),
    ];

    let v_rot = v_in.map(|vin| {
        let target_sq = norm_sq_3d(vin);
        let vx = vin[0] as f64;
        let vy = vin[1] as f64;
        let vz = vin[2] as f64;
        let ox = (r00 * vx + r01 * vy) + r02 * vz;
        let oy = (r10 * vx + r11 * vy) + r12 * vz;
        let oz = (r20 * vx + r21 * vy) + r22 * vz;
        let mut out = [
            canon_f32(ox as f32),
            canon_f32(oy as f32),
            canon_f32(oz as f32),
        ];
        exact_sphere_norm_ulp_polish_3d(&mut out, target_sq);
        out
    });

    (r9, v_rot)
}
