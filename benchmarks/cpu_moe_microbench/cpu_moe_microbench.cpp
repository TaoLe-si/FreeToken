// cpu_moe_microbench — standalone CPU MoE GEMV microbenchmark.
//
// The dot kernels below are transcribed VERBATIM from
// python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp (no torch/CUDA deps), so
// this harness measures the exact production arithmetic on any x86-64 machine:
//   * single-thread dot throughput per SIMD tier (scalar / avx2 / avx512f /
//     avx512bf16) and per weight format (bf16, nvfp4 W4A16, nvfp4 W4A8-VNNI,
//     ds_fp4, q4_0, mxfp4)
//   * multi-threaded pass1-style row-tiled GEMV (atomic work-grab + spin
//     barrier, same shape as the executor) -> achieved DRAM stream GB/s
//   * prefetch-distance A/B for the bf16 AVX-512 BF16 dot (the bandwidth
//     bottleneck kernel) at 1 and N threads
//   * per-token vs grouped (expert-dedup) GEMV at bs>1 — the numbers that
//     decide whether a grouped/batched kernel is worth building
//
// Build (any clang/gcc with x86 target support; zig works on Windows):
//   zig c++ -O3 -march=native -std=c++20 cpu_moe_microbench.cpp -o bench
//   ./bench [--rows 8192] [--n 4096] [--threads 8] [--iter 20]
//
// The measured quantity everywhere is "weight-stream GB/s" — bytes of expert
// weights read per second — because decode GEMV is DRAM-bandwidth-bound.

#include <algorithm>
#include <atomic>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(_M_AMD64) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

// MSVC compatibility: no per-function __attribute__((target(...))) -- compile with
// /arch:AVX2 or /arch:AVX512 and gate kernels on the feature macros below. MSVC
// exposes the AVX-512 BF16 / VNNI intrinsics under /arch:AVX512 even though it does
// not define the corresponding __AVX512*__ macros, so those are enabled manually.
#if defined(_MSC_VER)
#define __attribute__(x)
#include <intrin.h>
static inline int cpu_supports(const char* f) {
  int regs[4] = {0};
  __cpuid(regs, 0);
  const int maxleaf = regs[0];
  int e[4] = {0};
  if (maxleaf >= 7) __cpuidex(e, 7, 0);
  if (!std::strcmp(f, "avx2")) return (e[1] >> 5) & 1;
  if (!std::strcmp(f, "avx512f")) return (e[1] >> 16) & 1;
  if (!std::strcmp(f, "avxvnni")) return (e[2] >> 4) & 1;
  if (!std::strcmp(f, "avx512bf16")) return (e[2] >> 5) & 1;
  if (!std::strcmp(f, "fma")) { __cpuid(e, 1); return (e[2] >> 12) & 1; }
  return 0;
}
#define __builtin_cpu_supports(f) cpu_supports(f)
#if defined(__AVX2__)
#define CPU_MOE_HAS_AVX2 1
#endif
#if defined(__AVX512F__)
#define CPU_MOE_HAS_AVX512F 1
#define CPU_MOE_HAS_AVX2 1
#define CPU_MOE_HAS_AVX512BF16 1
#define CPU_MOE_HAS_AVX512VNNI 1
#define CPU_MOE_HAS_AVXVNNI_256 1
#endif
#else
#if defined(__AVX512F__)
#define CPU_MOE_HAS_AVX512F 1
#endif
#if defined(__AVX512BF16__)
#define CPU_MOE_HAS_AVX512BF16 1
#endif
#if defined(__AVX512VNNI__)
#define CPU_MOE_HAS_AVX512VNNI 1
#endif
#if defined(__AVXVNNI__)
#define CPU_MOE_HAS_AVXVNNI_256 1
#endif
#if defined(__AVX2__)
#define CPU_MOE_HAS_AVX2 1
#endif
#endif

#ifndef PF_AHEAD
#define PF_AHEAD 512
#endif

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ---------------- dot products (verbatim from cpu_moe_ext.cpp) ----------------
using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512F)
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#endif  // CPU_MOE_HAS_AVX512F

#if defined(CPU_MOE_HAS_AVX2)
__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#endif  // CPU_MOE_HAS_AVX2

#if defined(CPU_MOE_HAS_AVX512BF16)
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

// PF-A/B variants of the bf16 AVX-512 BF16 dot (the bandwidth bottleneck kernel).
template <int PF>
__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16_pf(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    if (PF > 0) _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

// 8-accumulator variant: doubles the in-flight load window; tests whether the
// 4-acc kernel leaves memory-level parallelism on the table.
template <int PF>
__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16_pf8(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
  __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 256 <= n; i += 256) {
    if (PF > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF, _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF + 64, _MM_HINT_T0);
    }
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
    a4 = _mm512_dpbf16_ps(a4, load_bh(w + i + 128), load_bh(x + i + 128));
    a5 = _mm512_dpbf16_ps(a5, load_bh(w + i + 160), load_bh(x + i + 160));
    a6 = _mm512_dpbf16_ps(a6, load_bh(w + i + 192), load_bh(x + i + 192));
    a7 = _mm512_dpbf16_ps(a7, load_bh(w + i + 224), load_bh(x + i + 224));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  __m512 s1 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
  __m512 s2 = _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(s1, s2));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16
#endif  // CPU_MOE_X86

// ------------------------ NVFP4 (W4A16) dequant ------------------------------
const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) { f = sign; }
    else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) { man <<= 1; --exp; }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX2)
__attribute__((target("avx2,fma")))
inline __m256 e2m1_decode8(__m256i codes, __m256 mag8) {
  __m256 mag = _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
  __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
  return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}

#endif  // CPU_MOE_HAS_AVX2

#if defined(CPU_MOE_HAS_AVX512F)
__attribute__((target("avx512f")))
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

__attribute__((target("avx512f")))
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 4 <= nb; b += 4) {
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

#endif  // CPU_MOE_HAS_AVX512F

#if defined(CPU_MOE_HAS_AVX2)
__attribute__((target("avx2,fma")))
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

__attribute__((target("avx2,fma")))
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}
#endif  // CPU_MOE_HAS_AVX2

// ---- NVFP4 W4A8 (int8 activations) ----
using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const int8_t*, int,
                             const float*, const float*);

#if defined(CPU_MOE_HAS_AVXVNNI_256)
__attribute__((target("avx2,avxvnni,fma")))
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));
}

__attribute__((target("avx2,avxvnni,fma")))
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global,
                        const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    __m256i w = _mm256_set_m128i(wb1, wb);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asi8 + (size_t)b * 16));
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sa = _mm256_sign_epi8(a, w);
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    __m256 scv = _mm256_blend_ps(_mm256_set1_ps(e4m3[scale[b]] * asb[b]),
                                 _mm256_set1_ps(e4m3[scale[b + 1]] * asb[b + 1]), 0xF0);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), scv, accF);
  }
  float s = hsum256(accF);
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

#endif  // CPU_MOE_HAS_AVXVNNI_256

#if defined(CPU_MOE_HAS_AVX512VNNI)
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  const int pfb = 512;
  const int pf = std::min(pfb, 2 * nb);
  int b = 0;
  for (; b + 16 <= nb; b += 16) {
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                              mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                              mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                              mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                              mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                              mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}
#endif  // avx512vnni
#endif  // CPU_MOE_X86

// ------------------------- ds_fp4 (W4A8) -------------------------------------
using dsdot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                           const float*, const float*);

float dot_dsfp4_scalar(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float sc = e8m0[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    float bsum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += sc * bsum;
  }
  return acc;
}

#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512F)
__attribute__((target("avx512f")))
inline __m512 dsfp4_blk(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                        __m512i loma, float sc) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  return _mm512_mul_ps(prod, _mm512_set1_ps(sc));
}

__attribute__((target("avx512f")))
float dot_dsfp4_avx512(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 32;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
    acc1 = _mm512_add_ps(acc1, dsfp4_blk(packed + (size_t)(b + 1) * 16, xe + (size_t)(b + 1) * 16,
                                         xo + (size_t)(b + 1) * 16, lut, loma, e8m0[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

#endif  // CPU_MOE_HAS_AVX512F

#if defined(CPU_MOE_HAS_AVX2)
__attribute__((target("avx2,fma")))
inline __m256 dsfp4_half_avx2(const uint8_t* pk, const float* xeb, const float* xob, __m256 mag8) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  return _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
}

__attribute__((target("avx2,fma")))
float dot_dsfp4_avx2(const uint8_t* packed, const uint8_t* scale, const float* xe,
                     const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    const __m256 sc = _mm256_set1_ps(e8m0[scale[b]]);
    acc0 = _mm256_fmadd_ps(dsfp4_half_avx2(pk, xeb, xob, mag8), sc, acc0);
    acc1 = _mm256_fmadd_ps(dsfp4_half_avx2(pk + 8, xeb + 8, xob + 8, mag8), sc, acc1);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
}
#endif  // CPU_MOE_HAS_AVX2
#endif

// ------------------------- Q4_0 (W4A8) ---------------------------------------
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX2)
__attribute__((target("f16c")))
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

__attribute__((target("avx2")))
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

__attribute__((target("avx2,fma,f16c")))
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);
    __m256i sa = _mm256_sign_epi8(a, wq);
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

#endif  // CPU_MOE_HAS_AVX2

#if defined(CPU_MOE_HAS_AVXVNNI_256)
__attribute__((target("avx2,avxvnni,fma,f16c")))
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);
    __m256i sa = _mm256_sign_epi8(a, wq);
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_HAS_AVXVNNI_256
#endif

// ------------------------- mxfp4 GEMV ----------------------------------------
using mxgemv_fn = void (*)(float*, const uint8_t*, const uint8_t*, const bf16_t*, int, int,
                           int, const float*, const float*);

void mxfp4_gemv_scalar(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  for (int c = 0; c < ncol; ++c) out[c] = 0.0f;
  for (int kb = 0; kb < Kpairs; ++kb) {
    const uint8_t* w = blk + (size_t)kb * N2;
    const uint8_t* s = scl + (size_t)(kb >> 4) * N2;
    const float xl = bf16_to_f32(x[2 * kb]);
    const float xh = bf16_to_f32(x[2 * kb + 1]);
    for (int c = 0; c < ncol; ++c) {
      const uint8_t byte = w[c];
      out[c] += (e2m1[byte & 0xF] * xl + e2m1[byte >> 4] * xh) * e8m0[s[c]];
    }
  }
}

#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512F)
__attribute__((target("avx512f")))
void mxfp4_gemv_avx512(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  int c0 = 0;
  for (; c0 + 16 <= ncol; c0 += 64) {
    const int nchunk = std::min(4, (ncol - c0) / 16);
    __m512 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {
      __m512 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 16));
        sc[ci] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(sraw), 23));
      }
      __m512 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm512_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m512 xl = _mm512_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m512 xh = _mm512_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m512i wi = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(wbase + ci * 16)));
          __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
          __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
          blk_acc[ci] = _mm512_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm512_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm512_storeu_ps(out + c0 + ci * 16, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}
#endif  // CPU_MOE_HAS_AVX512F
#endif

// --------------------------- benchmark drivers -------------------------------
static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double best_time(std::function<void()> fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_s();
    fn();
    double dt = now_s() - t0;
    if (dt < best) best = dt;
  }
  return best;
}

// Row-parallel pass1-style GEMV (atomic work-grab over 32-row tiles + barrier).
struct GemvCtx {
  int rows, n, nthreads, tile;
  const bf16_t* w;
  const bf16_t* x;
  float* out;
  dot_fn dot;
  std::atomic<int64_t> next{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};
};

void gemv_barrier(GemvCtx& c, int& local_sense) {
  local_sense ^= 1;
  if (c.bar_count.fetch_add(1) + 1 == c.nthreads) {
    c.bar_count.store(0);
    c.bar_sense.store(local_sense);
  } else {
    while (c.bar_sense.load() != local_sense) _mm_pause();
  }
}

void gemv_worker(GemvCtx* c, int tid) {
  int ls = 0;
  const int ntiles = (c->rows + c->tile - 1) / c->tile;
  for (;;) {
    const int64_t p = c->next.fetch_add(1, std::memory_order_relaxed);
    if (p >= ntiles) break;
    const int r0 = (int)p * c->tile;
    const int r1 = std::min(c->rows, r0 + c->tile);
    for (int r = r0; r < r1; ++r)
      c->out[r] = c->dot(c->w + (size_t)r * c->n, c->x, c->n);
  }
  gemv_barrier(*c, ls);
}

// Grouped (expert-dedup) GEMV: one weight row read, M activations dotted.
struct GrpCtx {
  int rows, n, nthreads, tile, M;
  const bf16_t* w;
  const bf16_t* xm;   // M*n
  float* out;         // M*rows
  dot_fn dot;
  std::atomic<int64_t> next{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};
};

void grp_barrier(GrpCtx& c, int& local_sense) {
  local_sense ^= 1;
  if (c.bar_count.fetch_add(1) + 1 == c.nthreads) {
    c.bar_count.store(0);
    c.bar_sense.store(local_sense);
  } else {
    while (c.bar_sense.load() != local_sense) _mm_pause();
  }
}

void grp_worker(GrpCtx* c, int tid) {
  int ls = 0;
  const int ntiles = (c->rows + c->tile - 1) / c->tile;
  for (;;) {
    const int64_t p = c->next.fetch_add(1, std::memory_order_relaxed);
    if (p >= ntiles) break;
    const int r0 = (int)p * c->tile;
    const int r1 = std::min(c->rows, r0 + c->tile);
    for (int r = r0; r < r1; ++r) {
      const bf16_t* wrow = c->w + (size_t)r * c->n;
      for (int m = 0; m < c->M; ++m)
        c->out[(size_t)m * c->rows + r] = c->dot(wrow, c->xm + (size_t)m * c->n, c->n);
    }
  }
  grp_barrier(*c, ls);
}

static int parse_int(const char* name, int def) {
  const char* v = getenv(name);
  return v && v[0] ? atoi(v) : def;
}

int main(int argc, char** argv) {
  int rows = parse_int("BENCH_ROWS", 8192);
  int n    = parse_int("BENCH_N", 4096);
  int thr  = parse_int("BENCH_THREADS", (int)std::thread::hardware_concurrency());
  int iter = parse_int("BENCH_ITER", 20);
  printf("CPU: %d threads | rows=%d n=%d iter=%d\n", thr, rows, n, iter);
#if CPU_MOE_X86
  { int e[4] = {0}, maxl = 0; __cpuid(e, 0); maxl = e[0]; if (maxl >= 7) __cpuidex(e, 7, 0); \
    printf("CPUID leaf7: ebx=0x%08x ecx=0x%08x edx=0x%08x\n", e[1], e[2], e[3]); }
  printf("ISA: avx512bf16=%d avx512f=%d avx2=%d avxvnni=%d\n",
         __builtin_cpu_supports("avx512bf16"), __builtin_cpu_supports("avx512f"),
         __builtin_cpu_supports("avx2"), __builtin_cpu_supports("avxvnni"));
#else
  printf("ISA: scalar only\n");
#endif

  std::vector<bf16_t> w((size_t)rows * n);
  std::vector<bf16_t> x(n);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < n; ++i) w[(size_t)r * n + i] = f32_to_bf16((float)((r * 31 + i * 17) % 251) / 100.0f - 1.25f);
  for (int i = 0; i < n; ++i) x[i] = f32_to_bf16((float)((i * 13) % 199) / 100.0f - 0.99f);
  std::vector<float> out(rows);

  // 1) single-thread dot throughput per tier (weight-stream GB/s)
  struct { const char* name; dot_fn fn; } dots[] = {
    {"bf16 scalar    ", dot_scalar},
#if defined(CPU_MOE_HAS_AVX2)
    {"bf16 avx2      ", dot_avx2},
#endif
#if defined(CPU_MOE_HAS_AVX512F)
    {"bf16 avx512f   ", dot_avx512f},
#endif
#if defined(CPU_MOE_HAS_AVX512BF16)
    {"bf16 avx512bf16", dot_avx512bf16},
#endif
  };
  // correctness vs scalar
  for (auto& d : dots) {
    if (d.fn == dot_scalar) continue;
    float a = 0, b = 0;
    for (int r = 0; r < rows; r += 997) { a += d.fn(w.data() + (size_t)r * n, x.data(), n); b += dot_scalar(w.data() + (size_t)r * n, x.data(), n); }
    double rel = std::fabs(a - b) / (std::fabs(b) + 1e-9);
    if (rel > 1e-4) printf("!! correctness mismatch %s: rel=%.2e\n", d.name, rel);
  }
  printf("\n-- single-thread bf16 dot (weight-stream GB/s) --\n");
  for (auto& d : dots) {
    double t = best_time([&] { for (int r = 0; r < rows; ++r) out[r] = d.fn(w.data() + (size_t)r * n, x.data(), n); }, 3, iter);
    double gbs = (double)rows * n * 2 / t / 1e9;
    printf("  %s: %7.1f GB/s  (%.1f us / %d rows)\n", d.name, gbs, t * 1e6, rows);
  }

  // 2) prefetch-distance A/B on the AVX-512 BF16 dot
#if defined(CPU_MOE_HAS_AVX512BF16)
  printf("\n-- avx512bf16 prefetch-distance A/B (single-thread, 4 acc) --\n");
  const int pfs[] = {0, 128, 256, 512, 1024, 2048, 4096, 8192};
#define BENCH_PF_CASE(PFV) \
  case PFV: { \
    double t = best_time([&] { for (int r = 0; r < rows; ++r) out[r] = dot_avx512bf16_pf<PFV>(w.data() + (size_t)r * n, x.data(), n); }, 3, iter); \
    printf("  pf=%5d: %7.1f GB/s\n", PFV, (double)rows * n * 2 / t / 1e9); \
    break; \
  }
  for (int pf : pfs) {
    switch (pf) {
      BENCH_PF_CASE(0)
      BENCH_PF_CASE(128)
      BENCH_PF_CASE(256)
      BENCH_PF_CASE(512)
      BENCH_PF_CASE(1024)
      BENCH_PF_CASE(2048)
      BENCH_PF_CASE(4096)
      BENCH_PF_CASE(8192)
      default: break;
    }
  }
#undef BENCH_PF_CASE
  printf("-- avx512bf16 accumulator-count x prefetch A/B --\n");
#define BENCH_ACC_CASE(PFV) \
  case PFV: { \
    double t4 = best_time([&] { for (int r = 0; r < rows; ++r) out[r] = dot_avx512bf16_pf<PFV>(w.data() + (size_t)r * n, x.data(), n); }, 3, iter); \
    double t8 = best_time([&] { for (int r = 0; r < rows; ++r) out[r] = dot_avx512bf16_pf8<PFV>(w.data() + (size_t)r * n, x.data(), n); }, 3, iter); \
    printf("  pf=%4d: 4 acc %7.1f GB/s | 8 acc %7.1f GB/s\n", PFV, (double)rows * n * 2 / t4 / 1e9, (double)rows * n * 2 / t8 / 1e9); \
    break; \
  }
  for (int pf : {0, 128, 256, 512, 1024}) {
    switch (pf) {
      BENCH_ACC_CASE(0)
      BENCH_ACC_CASE(128)
      BENCH_ACC_CASE(256)
      BENCH_ACC_CASE(512)
      BENCH_ACC_CASE(1024)
      default: break;
    }
  }
#undef BENCH_ACC_CASE
#endif

  // 3) multi-threaded pass1-style GEMV (bf16), scaling threads
  printf("\n-- multi-thread bf16 GEMV (pass1 shape, weight-stream GB/s) --\n");
  for (int nt : {1, 2, 4, 8, thr}) {
    GemvCtx c;
    c.rows = rows; c.n = n; c.nthreads = nt; c.tile = 32;
    c.w = w.data(); c.x = x.data(); c.out = out.data(); c.dot = dot_scalar;
#if defined(CPU_MOE_HAS_AVX512BF16)
    c.dot = dot_avx512bf16;
#endif
    double t = best_time([&] {
      c.next.store(0, std::memory_order_relaxed);
      c.bar_count.store(0, std::memory_order_relaxed);
      c.bar_sense.store(0, std::memory_order_relaxed);
      std::vector<std::thread> ts;
      for (int i = 0; i < nt; ++i) ts.emplace_back(gemv_worker, &c, i);
      for (auto& th : ts) th.join();
    }, 2, std::max(3, iter / 2));
    printf("  threads=%2d: %7.1f GB/s\n", nt, (double)rows * n * 2 / t / 1e9);
  }

  // 4) grouped vs per-token GEMV at bs>1 (the expert-dedup question)
#if defined(CPU_MOE_HAS_AVX512BF16)
  {
    const int M = parse_int("BENCH_M", 4);
    std::vector<bf16_t> xm((size_t)M * n);
    for (int m = 0; m < M; ++m)
      for (int i = 0; i < n; ++i) xm[(size_t)m * n + i] = f32_to_bf16((float)((i * 7 + m * 11) % 173) / 100.0f - 0.86f);
    std::vector<float> gout((size_t)M * rows);
    printf("\n-- bs=%d grouped (dedup) vs per-token GEMV, threads=%d --\n", M, thr);
    // per-token: M passes over the weight rows
    double tpt = best_time([&] {
      for (int m = 0; m < M; ++m)
        for (int r = 0; r < rows; ++r) gout[(size_t)m * rows + r] = dot_avx512bf16(w.data() + (size_t)r * n, xm.data() + (size_t)m * n, n);
    }, 2, std::max(3, iter / 2));
    printf("  per-token (M weight passes): %.1f us  | weight GB/s %.1f | routes/s %.1f k\n",
           tpt * 1e6, (double)M * rows * n * 2 / tpt / 1e9, (double)M * rows / tpt / 1e3);
    // grouped: one weight pass, M dots per row (single thread: x stays in L1)
    double tgrp = best_time([&] {
      for (int r = 0; r < rows; ++r) {
        const bf16_t* wrow = w.data() + (size_t)r * n;
        for (int m = 0; m < M; ++m) gout[(size_t)m * rows + r] = dot_avx512bf16(wrow, xm.data() + (size_t)m * n, n);
      }
    }, 2, std::max(3, iter / 2));
    printf("  grouped   (1 weight pass):   %.1f us  | weight GB/s %.1f | routes/s %.1f k\n",
           tgrp * 1e6, (double)rows * n * 2 / tgrp / 1e9, (double)M * rows / tgrp / 1e3);
    // multi-thread grouped
    GrpCtx gc;
    gc.rows = rows; gc.n = n; gc.nthreads = thr; gc.tile = 32; gc.M = M;
    gc.w = w.data(); gc.xm = xm.data(); gc.out = gout.data(); gc.dot = dot_avx512bf16;
    double tgmt = best_time([&] {
      gc.next.store(0, std::memory_order_relaxed);
      gc.bar_count.store(0, std::memory_order_relaxed);
      gc.bar_sense.store(0, std::memory_order_relaxed);
      std::vector<std::thread> ts;
      for (int i = 0; i < thr; ++i) ts.emplace_back(grp_worker, &gc, i);
      for (auto& th : ts) th.join();
    }, 2, std::max(3, iter / 2));
    printf("  grouped MT (threads=%d):     %.1f us  | weight GB/s %.1f | routes/s %.1f k\n",
           thr, tgmt * 1e6, (double)rows * n * 2 / tgmt / 1e9, (double)M * rows / tgmt / 1e3);
  }
#endif  // CPU_MOE_HAS_AVX512BF16

  // ---- host DRAM streaming read bandwidth (sequential, no reuse) ----
  {
    const size_t dram_bytes = (size_t)512 << 20;  // 512MB buffer (>> L3)
    std::vector<char> buf(dram_bytes, 1);
    printf("\n-- host DRAM streaming read bandwidth (512MB, sequential) --\n");
    const int tcs[] = {1, 8, 16};
    for (int tc : tcs) {
      const int iters = 3;
      auto t0 = std::chrono::steady_clock::now();
      std::atomic<size_t> total{0};
      std::vector<std::thread> ts;
      const size_t chunk = dram_bytes / tc;
      for (int t = 0; t < tc; ++t) {
        ts.emplace_back([&, t] {
          const char* p = buf.data() + (size_t)t * chunk;
          __m128i acc = _mm_setzero_si128();
          for (int it = 0; it < iters; ++it)
            for (size_t i = 0; i < chunk; i += 64) {
              acc = _mm_add_epi64(acc, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i)));
              acc = _mm_add_epi64(acc, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i + 16)));
              acc = _mm_add_epi64(acc, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i + 32)));
              acc = _mm_add_epi64(acc, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i + 48)));
            }
          total.fetch_add((size_t)_mm_cvtsi128_si64(acc), std::memory_order_relaxed);
        });
      }
      for (auto& th : ts) th.join();
      auto t1 = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t1 - t0).count();
      volatile size_t sink = total.load();
      (void)sink;
      printf("  threads=%2d: %7.1f GB/s\n", tc, (double)dram_bytes * iters / secs / 1e9);
    }
  }

  // ---- peak MAC throughput (L2-resident weights, per-thread private buffer) ----
  {
    printf("\n-- peak MAC throughput (L2-resident bf16 dot, per-core private 1MB) --\n");
#if defined(CPU_MOE_HAS_AVX512BF16)
    auto dotp = dot_avx512bf16;
#elif defined(CPU_MOE_HAS_AVX512F)
    auto dotp = dot_avx512f;
#else
    auto dotp = dot_avx2;
#endif
    const int pr = 256, pn = 2048;  // 256*2048*2B = 1MB per thread
    std::vector<std::vector<bf16_t>> wbufs(16);
    for (int t = 0; t < 16; ++t) {
      wbufs[t].resize((size_t)pr * pn);
      for (int i = 0; i < pr * pn; ++i)
        wbufs[t][i] = f32_to_bf16((float)((i * 31 + t * 17) % 997) / 100.0f - 0.5f);
    }
    std::vector<bf16_t> xb(pn);
    for (int i = 0; i < pn; ++i) xb[i] = f32_to_bf16((float)((i * 13) % 503) / 100.0f - 0.25f);
    auto bench_mac = [&](int tc, int iters) -> double {
      auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> ts;
      for (int t = 0; t < tc; ++t) {
        ts.emplace_back([&, t] {
          const bf16_t* w = wbufs[t].data();
          const bf16_t* x = xb.data();
          float s = 0.0f;
          for (int it = 0; it < iters; ++it)
            for (int r = 0; r < pr; ++r) s += dotp(w + (size_t)r * pn, x, pn);
          volatile float vs = s;
          (void)vs;
        });
      }
      for (auto& th : ts) th.join();
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double>(t1 - t0).count();
    };
    double t1s = bench_mac(1, 2000);
    printf("  single-thread: %7.1f GMAC/s\n", (double)pr * pn * 2000 / t1s / 1e9);
    double t8 = bench_mac(8, 2000);
    printf("  8-thread:      %7.1f GMAC/s  (%.1f T MAC/s)\n", (double)8 * pr * pn * 2000 / t8 / 1e9, (double)8 * pr * pn * 2000 / t8 / 1e12);
    double t16 = bench_mac(16, 2000);
    printf("  16-thread:     %7.1f GMAC/s  (%.1f T MAC/s)\n", (double)16 * pr * pn * 2000 / t16 / 1e9, (double)16 * pr * pn * 2000 / t16 / 1e12);
    // L1-resident single row (w+x = 8KB, fits L1D): pure instruction-throughput ceiling
    auto bench_l1 = [&](int tc, int iters) -> double {
      auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> ts;
      for (int t = 0; t < tc; ++t) {
        ts.emplace_back([&, t] {
          const bf16_t* w = wbufs[t].data();
          const bf16_t* x = xb.data();
          float s = 0.0f;
          for (int it = 0; it < iters; ++it) s += dotp(w, x, pn);
          volatile float vs = s;
          (void)vs;
        });
      }
      for (auto& th : ts) th.join();
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double>(t1 - t0).count();
    };
    double l1t1 = bench_l1(1, 300000);
    printf("  L1 single-row 1-thread: %7.1f GMAC/s\n", (double)pn * 300000 / l1t1 / 1e9);
    double l1t8 = bench_l1(8, 300000);
    printf("  L1 single-row 8-thread: %7.1f GMAC/s  (%.2f T MAC/s)\n", (double)8 * pn * 300000 / l1t8 / 1e9, (double)8 * pn * 300000 / l1t8 / 1e12);
#if defined(CPU_MOE_HAS_AVX512VNNI)
    // NVFP4 W4A8 (AVX-512 VNNI) L1-resident peak: ~1.5B/MAC data flow vs 4B/MAC bf16
    {
      const int kn = 2048;
      std::vector<uint8_t> wpk(kn / 2, 0x18);
      std::vector<uint8_t> wsc(kn / 16, 7);  // scale idx 7 -> e4m3=12
      std::vector<float> wasb(kn / 16, 1.0f);
      std::vector<int8_t> wai8(kn, 1);
      float we4[16];
      for (int i = 0; i < 16; ++i) we4[i] = (float)kE2M1x2[i];
      auto bench_w8 = [&](int tc, int iters) -> double {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int t = 0; t < tc; ++t) {
          ts.emplace_back([&] {
            float s = 0.0f;
            for (int it = 0; it < iters; ++it)
              s += dot_nvfp4_i8_avx512vnni(wpk.data(), wsc.data(), 1.0f, wai8.data(), kn, we4, wasb.data());
            volatile float vs = s;
            (void)vs;
          });
        }
        for (auto& th : ts) th.join();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
      };
      double w1 = bench_w8(1, 300000);
      double w8 = bench_w8(8, 300000);
      printf("  W4A8 VNNI 1-thread: %7.1f GMAC/s | 8-thread: %7.1f GMAC/s (%.2f T MAC/s)\n",
             (double)kn * 300000 / w1 / 1e9, (double)8 * kn * 300000 / w8 / 1e9,
             (double)8 * kn * 300000 / w8 / 1e12);
    }
#endif
  }

  printf("\ndone.\n");
  return 0;
}
