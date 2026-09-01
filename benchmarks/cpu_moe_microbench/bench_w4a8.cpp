// W4A8 NVFP4 dot 微基准（提取自 cpu_moe_ext.cpp，独立编译，无 CUDA/torch 依赖）
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <immintrin.h>

// 本机 7940H(Phoenix) CPUID: AVX-VNNI=0, AVX512-VNNI=1 → VEX vpdpbusd 非法，
// 统一用 EVEX.256（_mm256_dpbusd_epi32，AVX512VL+VNNI）
#define DPBUSD256(a, b, c) _mm256_dpbusd_epi32(a, b, c)
#ifdef __clang__
#define TGT_AVXVNNI __attribute__((target("avx2,avx512f,avx512bw,avx512vl,avx512vnni,fma")))
#define TGT_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2")))
#else
#define TGT_AVXVNNI
#define TGT_AVX512VNNI
#endif

alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;
  return sign * (1.0f + man / 8.0f) * ldexp(1.0f, (int)exp - 7);
}

static inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// ---- scalar 参考 ----
float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                          const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;
    const int8_t* ao = ae + 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += (float)(kE2M1x2[pk[j] & 0xF] * ae[j]);
      bsum += (float)(kE2M1x2[pk[j] >> 4] * ao[j]);
    }
    acc += (e4m3[scale[b]] * asb[b]) * bsum;
  }
  return acc * (0.5f * global);
}

// ---- AVX-VNNI (256-bit) ----
TGT_AVXVNNI
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));
}

TGT_AVXVNNI
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
    __m256i di = DPBUSD256(_mm256_setzero_si256(), aw, sa);
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

// ---- AVX-512 VNNI ----
TGT_AVX512VNNI
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

TGT_AVX512VNNI
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
  const int pf = 0;  // 预取禁用（诊断）
  int b = 0;
  for (; b + 16 <= nb; b += 16) {
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx, mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx, mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx, mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx, mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx, mask0F, idxsc));
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

int main(int argc, char** argv) {
  const int K = argc > 1 ? atoi(argv[1]) : 4096;
  const int sel = argc > 3 ? atoi(argv[3]) : 0;
  const int iters = argc > 2 ? atoi(argv[2]) : 20000;
  const int nb = K / 16;
  std::vector<uint8_t> packed(nb * 8), scale(nb);
  std::vector<int8_t> asi8(nb * 16);
  std::vector<float> asb(nb), e4m3(256);
  for (int i = 0; i < 256; ++i) e4m3[i] = e4m3_decode((uint8_t)i);
  srand(42);
  for (int i = 0; i < nb * 8; ++i) packed[i] = (uint8_t)(rand() & 0xFF);
  for (int i = 0; i < nb; ++i) scale[i] = (uint8_t)(rand() & 0x7F);
  for (int i = 0; i < nb * 16; ++i) asi8[i] = (int8_t)(rand() % 255 - 127);  // -127..127 排除 -128
  for (int i = 0; i < nb; ++i) asb[i] = 0.01f + 0.05f * (float)(rand() % 100) / 100.0f;
  const float global = 0.25f;
  // grp4 固定参数（诊断用）
  const __m512i lutv = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idxv = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i mask0Fv = _mm512_set1_epi8(0x0F);
  const __m512i idxscv = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  // 正确性（按 sel 隔离）
  float r0 = dot_nvfp4_i8_scalar(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
  float r1 = 0.0f, r2 = 0.0f;
  if (sel == 0 || sel == 2 || sel == 9) r1 = dot_nvfp4_i8_vnni(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
  if (sel == 0 || sel == 3 || sel == 9) r2 = dot_nvfp4_i8_avx512vnni(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
  // 逐块诊断（sel=3 时）
  if (sel == 3 || sel == 9) {
    if (nb >= 4) {
      printf("  blk0 packed: %02x %02x %02x %02x %02x %02x %02x %02x  scale: %u %u %u %u  asb: %.4f %.4f %.4f %.4f\n",
        packed[0], packed[1], packed[2], packed[3], packed[4], packed[5], packed[6], packed[7],
        scale[0], scale[1], scale[2], scale[3], asb[0], asb[1], asb[2], asb[3]);
      printf("  blk0 act: %d %d %d %d %d %d %d %d | %d %d %d %d %d %d %d %d\n",
        asi8[0], asi8[1], asi8[2], asi8[3], asi8[4], asi8[5], asi8[6], asi8[7],
        asi8[8], asi8[9], asi8[10], asi8[11], asi8[12], asi8[13], asi8[14], asi8[15]);
    }
    int bad = 0;
    for (int gb = 0; gb + 4 <= nb; gb += 4) {
      float s4 = 0.0f;
      for (int t = 0; t < 4; ++t) {
        const uint8_t* pk = packed.data() + (size_t)(gb + t) * 8;
        const int8_t* ae = asi8.data() + (size_t)(gb + t) * 16;
        const int8_t* ao = ae + 8;
        int isum = 0;
        for (int j = 0; j < 8; ++j)
          isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
        s4 += (e4m3[scale.data()[gb + t]] * asb[gb + t]) * (float)isum;
      }
      float g4 = _mm512_reduce_add_ps(nvfp4_i8_grp4(packed.data(), scale.data(), asi8.data(), e4m3.data(), asb.data(), gb, lutv, idxv, mask0Fv, idxscv));
      if (fabs(g4 - s4) > 0.05f * fabs(s4) + 1e-3f) {
        if (bad < 8) printf("  grp b=%d: scalar=%.4f grp4=%.4f\n", gb, s4, g4);
        bad++;
      }
    }
    printf("  bad groups: %d / %d\n", bad, nb / 4);
  }
  printf("K=%d  scalar=%.6f vnni=%.6f (err %.2e) avx512vnni=%.6f (err %.2e)\n",
         K, r0, r1, fabs(r1 - r0), r2, fabs(r2 - r0));
  const float rel = 1e-3f * (fabsf(r0) + 1.0f);
if ((sel != 1 && fabsf(r1 - r0) > rel) || (sel != 2 && fabsf(r2 - r0) > rel)) { printf("MISMATCH!\n"); return 1; }

  // 性能（单线程，ping-pong 防优化）
  volatile float sink = 0;
  auto bench = [&](const char* name, float (*fn)(const uint8_t*, const uint8_t*, float,
                   const int8_t*, int, const float*, const float*)) {
    for (int i = 0; i < 1000; ++i) sink += fn(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) sink += fn(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double macs = (double)K * iters / sec;
    double gbs = ((double)nb * 8 + (double)nb * 16) * iters / sec / 1e9;
    printf("%-12s %10.2f ms  %10.3f G MAC/s  %8.2f GB/s (weight+act)\n", name, sec * 1000, macs / 1e9, gbs);
  };
  if (sel == 0 || sel == 1) bench("scalar", dot_nvfp4_i8_scalar);
  if (sel == 0 || sel == 2) bench("avxvnni", dot_nvfp4_i8_vnni);
  if (sel == 0 || sel == 3) bench("avx512vnni", dot_nvfp4_i8_avx512vnni);
  printf("sink=%f\n", (float)sink);
  return 0;
}
