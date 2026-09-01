// 多行 GEMV 微基准：M 行 × K，共享 int8 act，多线程（模拟真实 MoE 专家）
// 用法: bench_w4a8_multi.exe [M] [K] [threads] [iters]
#include <immintrin.h>
#ifdef __clang__
#define TGT_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2")))
#else
#define TGT_AVX512VNNI
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

static const int8_t kE2M1x2[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static float e4m3_decode(uint8_t b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0xF, man = b & 7;
    float m = (float)man / 8.0f;
    if (exp == 0) return sign * m * (float)ldexp(1.0, -6);
    return sign * (1.0f + m) * (float)ldexp(1.0, exp - 7);
}
TGT_AVX512VNNI
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F, __m512i idxsc) {
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
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
    const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    const __m512i mask0F = _mm512_set1_epi8(0x0F);
    const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    const int nb = K / 16;
    int b = 0;
    for (; b + 16 <= nb; b += 16) {
        acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx, mask0F, idxsc));
        acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx, mask0F, idxsc));
        acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx, mask0F, idxsc));
        acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx, mask0F, idxsc));
    }
    for (; b + 4 <= nb; b += 4)
        acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx, mask0F, idxsc));
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
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
    const int M = argc > 1 ? atoi(argv[1]) : 32;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    const int NT = argc > 3 ? atoi(argv[3]) : 16;
    const int iters = argc > 4 ? atoi(argv[4]) : 300;
    const int nb = K / 16;
    std::vector<uint8_t> packed((size_t)M * nb * 8), scale((size_t)M * nb);
    std::vector<int8_t> asi8((size_t)nb * 16);
    std::vector<float> asb(nb), e4m3(256), out(M);
    for (int i = 0; i < 256; ++i) e4m3[i] = e4m3_decode((uint8_t)i);
    srand(42);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < scale.size(); ++i) scale[i] = (uint8_t)(rand() & 0x7F);
    for (int i = 0; i < nb * 16; ++i) asi8[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < nb; ++i) asb[i] = 0.01f + 0.05f * (float)(rand() % 100) / 100.0f;
    const float global = 0.25f;

    float r0 = dot_nvfp4_i8_avx512vnni(packed.data(), scale.data(), global, asi8.data(), K, e4m3.data(), asb.data());
    float rs = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int isum = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t pk = packed[(size_t)b * 8 + j];
            isum += (int)kE2M1x2[pk & 0xF] * (int)asi8[b*16+j] + (int)kE2M1x2[pk >> 4] * (int)asi8[b*16+8+j];
        }
        rs += (e4m3[scale[b]] * asb[b]) * (float)isum;
    }
    rs *= 0.5f * global;
    printf("M=%d K=%d threads=%d  dot0=%.6f scalar=%.6f err=%.2e %s\n", M, K, NT, r0, rs, fabsf(r0 - rs), fabsf(r0 - rs) < 1e-3f * (fabsf(rs) + 1.0f) ? "OK" : "MISMATCH");

    auto worker = [&](int t) {
        int rows = (M + NT - 1) / NT;
        int start = t * rows;
        int end = std::min(start + rows, M);
        for (int r = start; r < end; ++r)
            out[r] = dot_nvfp4_i8_avx512vnni(packed.data() + (size_t)r * nb * 8,
                                             scale.data() + (size_t)r * nb, global,
                                             asi8.data(), K, e4m3.data(), asb.data());
    };
    for (int t = 0; t < NT; ++t) worker(t);
    double best = 1e18;
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ths;
        for (int t = 0; t < NT; ++t) ths.emplace_back(worker, t);
        for (auto& th : ths) th.join();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, dt);
    }
    volatile float sink = 0;
    for (int i = 0; i < M; ++i) sink += out[i];
    double macs = (double)M * K;
    double wbytes = (double)M * nb * 8 + (double)nb * 16 + (double)M * nb + (double)nb * 4;
    printf("multi: %.3f ms  %7.1f G MAC/s  %6.1f GB/s (w+act)\n", best * 1000, macs / best / 1e9, wbytes / best / 1e9);
    return 0;
}
