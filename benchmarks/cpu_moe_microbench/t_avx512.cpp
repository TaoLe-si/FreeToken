#include <immintrin.h>
#include <cstdio>
int main() {
  __m512 a = _mm512_set1_ps(1.0f);
  __m512 b = _mm512_add_ps(a, a);
  float r = _mm512_reduce_add_ps(b);
  printf("avx512 ok: %f\n", r);
  // VNNI
  __m512i x = _mm512_set1_epi8(1), y = _mm512_set1_epi8(2);
  __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x, y);
  printf("vnni ok: %d\n", _mm512_reduce_add_epi32(d));
  return 0;
}
