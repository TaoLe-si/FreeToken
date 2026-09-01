#include <immintrin.h>
#include <cstdio>
int main() {
  __m256i x = _mm256_set1_epi8(1), y = _mm256_set1_epi8(2);
  __m256i d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
  int r = _mm256_extract_epi32(d, 0) + _mm256_extract_epi32(d, 1) + _mm256_extract_epi32(d, 2) + _mm256_extract_epi32(d, 3);
  printf("evex256 ok: %d\n", r);
  return 0;
}
