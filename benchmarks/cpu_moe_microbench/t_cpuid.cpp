#include <cstdio>
#include <intrin.h>
int main() {
  int r0[4], r1[4];
  __cpuidex(r0, 7, 0);
  __cpuidex(r1, 7, 1);
  printf("CPUID.7.0:ECX bit11 AVX512-VNNI: %d\n", (r0[2] >> 11) & 1);
  printf("CPUID.7.1:EAX bit4  AVX-VNNI:    %d\n", (r1[0] >> 4) & 1);
  printf("CPUID.7.0:EBX bit16 AVX512F:     %d\n", (r0[1] >> 16) & 1);
  printf("CPUID.7.0:EBX bit30 AVX512BW:    %d\n", (r0[1] >> 30) & 1);
  printf("CPUID.7.0:EBX bit31 AVX512VL:    %d\n", (r0[1] >> 31) & 1);
  printf("CPUID.7.0:ECX bit5  AVX512VBMI:  %d\n", (r0[2] >> 5) & 1);
  return 0;
}
