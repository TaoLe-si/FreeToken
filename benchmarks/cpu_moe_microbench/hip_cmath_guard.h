// hip_cmath_guard.h -- push_macro workaround for MSVC + ROCm 6.4 cmath conflict.
// 
// MSVC's <cmath> declares isgreater/isless/etc. as __host__ __device__ via
// _CLANG_BUILTIN2 macros (only in ClangCL mode, i.e. when clang frontend
// is driven by MSVC ABI). ROCm 6.4's __clang_cuda_math_forward_declares.h
// redeclares them as __device__ only, causing "cannot overload" errors.
//
// Workaround: include this BEFORE <hip/hip_runtime.h>. The header will:
//   1. #include <cmath> to trigger MSVC's declarations.
//   2. #pragma push_macro + #undef for each conflict, so they don't conflict.
//   3. When hip/hip_runtime.h is later included, MSVC's versions are gone,
//      ROCm can declare its __device__ versions cleanly.

#pragma once

#include <cmath>

#if defined(__clang__) && defined(_MSC_VER)
#  pragma push_macro("isgreater")
#  pragma push_macro("isgreaterequal")
#  pragma push_macro("isless")
#  pragma push_macro("islessequal")
#  pragma push_macro("islessgreater")
#  pragma push_macro("isunordered")
#  pragma push_macro("isfinite")
#  pragma push_macro("isinf")
#  pragma push_macro("isnan")
#  pragma push_macro("isnormal")
#  pragma push_macro("signbit")
#  undef isgreater
#  undef isgreaterequal
#  undef isless
#  undef islessequal
#  undef islessgreater
#  undef isunordered
#  undef isfinite
#  undef isinf
#  undef isnan
#  undef isnormal
#  undef signbit
#endif
