#pragma section("__nv_managed_data__")
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1 "t.cu"
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#define __nv_is_extended_device_lambda_with_preserved_return_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)&& defined(__nv_is_extended_device_lambda_with_preserved_return_type)
#endif

#line 1
#line 67 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin/../include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\sal.h"
extern "C" {
#line 2971 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\sal.h"
}
#line 22 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\concurrencysal.h"
extern "C" {
#line 391 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\concurrencysal.h"
}
#line 19 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 51 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 55
extern "C" {
#line 65 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 76 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
typedef char *va_list; 
#line 153 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 165 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
}
#line 169 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
extern "C++" {
#line 171
template< class _Ty> 
#line 172
struct __vcrt_va_list_is_reference { 
#line 174
enum: bool { __the_value}; 
#line 175
}; 
#line 177
template< class _Ty> 
#line 178
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 180
enum: bool { __the_value = '\001'}; 
#line 181
}; 
#line 183
template< class _Ty> 
#line 184
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 186
enum: bool { __the_value = '\001'}; 
#line 187
}; 
#line 189
template< class _Ty> 
#line 190
struct __vcrt_assert_va_start_is_not_reference { 
#line 192
static_assert((!__vcrt_va_list_is_reference< _Ty> ::__the_value), "va_start argument must not have reference type and must not be parenthesized");
#line 194
}; 
#line 195
}
#line 205 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vadefs.h"
#pragma warning(pop)
#pragma pack ( pop )
#line 60 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 96 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 188 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#line 189
typedef __int64 ptrdiff_t; 
#line 190
typedef __int64 intptr_t; 
#line 198 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 245 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
extern "C++" {
#line 247
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 251
}
#line 380 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 389 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 390
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 394 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 402 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 404
#pragma warning(pop)
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 125
__pragma( pack ( push, 8 )) extern "C" {
#line 254 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
extern "C++" {
#line 256
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 259
template< class _Ty> 
#line 260
struct _CrtEnableIf< true, _Ty>  { 
#line 262
typedef _Ty _Type; 
#line 263
}; 
#line 264
}
#line 268 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 371 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
void __cdecl _invalid_parameter_noinfo(); 
#line 372
__declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 374
__declspec(noreturn) void __cdecl 
#line 375
_invoke_watson(const __wchar_t * _Expression, const __wchar_t * _FunctionName, const __wchar_t * _FileName, unsigned _LineNo, uintptr_t _Reserved); 
#line 604 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 605
typedef unsigned short wint_t; 
#line 606
typedef unsigned short wctype_t; 
#line 607
typedef long __time32_t; 
#line 608
typedef __int64 __time64_t; 
#line 615
typedef 
#line 610
struct __crt_locale_data_public { 
#line 612
const unsigned short *_locale_pctype; 
#line 613
int _locale_mb_cur_max; 
#line 614
unsigned _locale_lc_codepage; 
#line 615
} __crt_locale_data_public; 
#line 621
typedef 
#line 617
struct __crt_locale_pointers { 
#line 619
struct __crt_locale_data *locinfo; 
#line 620
struct __crt_multibyte_data *mbcinfo; 
#line 621
} __crt_locale_pointers; 
#line 623
typedef __crt_locale_pointers *_locale_t; 
#line 629
typedef 
#line 625
struct _Mbstatet { 
#line 627
unsigned long _Wchar; 
#line 628
unsigned short _Byte, _State; 
#line 629
} _Mbstatet; 
#line 631
typedef _Mbstatet mbstate_t; 
#line 684 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 694 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 2111 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 2114
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wctype.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wctype.h"
const unsigned short *__cdecl __pctype_func(); 
#line 36
const wctype_t *__cdecl __pwctype_func(); 
#line 67 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wctype.h"
int __cdecl iswalnum(wint_t _C); 
#line 68
int __cdecl iswalpha(wint_t _C); 
#line 69
int __cdecl iswascii(wint_t _C); 
#line 70
int __cdecl iswblank(wint_t _C); 
#line 71
int __cdecl iswcntrl(wint_t _C); 
#line 74
int __cdecl iswdigit(wint_t _C); 
#line 76
int __cdecl iswgraph(wint_t _C); 
#line 77
int __cdecl iswlower(wint_t _C); 
#line 78
int __cdecl iswprint(wint_t _C); 
#line 79
int __cdecl iswpunct(wint_t _C); 
#line 80
int __cdecl iswspace(wint_t _C); 
#line 81
int __cdecl iswupper(wint_t _C); 
#line 82
int __cdecl iswxdigit(wint_t _C); 
#line 83
int __cdecl __iswcsymf(wint_t _C); 
#line 84
int __cdecl __iswcsym(wint_t _C); 
#line 86
int __cdecl _iswalnum_l(wint_t _C, _locale_t _Locale); 
#line 87
int __cdecl _iswalpha_l(wint_t _C, _locale_t _Locale); 
#line 88
int __cdecl _iswblank_l(wint_t _C, _locale_t _Locale); 
#line 89
int __cdecl _iswcntrl_l(wint_t _C, _locale_t _Locale); 
#line 90
int __cdecl _iswdigit_l(wint_t _C, _locale_t _Locale); 
#line 91
int __cdecl _iswgraph_l(wint_t _C, _locale_t _Locale); 
#line 92
int __cdecl _iswlower_l(wint_t _C, _locale_t _Locale); 
#line 93
int __cdecl _iswprint_l(wint_t _C, _locale_t _Locale); 
#line 94
int __cdecl _iswpunct_l(wint_t _C, _locale_t _Locale); 
#line 95
int __cdecl _iswspace_l(wint_t _C, _locale_t _Locale); 
#line 96
int __cdecl _iswupper_l(wint_t _C, _locale_t _Locale); 
#line 97
int __cdecl _iswxdigit_l(wint_t _C, _locale_t _Locale); 
#line 98
int __cdecl _iswcsymf_l(wint_t _C, _locale_t _Locale); 
#line 99
int __cdecl _iswcsym_l(wint_t _C, _locale_t _Locale); 
#line 102
wint_t __cdecl towupper(wint_t _C); 
#line 103
wint_t __cdecl towlower(wint_t _C); 
#line 104
int __cdecl iswctype(wint_t _C, wctype_t _Type); 
#line 106
wint_t __cdecl _towupper_l(wint_t _C, _locale_t _Locale); 
#line 107
wint_t __cdecl _towlower_l(wint_t _C, _locale_t _Locale); 
#line 108
int __cdecl _iswctype_l(wint_t _C, wctype_t _Type, _locale_t _Locale); 
#line 112
int __cdecl isleadbyte(int _C); 
#line 113
int __cdecl _isleadbyte_l(int _C, _locale_t _Locale); 
#line 115
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using iswctype instea" "d. See online help for details.")) int __cdecl is_wctype(wint_t _C, wctype_t _Type); 
#line 203 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wctype.h"
}__pragma( pack ( pop )) 
#line 205
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 29
int __cdecl _isctype(int _C, int _Type); 
#line 30
int __cdecl _isctype_l(int _C, int _Type, _locale_t _Locale); 
#line 31
int __cdecl isalpha(int _C); 
#line 32
int __cdecl _isalpha_l(int _C, _locale_t _Locale); 
#line 33
int __cdecl isupper(int _C); 
#line 34
int __cdecl _isupper_l(int _C, _locale_t _Locale); 
#line 35
int __cdecl islower(int _C); 
#line 36
int __cdecl _islower_l(int _C, _locale_t _Locale); 
#line 39
int __cdecl isdigit(int _C); 
#line 41
int __cdecl _isdigit_l(int _C, _locale_t _Locale); 
#line 42
int __cdecl isxdigit(int _C); 
#line 43
int __cdecl _isxdigit_l(int _C, _locale_t _Locale); 
#line 46
int __cdecl isspace(int _C); 
#line 48
int __cdecl _isspace_l(int _C, _locale_t _Locale); 
#line 49
int __cdecl ispunct(int _C); 
#line 50
int __cdecl _ispunct_l(int _C, _locale_t _Locale); 
#line 51
int __cdecl isblank(int _C); 
#line 52
int __cdecl _isblank_l(int _C, _locale_t _Locale); 
#line 53
int __cdecl isalnum(int _C); 
#line 54
int __cdecl _isalnum_l(int _C, _locale_t _Locale); 
#line 55
int __cdecl isprint(int _C); 
#line 56
int __cdecl _isprint_l(int _C, _locale_t _Locale); 
#line 57
int __cdecl isgraph(int _C); 
#line 58
int __cdecl _isgraph_l(int _C, _locale_t _Locale); 
#line 59
int __cdecl iscntrl(int _C); 
#line 60
int __cdecl _iscntrl_l(int _C, _locale_t _Locale); 
#line 63
int __cdecl toupper(int _C); 
#line 66
int __cdecl tolower(int _C); 
#line 68
int __cdecl _tolower(int _C); 
#line 69
int __cdecl _tolower_l(int _C, _locale_t _Locale); 
#line 70
int __cdecl _toupper(int _C); 
#line 71
int __cdecl _toupper_l(int _C, _locale_t _Locale); 
#line 73
int __cdecl __isascii(int _C); 
#line 74
int __cdecl __toascii(int _C); 
#line 75
int __cdecl __iscsymf(int _C); 
#line 76
int __cdecl __iscsym(int _C); 
#line 85
__inline int __cdecl __acrt_locale_get_ctype_array_value(const unsigned short *const 
#line 86
_Locale_pctype_array, const int 
#line 87
_Char_value, const int 
#line 88
_Mask) 
#line 90
{ 
#line 96
if ((_Char_value >= (-1)) && (_Char_value <= 255)) 
#line 97
{ 
#line 98
return (_Locale_pctype_array[_Char_value]) & _Mask; 
#line 99
}  
#line 101
return 0; 
#line 102
} 
#line 124 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
int __cdecl ___mb_cur_max_func(); 
#line 126
int __cdecl ___mb_cur_max_l_func(_locale_t _Locale); 
#line 152 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
__forceinline int __cdecl __ascii_tolower(const int _C) 
#line 153
{ 
#line 154
if ((_C >= ('A')) && (_C <= ('Z'))) 
#line 155
{ 
#line 156
return _C - (('A') - ('a')); 
#line 157
}  
#line 158
return _C; 
#line 159
} 
#line 161
__forceinline int __cdecl __ascii_toupper(const int _C) 
#line 162
{ 
#line 163
if ((_C >= ('a')) && (_C <= ('z'))) 
#line 164
{ 
#line 165
return _C - (('a') - ('A')); 
#line 166
}  
#line 167
return _C; 
#line 168
} 
#line 170
__forceinline int __cdecl __ascii_iswalpha(const int _C) 
#line 171
{ 
#line 172
return ((_C >= ('A')) && (_C <= ('Z'))) || ((_C >= ('a')) && (_C <= ('z'))); 
#line 173
} 
#line 175
__forceinline int __cdecl __ascii_iswdigit(const int _C) 
#line 176
{ 
#line 177
return (_C >= ('0')) && (_C <= ('9')); 
#line 178
} 
#line 180
__forceinline int __cdecl __ascii_towlower(const int _C) 
#line 181
{ 
#line 182
return __ascii_tolower(_C); 
#line 183
} 
#line 185
__forceinline int __cdecl __ascii_towupper(const int _C) 
#line 186
{ 
#line 187
return __ascii_toupper(_C); 
#line 188
} 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
__inline __crt_locale_data_public *__cdecl __acrt_get_locale_data_prefix(const volatile void *const _LocalePointers) 
#line 209
{ 
#line 210
const _locale_t _TypedLocalePointers = (_locale_t)_LocalePointers; 
#line 211
return (__crt_locale_data_public *)(_TypedLocalePointers->locinfo); 
#line 212
} 
#line 218
__inline int __cdecl _chvalidchk_l(const int 
#line 219
_C, const int 
#line 220
_Mask, const _locale_t 
#line 221
_Locale) 
#line 223
{ 
#line 227
if (!_Locale) 
#line 228
{ 
#line 229
return __acrt_locale_get_ctype_array_value(__pctype_func(), _C, _Mask); 
#line 230
}  
#line 232
return __acrt_locale_get_ctype_array_value(__acrt_get_locale_data_prefix(_Locale)->_locale_pctype, _C, _Mask); 
#line 234 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
} 
#line 239
__inline int __cdecl _ischartype_l(const int 
#line 240
_C, const int 
#line 241
_Mask, const _locale_t 
#line 242
_Locale) 
#line 244
{ 
#line 245
if (!_Locale) 
#line 246
{ 
#line 247
return _chvalidchk_l(_C, _Mask, 0); 
#line 248
}  
#line 250
if ((_C >= (-1)) && (_C <= 255)) 
#line 251
{ 
#line 252
return ((__acrt_get_locale_data_prefix(_Locale)->_locale_pctype)[_C]) & _Mask; 
#line 253
}  
#line 255
if ((__acrt_get_locale_data_prefix(_Locale)->_locale_mb_cur_max) > 1) 
#line 256
{ 
#line 257
return _isctype_l(_C, _Mask, _Locale); 
#line 258
}  
#line 260
return 0; 
#line 261
} 
#line 307 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\ctype.h"
}__pragma( pack ( pop )) 
#line 309
#pragma warning(pop)
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\device_types.h"
#if 0
#line 68
enum cudaRoundMode { 
#line 70
cudaRoundNearest, 
#line 71
cudaRoundZero, 
#line 72
cudaRoundPosInf, 
#line 73
cudaRoundMinInf
#line 74
}; 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 173
struct char1 { 
#line 175
signed char x; 
#line 176
}; 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 178
struct uchar1 { 
#line 180
unsigned char x; 
#line 181
}; 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 184
struct __declspec(align(2)) char2 { 
#line 186
signed char x, y; 
#line 187
}; 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 189
struct __declspec(align(2)) uchar2 { 
#line 191
unsigned char x, y; 
#line 192
}; 
#endif
#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 194
struct char3 { 
#line 196
signed char x, y, z; 
#line 197
}; 
#endif
#line 199 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 199
struct uchar3 { 
#line 201
unsigned char x, y, z; 
#line 202
}; 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 204
struct __declspec(align(4)) char4 { 
#line 206
signed char x, y, z, w; 
#line 207
}; 
#endif
#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 209
struct __declspec(align(4)) uchar4 { 
#line 211
unsigned char x, y, z, w; 
#line 212
}; 
#endif
#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 214
struct short1 { 
#line 216
short x; 
#line 217
}; 
#endif
#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 219
struct ushort1 { 
#line 221
unsigned short x; 
#line 222
}; 
#endif
#line 224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 224
struct __declspec(align(4)) short2 { 
#line 226
short x, y; 
#line 227
}; 
#endif
#line 229 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 229
struct __declspec(align(4)) ushort2 { 
#line 231
unsigned short x, y; 
#line 232
}; 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 234
struct short3 { 
#line 236
short x, y, z; 
#line 237
}; 
#endif
#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 239
struct ushort3 { 
#line 241
unsigned short x, y, z; 
#line 242
}; 
#endif
#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 244
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 245
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 247
struct int1 { 
#line 249
int x; 
#line 250
}; 
#endif
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 252
struct uint1 { 
#line 254
unsigned x; 
#line 255
}; 
#endif
#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 257
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 258
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 260
struct int3 { 
#line 262
int x, y, z; 
#line 263
}; 
#endif
#line 265 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 265
struct uint3 { 
#line 267
unsigned x, y, z; 
#line 268
}; 
#endif
#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 270
struct __declspec(align(16)) int4 { 
#line 272
int x, y, z, w; 
#line 273
}; 
#endif
#line 275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 275
struct __declspec(align(16)) uint4 { 
#line 277
unsigned x, y, z, w; 
#line 278
}; 
#endif
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 280
struct long1 { 
#line 282
long x; 
#line 283
}; 
#endif
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 285
struct ulong1 { 
#line 287
unsigned long x; 
#line 288
}; 
#endif
#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 291
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 292
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 307
struct long3 { 
#line 309
long x, y, z; 
#line 310
}; 
#endif
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 312
struct ulong3 { 
#line 314
unsigned long x, y, z; 
#line 315
}; 
#endif
#line 318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 318
struct
#line 317
 __declspec(deprecated("use long4_16a or long4_32a"))
#line 318
 __declspec(align(16)) long4 { 
#line 320
long x, y, z, w; 
#line 321
}; 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 324
struct
#line 323
 __declspec(deprecated("use ulong4_16a or ulong4_32a"))
#line 324
 __declspec(align(16)) ulong4 { 
#line 326
unsigned long x, y, z, w; 
#line 327
}; 
#endif
#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 329
struct __declspec(align(16)) long4_16a { 
#line 331
long x, y, z, w; 
#line 332
}; 
#endif
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 334
struct __declspec(align(16)) ulong4_16a { 
#line 336
unsigned long x, y, z, w; 
#line 337
}; 
#endif
#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#pragma warning(push)
#pragma warning(disable: 4324)
#line 343 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 343
struct __declspec(align(32)) long4_32a { 
#line 345
long x, y, z, w; 
#line 346
}; 
#endif
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 348
struct __declspec(align(32)) ulong4_32a { 
#line 350
unsigned long x, y, z, w; 
#line 351
}; 
#endif
#line 353 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#pragma warning(pop)
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 356
struct float1 { 
#line 358
float x; 
#line 359
}; 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 378
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 383
struct float3 { 
#line 385
float x, y, z; 
#line 386
}; 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 388
struct __declspec(align(16)) float4 { 
#line 390
float x, y, z, w; 
#line 391
}; 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 393
struct longlong1 { 
#line 395
__int64 x; 
#line 396
}; 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 398
struct ulonglong1 { 
#line 400
unsigned __int64 x; 
#line 401
}; 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 403
struct __declspec(align(16)) longlong2 { 
#line 405
__int64 x, y; 
#line 406
}; 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 408
struct __declspec(align(16)) ulonglong2 { 
#line 410
unsigned __int64 x, y; 
#line 411
}; 
#endif
#line 413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 413
struct longlong3 { 
#line 415
__int64 x, y, z; 
#line 416
}; 
#endif
#line 418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 418
struct ulonglong3 { 
#line 420
unsigned __int64 x, y, z; 
#line 421
}; 
#endif
#line 424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 424
struct
#line 423
 __declspec(deprecated("use longlong4_16a or longlong4_32a"))
#line 424
 __declspec(align(16)) longlong4 { 
#line 426
__int64 x, y, z, w; 
#line 427
}; 
#endif
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 429
struct __declspec(align(16)) longlong4_16a { 
#line 431
__int64 x, y, z, w; 
#line 432
}; 
#endif
#line 434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 434
struct __declspec(align(32)) longlong4_32a { 
#line 436
__int64 x, y, z, w; 
#line 437
}; 
#endif
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 440
struct
#line 439
 __declspec(deprecated("use ulonglong4_16a or ulonglong4_32a"))
#line 440
 __declspec(align(16)) ulonglong4 { 
#line 442
unsigned __int64 x, y, z, w; 
#line 443
}; 
#endif
#line 445 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 445
struct __declspec(align(16)) ulonglong4_16a { 
#line 447
unsigned __int64 x, y, z, w; 
#line 448
}; 
#endif
#line 450 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 450
struct __declspec(align(32)) ulonglong4_32a { 
#line 452
unsigned __int64 x, y, z, w; 
#line 453
}; 
#endif
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 456
struct double1 { 
#line 458
double x; 
#line 459
}; 
#endif
#line 461 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 461
struct __declspec(align(16)) double2 { 
#line 463
double x, y; 
#line 464
}; 
#endif
#line 466 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 466
struct double3 { 
#line 468
double x, y, z; 
#line 469
}; 
#endif
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 472
struct
#line 471
 __declspec(deprecated("use double4_16a or double4_32a"))
#line 472
 __declspec(align(16)) double4 { 
#line 474
double x, y, z, w; 
#line 475
}; 
#endif
#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 477
struct __declspec(align(16)) double4_16a { 
#line 479
double x, y, z, w; 
#line 480
}; 
#endif
#line 482 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 482
struct __declspec(align(32)) double4_32a { 
#line 484
double x, y, z, w; 
#line 485
}; 
#endif
#line 499 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef char1 
#line 499
char1; 
#endif
#line 500 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 500
uchar1; 
#endif
#line 501 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef char2 
#line 501
char2; 
#endif
#line 502 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 502
uchar2; 
#endif
#line 503 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef char3 
#line 503
char3; 
#endif
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 504
uchar3; 
#endif
#line 505 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef char4 
#line 505
char4; 
#endif
#line 506 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 506
uchar4; 
#endif
#line 507 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef short1 
#line 507
short1; 
#endif
#line 508 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 508
ushort1; 
#endif
#line 509 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef short2 
#line 509
short2; 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 510
ushort2; 
#endif
#line 511 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef short3 
#line 511
short3; 
#endif
#line 512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 512
ushort3; 
#endif
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef short4 
#line 513
short4; 
#endif
#line 514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 514
ushort4; 
#endif
#line 515 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef int1 
#line 515
int1; 
#endif
#line 516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uint1 
#line 516
uint1; 
#endif
#line 517 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef int2 
#line 517
int2; 
#endif
#line 518 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uint2 
#line 518
uint2; 
#endif
#line 519 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef int3 
#line 519
int3; 
#endif
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uint3 
#line 520
uint3; 
#endif
#line 521 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef int4 
#line 521
int4; 
#endif
#line 522 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef uint4 
#line 522
uint4; 
#endif
#line 523 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef long1 
#line 523
long1; 
#endif
#line 524 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 524
ulong1; 
#endif
#line 525 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef long2 
#line 525
long2; 
#endif
#line 526 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 526
ulong2; 
#endif
#line 527 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef long3 
#line 527
long3; 
#endif
#line 528 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 528
ulong3; 
#endif
#line 529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#if 0
#line 530
__declspec(deprecated("use long4_16a or long4_32a")) typedef long4 long4; 
#endif
#line 531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 531
__declspec(deprecated("use ulong4_16a or ulong4_32a")) typedef ulong4 ulong4; 
#endif
#line 532 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(pop)) 
#if 0
typedef long4_16a 
#line 533
long4_16a; 
#endif
#line 534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulong4_16a 
#line 534
ulong4_16a; 
#endif
#line 535 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef long4_32a 
#line 535
long4_32a; 
#endif
#line 536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulong4_32a 
#line 536
ulong4_32a; 
#endif
#line 537 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef float1 
#line 537
float1; 
#endif
#line 538 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef float2 
#line 538
float2; 
#endif
#line 539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef float3 
#line 539
float3; 
#endif
#line 540 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef float4 
#line 540
float4; 
#endif
#line 541 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 541
longlong1; 
#endif
#line 542 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 542
ulonglong1; 
#endif
#line 543 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 543
longlong2; 
#endif
#line 544 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 544
ulonglong2; 
#endif
#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 545
longlong3; 
#endif
#line 546 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 546
ulonglong3; 
#endif
#line 547 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#if 0
#line 548
__declspec(deprecated("use longlong4_16a or longlong4_32a")) typedef longlong4 longlong4; 
#endif
#line 549 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 549
__declspec(deprecated("use ulonglong4_16a or ulonglong4_32a")) typedef ulonglong4 ulonglong4; 
#endif
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(pop)) 
#if 0
typedef longlong4_16a 
#line 551
longlong4_16a; 
#endif
#line 552 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulonglong4_16a 
#line 552
ulonglong4_16a; 
#endif
#line 553 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef longlong4_32a 
#line 553
longlong4_32a; 
#endif
#line 554 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef ulonglong4_32a 
#line 554
ulonglong4_32a; 
#endif
#line 555 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef double1 
#line 555
double1; 
#endif
#line 556 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef double2 
#line 556
double2; 
#endif
#line 557 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef double3 
#line 557
double3; 
#endif
#line 558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#if 0
#line 559
__declspec(deprecated("use double4_16a or double4_32a")) typedef double4 double4; 
#endif
#line 560 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
__pragma( warning(pop)) 
#if 0
typedef double4_16a 
#line 561
double4_16a; 
#endif
#line 562 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef double4_32a 
#line 562
double4_32a; 
#endif
#line 574 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
#line 574
struct dim3 { 
#line 576
unsigned x, y, z; 
#line 591 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
}; 
#endif
#line 593 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_types.h"
#if 0
typedef dim3 
#line 593
dim3; 
#endif
#line 13 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\limits.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 16
__pragma( pack ( push, 8 )) extern "C" {
#line 74 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 76
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stddef.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
namespace std { 
#line 25
typedef decltype(nullptr) nullptr_t; 
#line 26
}
#line 28
using std::nullptr_t;
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stddef.h"
int *__cdecl _errno(); 
#line 38
errno_t __cdecl _set_errno(int _Value); 
#line 39
errno_t __cdecl _get_errno(int * _Value); 
#line 55 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stddef.h"
extern unsigned long __cdecl __threadid(); 
#line 57
extern uintptr_t __cdecl __threadhandle(); 
#line 61
}__pragma( pack ( pop )) 
#line 63
#pragma warning(pop)
#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 192
enum cudaError { 
#line 199
cudaSuccess, 
#line 205
cudaErrorInvalidValue, 
#line 211
cudaErrorMemoryAllocation, 
#line 217
cudaErrorInitializationError, 
#line 224
cudaErrorCudartUnloading, 
#line 231
cudaErrorProfilerDisabled, 
#line 239
cudaErrorProfilerNotInitialized, 
#line 246
cudaErrorProfilerAlreadyStarted, 
#line 253
cudaErrorProfilerAlreadyStopped, 
#line 261
cudaErrorInvalidConfiguration, 
#line 267
cudaErrorInvalidPitchValue = 12, 
#line 273
cudaErrorInvalidSymbol, 
#line 281
cudaErrorInvalidHostPointer = 16, 
#line 289
cudaErrorInvalidDevicePointer, 
#line 294
cudaErrorInvalidTexture, 
#line 300
cudaErrorInvalidTextureBinding, 
#line 307
cudaErrorInvalidChannelDescriptor, 
#line 313
cudaErrorInvalidMemcpyDirection, 
#line 323
cudaErrorAddressOfConstant, 
#line 332
cudaErrorTextureFetchFailed, 
#line 341
cudaErrorTextureNotBound, 
#line 350
cudaErrorSynchronizationError, 
#line 355
cudaErrorInvalidFilterSetting, 
#line 361
cudaErrorInvalidNormSetting, 
#line 369
cudaErrorMixedDeviceExecution, 
#line 377
cudaErrorNotYetImplemented = 31, 
#line 386
cudaErrorMemoryValueTooLarge, 
#line 392
cudaErrorStubLibrary = 34, 
#line 399
cudaErrorInsufficientDriver, 
#line 406
cudaErrorCallRequiresNewerDriver, 
#line 412
cudaErrorInvalidSurface, 
#line 418
cudaErrorDuplicateVariableName = 43, 
#line 424
cudaErrorDuplicateTextureName, 
#line 430
cudaErrorDuplicateSurfaceName, 
#line 440
cudaErrorDevicesUnavailable, 
#line 453
cudaErrorIncompatibleDriverContext = 49, 
#line 459
cudaErrorMissingConfiguration = 52, 
#line 468
cudaErrorPriorLaunchFailure, 
#line 474
cudaErrorLaunchMaxDepthExceeded = 65, 
#line 482
cudaErrorLaunchFileScopedTex, 
#line 490
cudaErrorLaunchFileScopedSurf, 
#line 506
cudaErrorSyncDepthExceeded, 
#line 518
cudaErrorLaunchPendingCountExceeded, 
#line 524
cudaErrorInvalidDeviceFunction = 98, 
#line 530
cudaErrorNoDevice = 100, 
#line 537
cudaErrorInvalidDevice, 
#line 542
cudaErrorDeviceNotLicensed, 
#line 551
cudaErrorSoftwareValidityNotEstablished, 
#line 556
cudaErrorStartupFailure = 127, 
#line 561
cudaErrorInvalidKernelImage = 200, 
#line 571
cudaErrorDeviceUninitialized, 
#line 576
cudaErrorMapBufferObjectFailed = 205, 
#line 581
cudaErrorUnmapBufferObjectFailed, 
#line 587
cudaErrorArrayIsMapped, 
#line 592
cudaErrorAlreadyMapped, 
#line 600
cudaErrorNoKernelImageForDevice, 
#line 605
cudaErrorAlreadyAcquired, 
#line 610
cudaErrorNotMapped, 
#line 616
cudaErrorNotMappedAsArray, 
#line 622
cudaErrorNotMappedAsPointer, 
#line 628
cudaErrorECCUncorrectable, 
#line 634
cudaErrorUnsupportedLimit, 
#line 640
cudaErrorDeviceAlreadyInUse, 
#line 646
cudaErrorPeerAccessUnsupported, 
#line 652
cudaErrorInvalidPtx, 
#line 657
cudaErrorInvalidGraphicsContext, 
#line 663
cudaErrorNvlinkUncorrectable, 
#line 670
cudaErrorJitCompilerNotFound, 
#line 677
cudaErrorUnsupportedPtxVersion, 
#line 684
cudaErrorJitCompilationDisabled, 
#line 689
cudaErrorUnsupportedExecAffinity, 
#line 695
cudaErrorUnsupportedDevSideSync, 
#line 706
cudaErrorContained, 
#line 711
cudaErrorInvalidSource = 300, 
#line 716
cudaErrorFileNotFound, 
#line 721
cudaErrorSharedObjectSymbolNotFound, 
#line 726
cudaErrorSharedObjectInitFailed, 
#line 731
cudaErrorOperatingSystem, 
#line 738
cudaErrorInvalidResourceHandle = 400, 
#line 744
cudaErrorIllegalState, 
#line 752
cudaErrorLossyQuery, 
#line 759
cudaErrorSymbolNotFound = 500, 
#line 767
cudaErrorNotReady = 600, 
#line 775
cudaErrorIllegalAddress = 700, 
#line 784
cudaErrorLaunchOutOfResources, 
#line 795
cudaErrorLaunchTimeout, 
#line 801
cudaErrorLaunchIncompatibleTexturing, 
#line 808
cudaErrorPeerAccessAlreadyEnabled, 
#line 815
cudaErrorPeerAccessNotEnabled, 
#line 828
cudaErrorSetOnActiveProcess = 708, 
#line 835
cudaErrorContextIsDestroyed, 
#line 842
cudaErrorAssert, 
#line 849
cudaErrorTooManyPeers, 
#line 855
cudaErrorHostMemoryAlreadyRegistered, 
#line 861
cudaErrorHostMemoryNotRegistered, 
#line 870
cudaErrorHardwareStackError, 
#line 878
cudaErrorIllegalInstruction, 
#line 887
cudaErrorMisalignedAddress, 
#line 898
cudaErrorInvalidAddressSpace, 
#line 906
cudaErrorInvalidPc, 
#line 917
cudaErrorLaunchFailure, 
#line 926
cudaErrorCooperativeLaunchTooLarge, 
#line 934
cudaErrorTensorMemoryLeak, 
#line 939
cudaErrorNotPermitted = 800, 
#line 945
cudaErrorNotSupported, 
#line 954
cudaErrorSystemNotReady, 
#line 961
cudaErrorSystemDriverMismatch, 
#line 970
cudaErrorCompatNotSupportedOnDevice, 
#line 975
cudaErrorMpsConnectionFailed, 
#line 980
cudaErrorMpsRpcFailure, 
#line 986
cudaErrorMpsServerNotReady, 
#line 991
cudaErrorMpsMaxClientsReached, 
#line 996
cudaErrorMpsMaxConnectionsReached, 
#line 1001
cudaErrorMpsClientTerminated, 
#line 1006
cudaErrorCdpNotSupported, 
#line 1011
cudaErrorCdpVersionMismatch, 
#line 1016
cudaErrorStreamCaptureUnsupported = 900, 
#line 1022
cudaErrorStreamCaptureInvalidated, 
#line 1028
cudaErrorStreamCaptureMerge, 
#line 1033
cudaErrorStreamCaptureUnmatched, 
#line 1039
cudaErrorStreamCaptureUnjoined, 
#line 1046
cudaErrorStreamCaptureIsolation, 
#line 1052
cudaErrorStreamCaptureImplicit, 
#line 1058
cudaErrorCapturedEvent, 
#line 1065
cudaErrorStreamCaptureWrongThread, 
#line 1070
cudaErrorTimeout, 
#line 1076
cudaErrorGraphExecUpdateFailure, 
#line 1086
cudaErrorExternalDevice, 
#line 1092
cudaErrorInvalidClusterSize, 
#line 1098
cudaErrorFunctionNotLoaded, 
#line 1104
cudaErrorInvalidResourceType, 
#line 1110
cudaErrorInvalidResourceConfiguration, 
#line 1115
cudaErrorUnknown = 999, 
#line 1123
cudaErrorApiFailureBase = 10000
#line 1124
}; 
#endif
#line 1129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1129
enum cudaChannelFormatKind { 
#line 1131
cudaChannelFormatKindSigned, 
#line 1132
cudaChannelFormatKindUnsigned, 
#line 1133
cudaChannelFormatKindFloat, 
#line 1134
cudaChannelFormatKindNone, 
#line 1135
cudaChannelFormatKindNV12, 
#line 1136
cudaChannelFormatKindUnsignedNormalized8X1, 
#line 1137
cudaChannelFormatKindUnsignedNormalized8X2, 
#line 1138
cudaChannelFormatKindUnsignedNormalized8X4, 
#line 1139
cudaChannelFormatKindUnsignedNormalized16X1, 
#line 1140
cudaChannelFormatKindUnsignedNormalized16X2, 
#line 1141
cudaChannelFormatKindUnsignedNormalized16X4, 
#line 1142
cudaChannelFormatKindSignedNormalized8X1, 
#line 1143
cudaChannelFormatKindSignedNormalized8X2, 
#line 1144
cudaChannelFormatKindSignedNormalized8X4, 
#line 1145
cudaChannelFormatKindSignedNormalized16X1, 
#line 1146
cudaChannelFormatKindSignedNormalized16X2, 
#line 1147
cudaChannelFormatKindSignedNormalized16X4, 
#line 1148
cudaChannelFormatKindUnsignedBlockCompressed1, 
#line 1149
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
#line 1150
cudaChannelFormatKindUnsignedBlockCompressed2, 
#line 1151
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
#line 1152
cudaChannelFormatKindUnsignedBlockCompressed3, 
#line 1153
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
#line 1154
cudaChannelFormatKindUnsignedBlockCompressed4, 
#line 1155
cudaChannelFormatKindSignedBlockCompressed4, 
#line 1156
cudaChannelFormatKindUnsignedBlockCompressed5, 
#line 1157
cudaChannelFormatKindSignedBlockCompressed5, 
#line 1158
cudaChannelFormatKindUnsignedBlockCompressed6H, 
#line 1159
cudaChannelFormatKindSignedBlockCompressed6H, 
#line 1160
cudaChannelFormatKindUnsignedBlockCompressed7, 
#line 1161
cudaChannelFormatKindUnsignedBlockCompressed7SRGB, 
#line 1162
cudaChannelFormatKindUnsignedNormalized1010102
#line 1164
}; 
#endif
#line 1169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1169
struct cudaChannelFormatDesc { 
#line 1171
int x; 
#line 1172
int y; 
#line 1173
int z; 
#line 1174
int w; 
#line 1175
cudaChannelFormatKind f; 
#line 1176
}; 
#endif
#line 1181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 1186
typedef const cudaArray *cudaArray_const_t; 
#line 1188
struct cudaArray; 
#line 1193
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 1198
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 1200
struct cudaMipmappedArray; 
#line 1210
#if 0
#line 1210
struct cudaArraySparseProperties { 
#line 1211
struct { 
#line 1212
unsigned width; 
#line 1213
unsigned height; 
#line 1214
unsigned depth; 
#line 1215
} tileExtent; 
#line 1216
unsigned miptailFirstLevel; 
#line 1217
unsigned __int64 miptailSize; 
#line 1218
unsigned flags; 
#line 1219
unsigned reserved[4]; 
#line 1220
}; 
#endif
#line 1225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1225
struct cudaArrayMemoryRequirements { 
#line 1226
size_t size; 
#line 1227
size_t alignment; 
#line 1228
unsigned reserved[4]; 
#line 1229
}; 
#endif
#line 1234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1234
enum cudaMemoryType { 
#line 1236
cudaMemoryTypeUnregistered, 
#line 1237
cudaMemoryTypeHost, 
#line 1238
cudaMemoryTypeDevice, 
#line 1239
cudaMemoryTypeManaged
#line 1240
}; 
#endif
#line 1245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1245
enum cudaMemcpyKind { 
#line 1247
cudaMemcpyHostToHost, 
#line 1248
cudaMemcpyHostToDevice, 
#line 1249
cudaMemcpyDeviceToHost, 
#line 1250
cudaMemcpyDeviceToDevice, 
#line 1251
cudaMemcpyDefault
#line 1252
}; 
#endif
#line 1259 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1259
struct cudaPitchedPtr { 
#line 1261
void *ptr; 
#line 1262
size_t pitch; 
#line 1263
size_t xsize; 
#line 1264
size_t ysize; 
#line 1265
}; 
#endif
#line 1272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1272
struct cudaExtent { 
#line 1274
size_t width; 
#line 1275
size_t height; 
#line 1276
size_t depth; 
#line 1277
}; 
#endif
#line 1284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1284
struct cudaPos { 
#line 1286
size_t x; 
#line 1287
size_t y; 
#line 1288
size_t z; 
#line 1289
}; 
#endif
#line 1294 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1294
struct cudaMemcpy3DParms { 
#line 1296
cudaArray_t srcArray; 
#line 1297
cudaPos srcPos; 
#line 1298
cudaPitchedPtr srcPtr; 
#line 1300
cudaArray_t dstArray; 
#line 1301
cudaPos dstPos; 
#line 1302
cudaPitchedPtr dstPtr; 
#line 1304
cudaExtent extent; 
#line 1305
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1306
}; 
#endif
#line 1311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1311
struct cudaMemcpyNodeParams { 
#line 1312
int flags; 
#line 1313
int reserved[3]; 
#line 1314
cudaMemcpy3DParms copyParams; 
#line 1315
}; 
#endif
#line 1320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1320
struct cudaMemcpy3DPeerParms { 
#line 1322
cudaArray_t srcArray; 
#line 1323
cudaPos srcPos; 
#line 1324
cudaPitchedPtr srcPtr; 
#line 1325
int srcDevice; 
#line 1327
cudaArray_t dstArray; 
#line 1328
cudaPos dstPos; 
#line 1329
cudaPitchedPtr dstPtr; 
#line 1330
int dstDevice; 
#line 1332
cudaExtent extent; 
#line 1333
}; 
#endif
#line 1338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1338
struct cudaMemsetParams { 
#line 1339
void *dst; 
#line 1340
size_t pitch; 
#line 1341
unsigned value; 
#line 1342
unsigned elementSize; 
#line 1343
size_t width; 
#line 1344
size_t height; 
#line 1345
}; 
#endif
#line 1350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1350
struct cudaMemsetParamsV2 { 
#line 1351
void *dst; 
#line 1352
size_t pitch; 
#line 1353
unsigned value; 
#line 1354
unsigned elementSize; 
#line 1355
size_t width; 
#line 1356
size_t height; 
#line 1357
}; 
#endif
#line 1362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1362
enum cudaAccessProperty { 
#line 1363
cudaAccessPropertyNormal, 
#line 1364
cudaAccessPropertyStreaming, 
#line 1365
cudaAccessPropertyPersisting
#line 1366
}; 
#endif
#line 1379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1379
struct cudaAccessPolicyWindow { 
#line 1380
void *base_ptr; 
#line 1381
size_t num_bytes; 
#line 1382
float hitRatio; 
#line 1383
cudaAccessProperty hitProp; 
#line 1384
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1385
}; 
#endif
#line 1397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef void (__stdcall *cudaHostFn_t)(void * userData); 
#line 1402
#if 0
#line 1402
struct cudaHostNodeParams { 
#line 1403
cudaHostFn_t fn; 
#line 1404
void *userData; 
#line 1405
}; 
#endif
#line 1410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1410
struct cudaHostNodeParamsV2 { 
#line 1411
cudaHostFn_t fn; 
#line 1412
void *userData; 
#line 1413
}; 
#endif
#line 1418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1418
enum cudaStreamCaptureStatus { 
#line 1419
cudaStreamCaptureStatusNone, 
#line 1420
cudaStreamCaptureStatusActive, 
#line 1421
cudaStreamCaptureStatusInvalidated
#line 1423
}; 
#endif
#line 1429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1429
enum cudaStreamCaptureMode { 
#line 1430
cudaStreamCaptureModeGlobal, 
#line 1431
cudaStreamCaptureModeThreadLocal, 
#line 1432
cudaStreamCaptureModeRelaxed
#line 1433
}; 
#endif
#line 1435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1435
enum cudaSynchronizationPolicy { 
#line 1436
cudaSyncPolicyAuto = 1, 
#line 1437
cudaSyncPolicySpin, 
#line 1438
cudaSyncPolicyYield, 
#line 1439
cudaSyncPolicyBlockingSync
#line 1440
}; 
#endif
#line 1445 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1445
enum cudaClusterSchedulingPolicy { 
#line 1446
cudaClusterSchedulingPolicyDefault, 
#line 1447
cudaClusterSchedulingPolicySpread, 
#line 1448
cudaClusterSchedulingPolicyLoadBalancing
#line 1449
}; 
#endif
#line 1454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1454
enum cudaStreamUpdateCaptureDependenciesFlags { 
#line 1455
cudaStreamAddCaptureDependencies, 
#line 1456
cudaStreamSetCaptureDependencies
#line 1457
}; 
#endif
#line 1462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1462
enum cudaUserObjectFlags { 
#line 1463
cudaUserObjectNoDestructorSync = 1
#line 1464
}; 
#endif
#line 1469 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1469
enum cudaUserObjectRetainFlags { 
#line 1470
cudaGraphUserObjectMove = 1
#line 1471
}; 
#endif
#line 1476 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 1481
#if 0
#line 1481
enum cudaGraphicsRegisterFlags { 
#line 1483
cudaGraphicsRegisterFlagsNone, 
#line 1484
cudaGraphicsRegisterFlagsReadOnly, 
#line 1485
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 1486
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 1487
cudaGraphicsRegisterFlagsTextureGather = 8
#line 1488
}; 
#endif
#line 1493 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1493
enum cudaGraphicsMapFlags { 
#line 1495
cudaGraphicsMapFlagsNone, 
#line 1496
cudaGraphicsMapFlagsReadOnly, 
#line 1497
cudaGraphicsMapFlagsWriteDiscard
#line 1498
}; 
#endif
#line 1503 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1503
enum cudaGraphicsCubeFace { 
#line 1505
cudaGraphicsCubeFacePositiveX, 
#line 1506
cudaGraphicsCubeFaceNegativeX, 
#line 1507
cudaGraphicsCubeFacePositiveY, 
#line 1508
cudaGraphicsCubeFaceNegativeY, 
#line 1509
cudaGraphicsCubeFacePositiveZ, 
#line 1510
cudaGraphicsCubeFaceNegativeZ
#line 1511
}; 
#endif
#line 1516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1516
enum cudaResourceType { 
#line 1518
cudaResourceTypeArray, 
#line 1519
cudaResourceTypeMipmappedArray, 
#line 1520
cudaResourceTypeLinear, 
#line 1521
cudaResourceTypePitch2D
#line 1522
}; 
#endif
#line 1527 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1527
enum cudaResourceViewFormat { 
#line 1529
cudaResViewFormatNone, 
#line 1530
cudaResViewFormatUnsignedChar1, 
#line 1531
cudaResViewFormatUnsignedChar2, 
#line 1532
cudaResViewFormatUnsignedChar4, 
#line 1533
cudaResViewFormatSignedChar1, 
#line 1534
cudaResViewFormatSignedChar2, 
#line 1535
cudaResViewFormatSignedChar4, 
#line 1536
cudaResViewFormatUnsignedShort1, 
#line 1537
cudaResViewFormatUnsignedShort2, 
#line 1538
cudaResViewFormatUnsignedShort4, 
#line 1539
cudaResViewFormatSignedShort1, 
#line 1540
cudaResViewFormatSignedShort2, 
#line 1541
cudaResViewFormatSignedShort4, 
#line 1542
cudaResViewFormatUnsignedInt1, 
#line 1543
cudaResViewFormatUnsignedInt2, 
#line 1544
cudaResViewFormatUnsignedInt4, 
#line 1545
cudaResViewFormatSignedInt1, 
#line 1546
cudaResViewFormatSignedInt2, 
#line 1547
cudaResViewFormatSignedInt4, 
#line 1548
cudaResViewFormatHalf1, 
#line 1549
cudaResViewFormatHalf2, 
#line 1550
cudaResViewFormatHalf4, 
#line 1551
cudaResViewFormatFloat1, 
#line 1552
cudaResViewFormatFloat2, 
#line 1553
cudaResViewFormatFloat4, 
#line 1554
cudaResViewFormatUnsignedBlockCompressed1, 
#line 1555
cudaResViewFormatUnsignedBlockCompressed2, 
#line 1556
cudaResViewFormatUnsignedBlockCompressed3, 
#line 1557
cudaResViewFormatUnsignedBlockCompressed4, 
#line 1558
cudaResViewFormatSignedBlockCompressed4, 
#line 1559
cudaResViewFormatUnsignedBlockCompressed5, 
#line 1560
cudaResViewFormatSignedBlockCompressed5, 
#line 1561
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 1562
cudaResViewFormatSignedBlockCompressed6H, 
#line 1563
cudaResViewFormatUnsignedBlockCompressed7
#line 1564
}; 
#endif
#line 1569 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1569
struct cudaResourceDesc { 
#line 1570
cudaResourceType resType; 
#line 1572
union { 
#line 1573
struct { 
#line 1574
cudaArray_t array; 
#line 1575
} array; 
#line 1576
struct { 
#line 1577
cudaMipmappedArray_t mipmap; 
#line 1578
} mipmap; 
#line 1579
struct { 
#line 1580
void *devPtr; 
#line 1581
cudaChannelFormatDesc desc; 
#line 1582
size_t sizeInBytes; 
#line 1583
} linear; 
#line 1584
struct { 
#line 1585
void *devPtr; 
#line 1586
cudaChannelFormatDesc desc; 
#line 1587
size_t width; 
#line 1588
size_t height; 
#line 1589
size_t pitchInBytes; 
#line 1590
} pitch2D; 
#line 1591
struct { 
#line 1592
int reserved[32]; 
#line 1593
} reserved; 
#line 1594
} res; 
#line 1596
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1597
}; 
#endif
#line 1602 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1602
struct cudaResourceViewDesc { 
#line 1604
cudaResourceViewFormat format; 
#line 1605
size_t width; 
#line 1606
size_t height; 
#line 1607
size_t depth; 
#line 1608
unsigned firstMipmapLevel; 
#line 1609
unsigned lastMipmapLevel; 
#line 1610
unsigned firstLayer; 
#line 1611
unsigned lastLayer; 
#line 1612
unsigned reserved[16]; 
#line 1613
}; 
#endif
#line 1618 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1618
struct cudaPointerAttributes { 
#line 1624
cudaMemoryType type; 
#line 1635
int device; 
#line 1641
void *devicePointer; 
#line 1650
void *hostPointer; 
#line 1655
long reserved[8]; 
#line 1656
}; 
#endif
#line 1661 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1661
struct cudaFuncAttributes { 
#line 1668
size_t sharedSizeBytes; 
#line 1674
size_t constSizeBytes; 
#line 1679
size_t localSizeBytes; 
#line 1686
int maxThreadsPerBlock; 
#line 1691
int numRegs; 
#line 1698
int ptxVersion; 
#line 1705
int binaryVersion; 
#line 1711
int cacheModeCA; 
#line 1718
int maxDynamicSharedSizeBytes; 
#line 1727
int preferredShmemCarveout; 
#line 1733
int clusterDimMustBeSet; 
#line 1744
int requiredClusterWidth; 
#line 1745
int requiredClusterHeight; 
#line 1746
int requiredClusterDepth; 
#line 1752
int clusterSchedulingPolicyPreference; 
#line 1774
int nonPortableClusterSizeAllowed; 
#line 1779
int reserved[16]; 
#line 1780
}; 
#endif
#line 1785 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1785
enum cudaFuncAttribute { 
#line 1787
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
#line 1788
cudaFuncAttributePreferredSharedMemoryCarveout, 
#line 1789
cudaFuncAttributeClusterDimMustBeSet, 
#line 1790
cudaFuncAttributeRequiredClusterWidth, 
#line 1791
cudaFuncAttributeRequiredClusterHeight, 
#line 1792
cudaFuncAttributeRequiredClusterDepth, 
#line 1793
cudaFuncAttributeNonPortableClusterSizeAllowed, 
#line 1794
cudaFuncAttributeClusterSchedulingPolicyPreference, 
#line 1795
cudaFuncAttributeMax
#line 1796
}; 
#endif
#line 1801 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1801
enum cudaFuncCache { 
#line 1803
cudaFuncCachePreferNone, 
#line 1804
cudaFuncCachePreferShared, 
#line 1805
cudaFuncCachePreferL1, 
#line 1806
cudaFuncCachePreferEqual
#line 1807
}; 
#endif
#line 1813 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1813
enum cudaSharedMemConfig { 
#line 1815
cudaSharedMemBankSizeDefault, 
#line 1816
cudaSharedMemBankSizeFourByte, 
#line 1817
cudaSharedMemBankSizeEightByte
#line 1818
}; 
#endif
#line 1823 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1823
enum cudaSharedCarveout { 
#line 1824
cudaSharedmemCarveoutDefault = (-1), 
#line 1825
cudaSharedmemCarveoutMaxShared = 100, 
#line 1826
cudaSharedmemCarveoutMaxL1 = 0
#line 1827
}; 
#endif
#line 1832 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1832
enum cudaComputeMode { 
#line 1834
cudaComputeModeDefault, 
#line 1835
cudaComputeModeExclusive, 
#line 1836
cudaComputeModeProhibited, 
#line 1837
cudaComputeModeExclusiveProcess
#line 1838
}; 
#endif
#line 1843 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1843
enum cudaLimit { 
#line 1845
cudaLimitStackSize, 
#line 1846
cudaLimitPrintfFifoSize, 
#line 1847
cudaLimitMallocHeapSize, 
#line 1848
cudaLimitDevRuntimeSyncDepth, 
#line 1849
cudaLimitDevRuntimePendingLaunchCount, 
#line 1850
cudaLimitMaxL2FetchGranularity, 
#line 1851
cudaLimitPersistingL2CacheSize
#line 1852
}; 
#endif
#line 1857 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1857
enum cudaMemoryAdvise { 
#line 1859
cudaMemAdviseSetReadMostly = 1, 
#line 1860
cudaMemAdviseUnsetReadMostly, 
#line 1861
cudaMemAdviseSetPreferredLocation, 
#line 1862
cudaMemAdviseUnsetPreferredLocation, 
#line 1863
cudaMemAdviseSetAccessedBy, 
#line 1864
cudaMemAdviseUnsetAccessedBy
#line 1865
}; 
#endif
#line 1870 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1870
enum cudaMemRangeAttribute { 
#line 1872
cudaMemRangeAttributeReadMostly = 1, 
#line 1873
cudaMemRangeAttributePreferredLocation, 
#line 1874
cudaMemRangeAttributeAccessedBy, 
#line 1875
cudaMemRangeAttributeLastPrefetchLocation, 
#line 1876
cudaMemRangeAttributePreferredLocationType, 
#line 1877
cudaMemRangeAttributePreferredLocationId, 
#line 1878
cudaMemRangeAttributeLastPrefetchLocationType, 
#line 1879
cudaMemRangeAttributeLastPrefetchLocationId
#line 1880
}; 
#endif
#line 1885 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1885
enum cudaFlushGPUDirectRDMAWritesOptions { 
#line 1886
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
#line 1887
cudaFlushGPUDirectRDMAWritesOptionMemOps
#line 1888
}; 
#endif
#line 1893 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1893
enum cudaGPUDirectRDMAWritesOrdering { 
#line 1894
cudaGPUDirectRDMAWritesOrderingNone, 
#line 1895
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
#line 1896
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
#line 1897
}; 
#endif
#line 1902 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1902
enum cudaFlushGPUDirectRDMAWritesScope { 
#line 1903
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
#line 1904
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
#line 1905
}; 
#endif
#line 1910 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1910
enum cudaFlushGPUDirectRDMAWritesTarget { 
#line 1911
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
#line 1912
}; 
#endif
#line 1918 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 1918
enum cudaDeviceAttr { 
#line 1920
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1921
cudaDevAttrMaxBlockDimX, 
#line 1922
cudaDevAttrMaxBlockDimY, 
#line 1923
cudaDevAttrMaxBlockDimZ, 
#line 1924
cudaDevAttrMaxGridDimX, 
#line 1925
cudaDevAttrMaxGridDimY, 
#line 1926
cudaDevAttrMaxGridDimZ, 
#line 1927
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1928
cudaDevAttrTotalConstantMemory, 
#line 1929
cudaDevAttrWarpSize, 
#line 1930
cudaDevAttrMaxPitch, 
#line 1931
cudaDevAttrMaxRegistersPerBlock, 
#line 1932
cudaDevAttrClockRate, 
#line 1933
cudaDevAttrTextureAlignment, 
#line 1934
cudaDevAttrGpuOverlap, 
#line 1935
cudaDevAttrMultiProcessorCount, 
#line 1936
cudaDevAttrKernelExecTimeout, 
#line 1937
cudaDevAttrIntegrated, 
#line 1938
cudaDevAttrCanMapHostMemory, 
#line 1939
cudaDevAttrComputeMode, 
#line 1940
cudaDevAttrMaxTexture1DWidth, 
#line 1941
cudaDevAttrMaxTexture2DWidth, 
#line 1942
cudaDevAttrMaxTexture2DHeight, 
#line 1943
cudaDevAttrMaxTexture3DWidth, 
#line 1944
cudaDevAttrMaxTexture3DHeight, 
#line 1945
cudaDevAttrMaxTexture3DDepth, 
#line 1946
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1947
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1948
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1949
cudaDevAttrSurfaceAlignment, 
#line 1950
cudaDevAttrConcurrentKernels, 
#line 1951
cudaDevAttrEccEnabled, 
#line 1952
cudaDevAttrPciBusId, 
#line 1953
cudaDevAttrPciDeviceId, 
#line 1954
cudaDevAttrTccDriver, 
#line 1955
cudaDevAttrMemoryClockRate, 
#line 1956
cudaDevAttrGlobalMemoryBusWidth, 
#line 1957
cudaDevAttrL2CacheSize, 
#line 1958
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1959
cudaDevAttrAsyncEngineCount, 
#line 1960
cudaDevAttrUnifiedAddressing, 
#line 1961
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1962
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1963
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1964
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1965
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1966
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1967
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1968
cudaDevAttrPciDomainId, 
#line 1969
cudaDevAttrTexturePitchAlignment, 
#line 1970
cudaDevAttrMaxTextureCubemapWidth, 
#line 1971
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1972
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1973
cudaDevAttrMaxSurface1DWidth, 
#line 1974
cudaDevAttrMaxSurface2DWidth, 
#line 1975
cudaDevAttrMaxSurface2DHeight, 
#line 1976
cudaDevAttrMaxSurface3DWidth, 
#line 1977
cudaDevAttrMaxSurface3DHeight, 
#line 1978
cudaDevAttrMaxSurface3DDepth, 
#line 1979
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1980
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1981
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1982
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1983
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1984
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1985
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1986
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1987
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1988
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1989
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1990
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1991
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1992
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1993
cudaDevAttrComputeCapabilityMajor, 
#line 1994
cudaDevAttrComputeCapabilityMinor, 
#line 1995
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1996
cudaDevAttrStreamPrioritiesSupported, 
#line 1997
cudaDevAttrGlobalL1CacheSupported, 
#line 1998
cudaDevAttrLocalL1CacheSupported, 
#line 1999
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 2000
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 2001
cudaDevAttrManagedMemory, 
#line 2002
cudaDevAttrIsMultiGpuBoard, 
#line 2003
cudaDevAttrMultiGpuBoardGroupID, 
#line 2004
cudaDevAttrHostNativeAtomicSupported, 
#line 2005
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 2006
cudaDevAttrPageableMemoryAccess, 
#line 2007
cudaDevAttrConcurrentManagedAccess, 
#line 2008
cudaDevAttrComputePreemptionSupported, 
#line 2009
cudaDevAttrCanUseHostPointerForRegisteredMem, 
#line 2010
cudaDevAttrReserved92, 
#line 2011
cudaDevAttrReserved93, 
#line 2012
cudaDevAttrReserved94, 
#line 2013
cudaDevAttrCooperativeLaunch, 
#line 2014
cudaDevAttrReserved96, 
#line 2015
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
#line 2016
cudaDevAttrCanFlushRemoteWrites, 
#line 2017
cudaDevAttrHostRegisterSupported, 
#line 2018
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
#line 2019
cudaDevAttrDirectManagedMemAccessFromHost, 
#line 2020
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
#line 2021
cudaDevAttrMaxPersistingL2CacheSize = 108, 
#line 2022
cudaDevAttrMaxAccessPolicyWindowSize, 
#line 2023
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
#line 2024
cudaDevAttrSparseCudaArraySupported, 
#line 2025
cudaDevAttrHostRegisterReadOnlySupported, 
#line 2026
cudaDevAttrTimelineSemaphoreInteropSupported, 
#line 2027
cudaDevAttrMemoryPoolsSupported, 
#line 2028
cudaDevAttrGPUDirectRDMASupported, 
#line 2029
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
#line 2030
cudaDevAttrGPUDirectRDMAWritesOrdering, 
#line 2031
cudaDevAttrMemoryPoolSupportedHandleTypes, 
#line 2032
cudaDevAttrClusterLaunch, 
#line 2033
cudaDevAttrDeferredMappingCudaArraySupported, 
#line 2034
cudaDevAttrReserved122, 
#line 2035
cudaDevAttrReserved123, 
#line 2036
cudaDevAttrReserved124, 
#line 2037
cudaDevAttrIpcEventSupport, 
#line 2038
cudaDevAttrMemSyncDomainCount, 
#line 2039
cudaDevAttrReserved127, 
#line 2040
cudaDevAttrReserved128, 
#line 2041
cudaDevAttrReserved129, 
#line 2042
cudaDevAttrNumaConfig, 
#line 2043
cudaDevAttrNumaId, 
#line 2044
cudaDevAttrReserved132, 
#line 2045
cudaDevAttrMpsEnabled, 
#line 2046
cudaDevAttrHostNumaId, 
#line 2047
cudaDevAttrD3D12CigSupported, 
#line 2048
cudaDevAttrVulkanCigSupported = 138, 
#line 2049
cudaDevAttrGpuPciDeviceId, 
#line 2050
cudaDevAttrGpuPciSubsystemId, 
#line 2051
cudaDevAttrReserved141, 
#line 2052
cudaDevAttrHostNumaMemoryPoolsSupported, 
#line 2053
cudaDevAttrHostNumaMultinodeIpcSupported, 
#line 2054
cudaDevAttrHostMemoryPoolsSupported, 
#line 2055
cudaDevAttrReserved145, 
#line 2056
cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147, 
#line 2058
cudaDevAttrMax
#line 2059
}; 
#endif
#line 2064 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2064
enum cudaMemPoolAttr { 
#line 2074
cudaMemPoolReuseFollowEventDependencies = 1, 
#line 2081
cudaMemPoolReuseAllowOpportunistic, 
#line 2089
cudaMemPoolReuseAllowInternalDependencies, 
#line 2100
cudaMemPoolAttrReleaseThreshold, 
#line 2106
cudaMemPoolAttrReservedMemCurrent, 
#line 2113
cudaMemPoolAttrReservedMemHigh, 
#line 2119
cudaMemPoolAttrUsedMemCurrent, 
#line 2126
cudaMemPoolAttrUsedMemHigh
#line 2127
}; 
#endif
#line 2132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2132
enum cudaMemLocationType { 
#line 2133
cudaMemLocationTypeInvalid, 
#line 2134
cudaMemLocationTypeNone = 0, 
#line 2135
cudaMemLocationTypeDevice, 
#line 2136
cudaMemLocationTypeHost, 
#line 2137
cudaMemLocationTypeHostNuma, 
#line 2138
cudaMemLocationTypeHostNumaCurrent
#line 2139
}; 
#endif
#line 2147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2147
struct cudaMemLocation { 
#line 2148
cudaMemLocationType type; 
#line 2149
int id; 
#line 2150
}; 
#endif
#line 2155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2155
enum cudaMemAccessFlags { 
#line 2156
cudaMemAccessFlagsProtNone, 
#line 2157
cudaMemAccessFlagsProtRead, 
#line 2158
cudaMemAccessFlagsProtReadWrite = 3
#line 2159
}; 
#endif
#line 2164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2164
struct cudaMemAccessDesc { 
#line 2165
cudaMemLocation location; 
#line 2166
cudaMemAccessFlags flags; 
#line 2167
}; 
#endif
#line 2172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2172
enum cudaMemAllocationType { 
#line 2173
cudaMemAllocationTypeInvalid, 
#line 2177
cudaMemAllocationTypePinned, 
#line 2180
cudaMemAllocationTypeManaged, 
#line 2181
cudaMemAllocationTypeMax = 2147483647
#line 2182
}; 
#endif
#line 2187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2187
enum cudaMemAllocationHandleType { 
#line 2188
cudaMemHandleTypeNone, 
#line 2189
cudaMemHandleTypePosixFileDescriptor, 
#line 2190
cudaMemHandleTypeWin32, 
#line 2191
cudaMemHandleTypeWin32Kmt = 4, 
#line 2192
cudaMemHandleTypeFabric = 8
#line 2193
}; 
#endif
#line 2204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2204
struct cudaMemPoolProps { 
#line 2205
cudaMemAllocationType allocType; 
#line 2206
cudaMemAllocationHandleType handleTypes; 
#line 2207
cudaMemLocation location; 
#line 2214
void *win32SecurityAttributes; 
#line 2215
size_t maxSize; 
#line 2216
unsigned short usage; 
#line 2217
unsigned char reserved[54]; 
#line 2218
}; 
#endif
#line 2223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2223
struct cudaMemPoolPtrExportData { 
#line 2224
unsigned char reserved[64]; 
#line 2225
}; 
#endif
#line 2230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2230
struct cudaMemAllocNodeParams { 
#line 2235
cudaMemPoolProps poolProps; 
#line 2236
const cudaMemAccessDesc *accessDescs; 
#line 2237
size_t accessDescCount; 
#line 2238
size_t bytesize; 
#line 2239
void *dptr; 
#line 2240
}; 
#endif
#line 2245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2245
struct cudaMemAllocNodeParamsV2 { 
#line 2250
cudaMemPoolProps poolProps; 
#line 2251
const cudaMemAccessDesc *accessDescs; 
#line 2252
size_t accessDescCount; 
#line 2253
size_t bytesize; 
#line 2254
void *dptr; 
#line 2255
}; 
#endif
#line 2260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2260
struct cudaMemFreeNodeParams { 
#line 2261
void *dptr; 
#line 2262
}; 
#endif
#line 2267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2267
enum cudaGraphMemAttributeType { 
#line 2272
cudaGraphMemAttrUsedMemCurrent, 
#line 2279
cudaGraphMemAttrUsedMemHigh, 
#line 2286
cudaGraphMemAttrReservedMemCurrent, 
#line 2293
cudaGraphMemAttrReservedMemHigh
#line 2294
}; 
#endif
#line 2299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2299
enum cudaMemcpyFlags { 
#line 2300
cudaMemcpyFlagDefault, 
#line 2305
cudaMemcpyFlagPreferOverlapWithCompute
#line 2306
}; 
#endif
#line 2308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2308
enum cudaMemcpySrcAccessOrder { 
#line 2312
cudaMemcpySrcAccessOrderInvalid, 
#line 2317
cudaMemcpySrcAccessOrderStream, 
#line 2328
cudaMemcpySrcAccessOrderDuringApiCall, 
#line 2337
cudaMemcpySrcAccessOrderAny, 
#line 2339
cudaMemcpySrcAccessOrderMax = 2147483647
#line 2340
}; 
#endif
#line 2345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2345
struct cudaMemcpyAttributes { 
#line 2346
cudaMemcpySrcAccessOrder srcAccessOrder; 
#line 2347
cudaMemLocation srcLocHint; 
#line 2348
cudaMemLocation dstLocHint; 
#line 2349
unsigned flags; 
#line 2350
}; 
#endif
#line 2355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2355
enum cudaMemcpy3DOperandType { 
#line 2356
cudaMemcpyOperandTypePointer = 1, 
#line 2357
cudaMemcpyOperandTypeArray, 
#line 2358
cudaMemcpyOperandTypeMax = 2147483647
#line 2359
}; 
#endif
#line 2364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2364
struct cudaOffset3D { 
#line 2365
size_t x; 
#line 2366
size_t y; 
#line 2367
size_t z; 
#line 2368
}; 
#endif
#line 2373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2373
struct cudaMemcpy3DOperand { 
#line 2374
cudaMemcpy3DOperandType type; 
#line 2375
union { 
#line 2379
struct { 
#line 2380
void *ptr; 
#line 2381
size_t rowLength; 
#line 2382
size_t layerHeight; 
#line 2383
cudaMemLocation locHint; 
#line 2384
} ptr; 
#line 2389
struct { 
#line 2390
cudaArray_t array; 
#line 2391
cudaOffset3D offset; 
#line 2392
} array; 
#line 2393
} op; 
#line 2394
}; 
#endif
#line 2396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2396
struct cudaMemcpy3DBatchOp { 
#line 2397
cudaMemcpy3DOperand src; 
#line 2398
cudaMemcpy3DOperand dst; 
#line 2399
cudaExtent extent; 
#line 2400
cudaMemcpySrcAccessOrder srcAccessOrder; 
#line 2401
unsigned flags; 
#line 2402
}; 
#endif
#line 2408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2408
enum cudaDeviceP2PAttr { 
#line 2409
cudaDevP2PAttrPerformanceRank = 1, 
#line 2410
cudaDevP2PAttrAccessSupported, 
#line 2411
cudaDevP2PAttrNativeAtomicSupported, 
#line 2412
cudaDevP2PAttrCudaArrayAccessSupported, 
#line 2414
cudaDevP2PAttrOnlyPartialNativeAtomicSupported
#line 2416
}; 
#endif
#line 2421 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2421
enum cudaAtomicOperation { 
#line 2422
cudaAtomicOperationIntegerAdd, 
#line 2423
cudaAtomicOperationIntegerMin, 
#line 2424
cudaAtomicOperationIntegerMax, 
#line 2425
cudaAtomicOperationIntegerIncrement, 
#line 2426
cudaAtomicOperationIntegerDecrement, 
#line 2427
cudaAtomicOperationAnd, 
#line 2428
cudaAtomicOperationOr, 
#line 2429
cudaAtomicOperationXOR, 
#line 2430
cudaAtomicOperationExchange, 
#line 2431
cudaAtomicOperationCAS, 
#line 2432
cudaAtomicOperationFloatAdd, 
#line 2433
cudaAtomicOperationFloatMin, 
#line 2434
cudaAtomicOperationFloatMax
#line 2435
}; 
#endif
#line 2440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2440
enum cudaAtomicOperationCapability { 
#line 2441
cudaAtomicCapabilitySigned = (1U << 0), 
#line 2442
cudaAtomicCapabilityUnsigned, 
#line 2443
cudaAtomicCapabilityReduction = (1U << 2), 
#line 2444
cudaAtomicCapabilityScalar32 = (1U << 3), 
#line 2445
cudaAtomicCapabilityScalar64 = (1U << 4), 
#line 2446
cudaAtomicCapabilityScalar128 = (1U << 5), 
#line 2447
cudaAtomicCapabilityVector32x4 = (1U << 6)
#line 2448
}; 
#endif
#line 2457 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2457
struct CUuuid_st { 
#line 2458
char bytes[16]; 
#line 2459
}; 
#endif
#line 2460 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2460
CUuuid; 
#endif
#line 2462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2462
cudaUUID_t; 
#endif
#line 2467 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2467
struct cudaDeviceProp { 
#line 2469
char name[256]; 
#line 2470
cudaUUID_t uuid; 
#line 2471
char luid[8]; 
#line 2472
unsigned luidDeviceNodeMask; 
#line 2473
size_t totalGlobalMem; 
#line 2474
size_t sharedMemPerBlock; 
#line 2475
int regsPerBlock; 
#line 2476
int warpSize; 
#line 2477
size_t memPitch; 
#line 2478
int maxThreadsPerBlock; 
#line 2479
int maxThreadsDim[3]; 
#line 2480
int maxGridSize[3]; 
#line 2481
size_t totalConstMem; 
#line 2482
int major; 
#line 2483
int minor; 
#line 2484
size_t textureAlignment; 
#line 2485
size_t texturePitchAlignment; 
#line 2486
int multiProcessorCount; 
#line 2487
int integrated; 
#line 2488
int canMapHostMemory; 
#line 2489
int maxTexture1D; 
#line 2490
int maxTexture1DMipmap; 
#line 2491
int maxTexture2D[2]; 
#line 2492
int maxTexture2DMipmap[2]; 
#line 2493
int maxTexture2DLinear[3]; 
#line 2494
int maxTexture2DGather[2]; 
#line 2495
int maxTexture3D[3]; 
#line 2496
int maxTexture3DAlt[3]; 
#line 2497
int maxTextureCubemap; 
#line 2498
int maxTexture1DLayered[2]; 
#line 2499
int maxTexture2DLayered[3]; 
#line 2500
int maxTextureCubemapLayered[2]; 
#line 2501
int maxSurface1D; 
#line 2502
int maxSurface2D[2]; 
#line 2503
int maxSurface3D[3]; 
#line 2504
int maxSurface1DLayered[2]; 
#line 2505
int maxSurface2DLayered[3]; 
#line 2506
int maxSurfaceCubemap; 
#line 2507
int maxSurfaceCubemapLayered[2]; 
#line 2508
size_t surfaceAlignment; 
#line 2509
int concurrentKernels; 
#line 2510
int ECCEnabled; 
#line 2511
int pciBusID; 
#line 2512
int pciDeviceID; 
#line 2513
int pciDomainID; 
#line 2514
int tccDriver; 
#line 2515
int asyncEngineCount; 
#line 2516
int unifiedAddressing; 
#line 2517
int memoryBusWidth; 
#line 2518
int l2CacheSize; 
#line 2519
int persistingL2CacheMaxSize; 
#line 2520
int maxThreadsPerMultiProcessor; 
#line 2521
int streamPrioritiesSupported; 
#line 2522
int globalL1CacheSupported; 
#line 2523
int localL1CacheSupported; 
#line 2524
size_t sharedMemPerMultiprocessor; 
#line 2525
int regsPerMultiprocessor; 
#line 2526
int managedMemory; 
#line 2527
int isMultiGpuBoard; 
#line 2528
int multiGpuBoardGroupID; 
#line 2529
int hostNativeAtomicSupported; 
#line 2530
int pageableMemoryAccess; 
#line 2531
int concurrentManagedAccess; 
#line 2532
int computePreemptionSupported; 
#line 2533
int canUseHostPointerForRegisteredMem; 
#line 2534
int cooperativeLaunch; 
#line 2535
size_t sharedMemPerBlockOptin; 
#line 2536
int pageableMemoryAccessUsesHostPageTables; 
#line 2537
int directManagedMemAccessFromHost; 
#line 2538
int maxBlocksPerMultiProcessor; 
#line 2539
int accessPolicyMaxWindowSize; 
#line 2540
size_t reservedSharedMemPerBlock; 
#line 2541
int hostRegisterSupported; 
#line 2542
int sparseCudaArraySupported; 
#line 2543
int hostRegisterReadOnlySupported; 
#line 2544
int timelineSemaphoreInteropSupported; 
#line 2545
int memoryPoolsSupported; 
#line 2546
int gpuDirectRDMASupported; 
#line 2547
unsigned gpuDirectRDMAFlushWritesOptions; 
#line 2548
int gpuDirectRDMAWritesOrdering; 
#line 2549
unsigned memoryPoolSupportedHandleTypes; 
#line 2550
int deferredMappingCudaArraySupported; 
#line 2551
int ipcEventSupported; 
#line 2552
int clusterLaunch; 
#line 2553
int unifiedFunctionPointers; 
#line 2554
int deviceNumaConfig; 
#line 2555
int deviceNumaId; 
#line 2556
int mpsEnabled; 
#line 2557
int hostNumaId; 
#line 2558
unsigned gpuPciDeviceID; 
#line 2559
unsigned gpuPciSubsystemID; 
#line 2560
int hostNumaMultinodeIpcSupported; 
#line 2561
int reserved[56]; 
#line 2562
}; 
#endif
#line 2575 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 2572
struct cudaIpcEventHandle_st { 
#line 2574
char reserved[64]; 
#line 2575
} cudaIpcEventHandle_t; 
#endif
#line 2583 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 2580
struct cudaIpcMemHandle_st { 
#line 2582
char reserved[64]; 
#line 2583
} cudaIpcMemHandle_t; 
#endif
#line 2591 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 2588
struct cudaMemFabricHandle_st { 
#line 2590
char reserved[64]; 
#line 2591
} cudaMemFabricHandle_t; 
#endif
#line 2596 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2596
enum cudaExternalMemoryHandleType { 
#line 2600
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
#line 2604
cudaExternalMemoryHandleTypeOpaqueWin32, 
#line 2608
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
#line 2612
cudaExternalMemoryHandleTypeD3D12Heap, 
#line 2616
cudaExternalMemoryHandleTypeD3D12Resource, 
#line 2620
cudaExternalMemoryHandleTypeD3D11Resource, 
#line 2624
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
#line 2628
cudaExternalMemoryHandleTypeNvSciBuf
#line 2629
}; 
#endif
#line 2671 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2671
struct cudaExternalMemoryHandleDesc { 
#line 2675
cudaExternalMemoryHandleType type; 
#line 2676
union { 
#line 2682
int fd; 
#line 2698
struct { 
#line 2702
void *handle; 
#line 2707
const void *name; 
#line 2708
} win32; 
#line 2713
const void *nvSciBufObject; 
#line 2714
} handle; 
#line 2718
unsigned __int64 size; 
#line 2722
unsigned flags; 
#line 2726
unsigned reserved[16]; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2727
}; 
#endif
#line 2732 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2732
struct cudaExternalMemoryBufferDesc { 
#line 2736
unsigned __int64 offset; 
#line 2740
unsigned __int64 size; 
#line 2744
unsigned flags; 
#line 2748
unsigned reserved[16]; 
#line 2749
}; 
#endif
#line 2754 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2754
struct cudaExternalMemoryMipmappedArrayDesc { 
#line 2759
unsigned __int64 offset; 
#line 2763
cudaChannelFormatDesc formatDesc; 
#line 2767
cudaExtent extent; 
#line 2772
unsigned flags; 
#line 2776
unsigned numLevels; 
#line 2780
unsigned reserved[16]; 
#line 2781
}; 
#endif
#line 2786 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2786
enum cudaExternalSemaphoreHandleType { 
#line 2790
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
#line 2794
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
#line 2798
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
#line 2802
cudaExternalSemaphoreHandleTypeD3D12Fence, 
#line 2806
cudaExternalSemaphoreHandleTypeD3D11Fence, 
#line 2810
cudaExternalSemaphoreHandleTypeNvSciSync, 
#line 2814
cudaExternalSemaphoreHandleTypeKeyedMutex, 
#line 2818
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
#line 2822
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
#line 2826
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
#line 2827
}; 
#endif
#line 2832 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2832
struct cudaExternalSemaphoreHandleDesc { 
#line 2836
cudaExternalSemaphoreHandleType type; 
#line 2837
union { 
#line 2844
int fd; 
#line 2860
struct { 
#line 2864
void *handle; 
#line 2869
const void *name; 
#line 2870
} win32; 
#line 2874
const void *nvSciSyncObj; 
#line 2875
} handle; 
#line 2879
unsigned flags; 
#line 2883
unsigned reserved[16]; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2884
}; 
#endif
#line 2889 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2889
struct cudaExternalSemaphoreSignalParams { 
#line 2890
struct { 
#line 2894
struct { 
#line 2898
unsigned __int64 value; 
#line 2899
} fence; 
#line 2900
union { 
#line 2905
void *fence; 
#line 2906
unsigned __int64 reserved; 
#line 2907
} nvSciSync; 
#line 2911
struct { 
#line 2915
unsigned __int64 key; 
#line 2916
} keyedMutex; 
#line 2917
unsigned reserved[12]; 
#line 2918
} params; 
#line 2929
unsigned flags; 
#line 2930
unsigned reserved[16]; 
#line 2931
}; 
#endif
#line 2936 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 2936
struct cudaExternalSemaphoreWaitParams { 
#line 2937
struct { 
#line 2941
struct { 
#line 2945
unsigned __int64 value; 
#line 2946
} fence; 
#line 2947
union { 
#line 2952
void *fence; 
#line 2953
unsigned __int64 reserved; 
#line 2954
} nvSciSync; 
#line 2958
struct { 
#line 2962
unsigned __int64 key; 
#line 2966
unsigned timeoutMs; 
#line 2967
} keyedMutex; 
#line 2968
unsigned reserved[10]; 
#line 2969
} params; 
#line 2980
unsigned flags; 
#line 2981
unsigned reserved[16]; 
#line 2982
}; 
#endif
#line 2993 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 2993
cudaError_t; 
#endif
#line 2998 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 2998
cudaStream_t; 
#endif
#line 3003 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 3003
cudaEvent_t; 
#endif
#line 3008 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 3008
cudaGraphicsResource_t; 
#endif
#line 3013 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
#line 3013
cudaExternalMemory_t; 
#endif
#line 3018 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
#line 3018
cudaExternalSemaphore_t; 
#endif
#line 3023 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUgraph_st *
#line 3023
cudaGraph_t; 
#endif
#line 3028 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUgraphNode_st *
#line 3028
cudaGraphNode_t; 
#endif
#line 3033 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUuserObject_st *
#line 3033
cudaUserObject_t; 
#endif
#line 3038 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef unsigned __int64 
#line 3038
cudaGraphConditionalHandle; 
#endif
#line 3043 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUfunc_st *
#line 3043
cudaFunction_t; 
#endif
#line 3048 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUkern_st *
#line 3048
cudaKernel_t; 
#endif
#line 3053 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3053
enum cudaJitOption { 
#line 3060
cudaJitMaxRegisters, 
#line 3074
cudaJitThreadsPerBlock, 
#line 3082
cudaJitWallTime, 
#line 3091
cudaJitInfoLogBuffer, 
#line 3100
cudaJitInfoLogBufferSizeBytes, 
#line 3109
cudaJitErrorLogBuffer, 
#line 3118
cudaJitErrorLogBufferSizeBytes, 
#line 3126
cudaJitOptimizationLevel, 
#line 3134
cudaJitFallbackStrategy = 10, 
#line 3142
cudaJitGenerateDebugInfo, 
#line 3149
cudaJitLogVerbose, 
#line 3156
cudaJitGenerateLineInfo, 
#line 3164
cudaJitCacheMode, 
#line 3171
cudaJitPositionIndependentCode = 30, 
#line 3184
cudaJitMinCtaPerSm, 
#line 3197
cudaJitMaxThreadsPerBlock, 
#line 3207
cudaJitOverrideDirectiveValues
#line 3208
}; 
#endif
#line 3214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3214
enum cudaLibraryOption { 
#line 3216
cudaLibraryHostUniversalFunctionAndDataTable, 
#line 3227
cudaLibraryBinaryIsPreserved
#line 3228
}; 
#endif
#line 3230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3230
struct cudalibraryHostUniversalFunctionAndDataTable { 
#line 3232
void *functionTable; 
#line 3233
size_t functionWindowSize; 
#line 3234
void *dataTable; 
#line 3235
size_t dataWindowSize; 
#line 3236
}; 
#endif
#line 3241 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3241
enum cudaJit_CacheMode { 
#line 3243
cudaJitCacheOptionNone, 
#line 3244
cudaJitCacheOptionCG, 
#line 3245
cudaJitCacheOptionCA
#line 3246
}; 
#endif
#line 3251 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3251
enum cudaJit_Fallback { 
#line 3253
cudaPreferPtx, 
#line 3255
cudaPreferBinary
#line 3256
}; 
#endif
#line 3261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUlib_st *
#line 3261
cudaLibrary_t; 
#endif
#line 3266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
#line 3266
cudaMemPool_t; 
#endif
#line 3271 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3271
enum cudaCGScope { 
#line 3272
cudaCGScopeInvalid, 
#line 3273
cudaCGScopeGrid, 
#line 3274
cudaCGScopeReserved
#line 3275
}; 
#endif
#line 3280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3280
struct cudaKernelNodeParams { 
#line 3281
void *func; 
#line 3282
dim3 gridDim; 
#line 3283
dim3 blockDim; 
#line 3284
unsigned sharedMemBytes; 
#line 3285
void **kernelParams; 
#line 3286
void **extra; 
#line 3287
}; 
#endif
#line 3292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3292
struct cudaKernelNodeParamsV2 { 
#line 3293
void *func; 
#line 3299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
uint3 gridDim; 
#line 3300
uint3 blockDim; 
#line 3302 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
unsigned sharedMemBytes; 
#line 3303
void **kernelParams; 
#line 3304
void **extra; 
#line 3305
}; 
#endif
#line 3310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3310
struct cudaExternalSemaphoreSignalNodeParams { 
#line 3311
cudaExternalSemaphore_t *extSemArray; 
#line 3312
const cudaExternalSemaphoreSignalParams *paramsArray; 
#line 3313
unsigned numExtSems; 
#line 3314
}; 
#endif
#line 3319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3319
struct cudaExternalSemaphoreSignalNodeParamsV2 { 
#line 3320
cudaExternalSemaphore_t *extSemArray; 
#line 3321
const cudaExternalSemaphoreSignalParams *paramsArray; 
#line 3322
unsigned numExtSems; 
#line 3323
}; 
#endif
#line 3328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3328
struct cudaExternalSemaphoreWaitNodeParams { 
#line 3329
cudaExternalSemaphore_t *extSemArray; 
#line 3330
const cudaExternalSemaphoreWaitParams *paramsArray; 
#line 3331
unsigned numExtSems; 
#line 3332
}; 
#endif
#line 3337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3337
struct cudaExternalSemaphoreWaitNodeParamsV2 { 
#line 3338
cudaExternalSemaphore_t *extSemArray; 
#line 3339
const cudaExternalSemaphoreWaitParams *paramsArray; 
#line 3340
unsigned numExtSems; 
#line 3341
}; 
#endif
#line 3343 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3343
enum cudaGraphConditionalHandleFlags { 
#line 3344
cudaGraphCondAssignDefault = 1
#line 3345
}; 
#endif
#line 3350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3350
enum cudaGraphConditionalNodeType { 
#line 3351
cudaGraphCondTypeIf, 
#line 3352
cudaGraphCondTypeWhile, 
#line 3353
cudaGraphCondTypeSwitch
#line 3354
}; 
#endif
#line 3359 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3359
struct cudaConditionalNodeParams { 
#line 3360
cudaGraphConditionalHandle handle; 
#line 3363
cudaGraphConditionalNodeType type; 
#line 3364
unsigned size; 
#line 3366
cudaGraph_t *phGraph_out; 
#line 3385
}; 
#endif
#line 3390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3390
enum cudaGraphNodeType { 
#line 3391
cudaGraphNodeTypeKernel, 
#line 3392
cudaGraphNodeTypeMemcpy, 
#line 3393
cudaGraphNodeTypeMemset, 
#line 3394
cudaGraphNodeTypeHost, 
#line 3395
cudaGraphNodeTypeGraph, 
#line 3396
cudaGraphNodeTypeEmpty, 
#line 3397
cudaGraphNodeTypeWaitEvent, 
#line 3398
cudaGraphNodeTypeEventRecord, 
#line 3399
cudaGraphNodeTypeExtSemaphoreSignal, 
#line 3400
cudaGraphNodeTypeExtSemaphoreWait, 
#line 3401
cudaGraphNodeTypeMemAlloc, 
#line 3402
cudaGraphNodeTypeMemFree, 
#line 3403
cudaGraphNodeTypeConditional = 13, 
#line 3420
cudaGraphNodeTypeCount
#line 3421
}; 
#endif
#line 3426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3426
enum cudaGraphChildGraphNodeOwnership { 
#line 3427
cudaGraphChildGraphOwnershipClone, 
#line 3430
cudaGraphChildGraphOwnershipMove
#line 3439
}; 
#endif
#line 3444 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3444
struct cudaChildGraphNodeParams { 
#line 3445
cudaGraph_t graph; 
#line 3451
cudaGraphChildGraphNodeOwnership ownership; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3452
}; 
#endif
#line 3457 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3457
struct cudaEventRecordNodeParams { 
#line 3458
cudaEvent_t event; 
#line 3459
}; 
#endif
#line 3464 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3464
struct cudaEventWaitNodeParams { 
#line 3465
cudaEvent_t event; 
#line 3466
}; 
#endif
#line 3471 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3471
struct cudaGraphNodeParams { 
#line 3472
cudaGraphNodeType type; 
#line 3473
int reserved0[3]; 
#line 3475
union { 
#line 3476
__int64 reserved1[29]; 
#line 3477
cudaKernelNodeParamsV2 kernel; 
#line 3478
cudaMemcpyNodeParams memcpy; 
#line 3479
cudaMemsetParamsV2 memset; 
#line 3480
cudaHostNodeParamsV2 host; 
#line 3481
cudaChildGraphNodeParams graph; 
#line 3482
cudaEventWaitNodeParams eventWait; 
#line 3483
cudaEventRecordNodeParams eventRecord; 
#line 3484
cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal; 
#line 3485
cudaExternalSemaphoreWaitNodeParamsV2 extSemWait; 
#line 3486
cudaMemAllocNodeParamsV2 alloc; 
#line 3487
cudaMemFreeNodeParams free; 
#line 3488
cudaConditionalNodeParams conditional; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3489
}; 
#line 3491
__int64 reserved2; 
#line 3492
}; 
#endif
#line 3504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3497
enum cudaGraphDependencyType_enum { 
#line 3498
cudaGraphDependencyTypeDefault, 
#line 3499
cudaGraphDependencyTypeProgrammatic
#line 3504
} cudaGraphDependencyType; 
#endif
#line 3534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3511
struct cudaGraphEdgeData_st { 
#line 3512
unsigned char from_port; 
#line 3522
unsigned char to_port; 
#line 3529
unsigned char type; 
#line 3532
unsigned char reserved[5]; 
#line 3534
} cudaGraphEdgeData; 
#endif
#line 3555 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
#line 3560
#if 0
#line 3560
enum cudaGraphExecUpdateResult { 
#line 3561
cudaGraphExecUpdateSuccess, 
#line 3562
cudaGraphExecUpdateError, 
#line 3563
cudaGraphExecUpdateErrorTopologyChanged, 
#line 3564
cudaGraphExecUpdateErrorNodeTypeChanged, 
#line 3565
cudaGraphExecUpdateErrorFunctionChanged, 
#line 3566
cudaGraphExecUpdateErrorParametersChanged, 
#line 3567
cudaGraphExecUpdateErrorNotSupported, 
#line 3568
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
#line 3569
cudaGraphExecUpdateErrorAttributesChanged
#line 3570
}; 
#endif
#line 3582 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3575
enum cudaGraphInstantiateResult { 
#line 3576
cudaGraphInstantiateSuccess, 
#line 3577
cudaGraphInstantiateError, 
#line 3578
cudaGraphInstantiateInvalidStructure, 
#line 3579
cudaGraphInstantiateNodeOperationNotSupported, 
#line 3580
cudaGraphInstantiateMultipleDevicesNotSupported, 
#line 3581
cudaGraphInstantiateConditionalHandleUnused
#line 3582
} cudaGraphInstantiateResult; 
#endif
#line 3593 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3587
struct cudaGraphInstantiateParams_st { 
#line 3589
unsigned __int64 flags; 
#line 3590
cudaStream_t uploadStream; 
#line 3591
cudaGraphNode_t errNode_out; 
#line 3592
cudaGraphInstantiateResult result_out; 
#line 3593
} cudaGraphInstantiateParams; 
#endif
#line 3615 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3598
struct cudaGraphExecUpdateResultInfo_st { 
#line 3602
cudaGraphExecUpdateResult result; 
#line 3609
cudaGraphNode_t errorNode; 
#line 3614
cudaGraphNode_t errorFromNode; 
#line 3615
} cudaGraphExecUpdateResultInfo; 
#endif
#line 3620 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef struct CUgraphDeviceUpdatableNode_st *cudaGraphDeviceNode_t; 
#line 3625
#if 0
#line 3625
enum cudaGraphKernelNodeField { 
#line 3627
cudaGraphKernelNodeFieldInvalid, 
#line 3628
cudaGraphKernelNodeFieldGridDim, 
#line 3629
cudaGraphKernelNodeFieldParam, 
#line 3630
cudaGraphKernelNodeFieldEnabled
#line 3631
}; 
#endif
#line 3636 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3636
struct cudaGraphKernelNodeUpdate { 
#line 3637
cudaGraphDeviceNode_t node; 
#line 3638
cudaGraphKernelNodeField field; 
#line 3639
union { 
#line 3644 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
uint3 gridDim; 
#line 3646 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
struct { 
#line 3647
const void *pValue; 
#line 3648
size_t offset; 
#line 3649
size_t size; 
#line 3650
} param; 
#line 3651
unsigned isEnabled; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3652
} updateData; 
#line 3653
}; 
#endif
#line 3659 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3659
enum cudaGetDriverEntryPointFlags { 
#line 3660
cudaEnableDefault, 
#line 3661
cudaEnableLegacyStream, 
#line 3662
cudaEnablePerThreadDefaultStream
#line 3663
}; 
#endif
#line 3668 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3668
enum cudaDriverEntryPointQueryResult { 
#line 3669
cudaDriverEntryPointSuccess, 
#line 3670
cudaDriverEntryPointSymbolNotFound, 
#line 3671
cudaDriverEntryPointVersionNotSufficent
#line 3672
}; 
#endif
#line 3677 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3677
enum cudaGraphDebugDotFlags { 
#line 3678
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
#line 3679
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
#line 3680
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
#line 3681
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
#line 3682
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
#line 3683
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
#line 3684
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
#line 3685
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
#line 3686
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
#line 3687
cudaGraphDebugDotFlagsHandles = (1 << 10), 
#line 3688
cudaGraphDebugDotFlagsConditionalNodeParams = (1 << 15)
#line 3689
}; 
#endif
#line 3694 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 3694
enum cudaGraphInstantiateFlags { 
#line 3695
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
#line 3696
cudaGraphInstantiateFlagUpload, 
#line 3699
cudaGraphInstantiateFlagDeviceLaunch = 4, 
#line 3702
cudaGraphInstantiateFlagUseNodePriority = 8
#line 3704
}; 
#endif
#line 3725 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3722
enum cudaLaunchMemSyncDomain { 
#line 3723
cudaLaunchMemSyncDomainDefault, 
#line 3724
cudaLaunchMemSyncDomainRemote
#line 3725
} cudaLaunchMemSyncDomain; 
#endif
#line 3741 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3738
struct cudaLaunchMemSyncDomainMap_st { 
#line 3739
unsigned char default_; 
#line 3740
unsigned char remote; 
#line 3741
} cudaLaunchMemSyncDomainMap; 
#endif
#line 3906 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3746
enum cudaLaunchAttributeID { 
#line 3747
cudaLaunchAttributeIgnore, 
#line 3748
cudaLaunchAttributeAccessPolicyWindow, 
#line 3750
cudaLaunchAttributeCooperative, 
#line 3752
cudaLaunchAttributeSynchronizationPolicy, 
#line 3753
cudaLaunchAttributeClusterDimension, 
#line 3755
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
#line 3757
cudaLaunchAttributeProgrammaticStreamSerialization, 
#line 3768
cudaLaunchAttributeProgrammaticEvent, 
#line 3794
cudaLaunchAttributePriority, 
#line 3796
cudaLaunchAttributeMemSyncDomainMap, 
#line 3798
cudaLaunchAttributeMemSyncDomain, 
#line 3800
cudaLaunchAttributePreferredClusterDimension, 
#line 3836
cudaLaunchAttributeLaunchCompletionEvent, 
#line 3858
cudaLaunchAttributeDeviceUpdatableKernelNode, 
#line 3886
cudaLaunchAttributePreferredSharedMemoryCarveout, 
#line 3893
cudaLaunchAttributeNvlinkUtilCentricScheduling = 16
#line 3906
} cudaLaunchAttributeID; 
#endif
#line 4004 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 3911
union cudaLaunchAttributeValue { 
#line 3912
char pad[64]; 
#line 3913
cudaAccessPolicyWindow accessPolicyWindow; 
#line 3914
int cooperative; 
#line 3916
cudaSynchronizationPolicy syncPolicy; 
#line 3930
struct { 
#line 3931
unsigned x; 
#line 3932
unsigned y; 
#line 3933
unsigned z; 
#line 3934
} clusterDim; 
#line 3935
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
#line 3938
int programmaticStreamSerializationAllowed; 
#line 3949
struct { 
#line 3950
cudaEvent_t event; 
#line 3951
int flags; 
#line 3952
int triggerAtBlockStart; 
#line 3953
} programmaticEvent; 
#line 3954
int priority; 
#line 3955
cudaLaunchMemSyncDomainMap memSyncDomainMap; 
#line 3958
cudaLaunchMemSyncDomain memSyncDomain; 
#line 3973
struct { 
#line 3974
unsigned x; 
#line 3975
unsigned y; 
#line 3976
unsigned z; 
#line 3977
} preferredClusterDim; 
#line 3986
struct { 
#line 3987
cudaEvent_t event; 
#line 3988
int flags; 
#line 3989
} launchCompletionEvent; 
#line 3997
struct { 
#line 3998
int deviceUpdatable; 
#line 3999
cudaGraphDeviceNode_t devNode; 
#line 4000
} deviceUpdatableKernelNode; 
#line 4001
unsigned sharedMemCarveout; 
#line 4002
unsigned nvlinkUtilCentricScheduling; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 4004
} cudaLaunchAttributeValue; 
#endif
#line 4013 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 4009
struct cudaLaunchAttribute_st { 
#line 4010
cudaLaunchAttributeID id; 
#line 4011
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
#line 4012
cudaLaunchAttributeValue val; 
#line 4013
} cudaLaunchAttribute; 
#endif
#line 4025 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 4018
struct cudaLaunchConfig_st { 
#line 4019
dim3 gridDim; 
#line 4020
dim3 blockDim; 
#line 4021
size_t dynamicSmemBytes; 
#line 4022
cudaStream_t stream; 
#line 4023
cudaLaunchAttribute *attrs; 
#line 4024
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 4025
} cudaLaunchConfig_t; 
#endif
#line 4054 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
#line 4054
enum cudaDeviceNumaConfig { 
#line 4055
cudaDeviceNumaConfigNone, 
#line 4056
cudaDeviceNumaConfigNumaNode
#line 4057
}; 
#endif
#line 4062 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef struct cudaAsyncCallbackEntry *cudaAsyncCallbackHandle_t; 
#line 4064
struct cudaAsyncCallbackEntry; 
#line 4071
#if 0
typedef 
#line 4069
enum cudaAsyncNotificationType_enum { 
#line 4070
cudaAsyncNotificationTypeOverBudget = 1
#line 4071
} cudaAsyncNotificationType; 
#endif
#line 4084 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef 
#line 4076
struct cudaAsyncNotificationInfo { 
#line 4078
cudaAsyncNotificationType type; 
#line 4079
union { 
#line 4080
struct { 
#line 4081
unsigned __int64 bytesOverBudget; 
#line 4082
} overBudget; 
#line 4083
} info; 
#line 4084
} cudaAsyncNotificationInfo_t; 
#endif
#line 4086 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
typedef void (*cudaAsyncCallback)(cudaAsyncNotificationInfo_t *, void *, cudaAsyncCallbackHandle_t); 
#line 4091
#if 0
typedef 
#line 4088
enum CUDAlogLevel_enum { 
#line 4089
cudaLogLevelError, 
#line 4090
cudaLogLevelWarning
#line 4091
} cudaLogLevel; 
#endif
#line 4093 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef struct CUlogsCallbackEntry_st *
#line 4093
cudaLogsCallbackHandle; 
#endif
#line 4094 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_types.h"
#if 0
typedef unsigned 
#line 4094
cudaLogIterator; 
#endif
#line 86 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\surface_types.h"
#if 0
#line 86
enum cudaSurfaceBoundaryMode { 
#line 88
cudaBoundaryModeZero, 
#line 89
cudaBoundaryModeClamp, 
#line 90
cudaBoundaryModeTrap
#line 91
}; 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\surface_types.h"
#if 0
#line 96
enum cudaSurfaceFormatMode { 
#line 98
cudaFormatModeForced, 
#line 99
cudaFormatModeAuto
#line 100
}; 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 105
cudaSurfaceObject_t; 
#endif
#line 86 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\texture_types.h"
#if 0
#line 86
enum cudaTextureAddressMode { 
#line 88
cudaAddressModeWrap, 
#line 89
cudaAddressModeClamp, 
#line 90
cudaAddressModeMirror, 
#line 91
cudaAddressModeBorder
#line 92
}; 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\texture_types.h"
#if 0
#line 97
enum cudaTextureFilterMode { 
#line 99
cudaFilterModePoint, 
#line 100
cudaFilterModeLinear
#line 101
}; 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\texture_types.h"
#if 0
#line 106
enum cudaTextureReadMode { 
#line 108
cudaReadModeElementType, 
#line 109
cudaReadModeNormalizedFloat
#line 110
}; 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\texture_types.h"
#if 0
#line 115
struct cudaTextureDesc { 
#line 120
cudaTextureAddressMode addressMode[3]; 
#line 124
cudaTextureFilterMode filterMode; 
#line 128
cudaTextureReadMode readMode; 
#line 132
int sRGB; 
#line 136
float borderColor[4]; 
#line 140
int normalizedCoords; 
#line 144
unsigned maxAnisotropy; 
#line 148
cudaTextureFilterMode mipmapFilterMode; 
#line 152
float mipmapLevelBias; 
#line 156
float minMipmapLevelClamp; 
#line 160
float maxMipmapLevelClamp; 
#line 164
int disableTrilinearOptimization; 
#line 168
int seamlessCubemap; 
#line 169
}; 
#endif
#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 174
cudaTextureObject_t; 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\library_types.h"
typedef 
#line 57
enum cudaDataType_t { 
#line 59
CUDA_R_16F = 2, 
#line 60
CUDA_C_16F = 6, 
#line 61
CUDA_R_16BF = 14, 
#line 62
CUDA_C_16BF, 
#line 63
CUDA_R_32F = 0, 
#line 64
CUDA_C_32F = 4, 
#line 65
CUDA_R_64F = 1, 
#line 66
CUDA_C_64F = 5, 
#line 67
CUDA_R_4I = 16, 
#line 68
CUDA_C_4I, 
#line 69
CUDA_R_4U, 
#line 70
CUDA_C_4U, 
#line 71
CUDA_R_8I = 3, 
#line 72
CUDA_C_8I = 7, 
#line 73
CUDA_R_8U, 
#line 74
CUDA_C_8U, 
#line 75
CUDA_R_16I = 20, 
#line 76
CUDA_C_16I, 
#line 77
CUDA_R_16U, 
#line 78
CUDA_C_16U, 
#line 79
CUDA_R_32I = 10, 
#line 80
CUDA_C_32I, 
#line 81
CUDA_R_32U, 
#line 82
CUDA_C_32U, 
#line 83
CUDA_R_64I = 24, 
#line 84
CUDA_C_64I, 
#line 85
CUDA_R_64U, 
#line 86
CUDA_C_64U, 
#line 87
CUDA_R_8F_E4M3, 
#line 88
CUDA_R_8F_UE4M3 = CUDA_R_8F_E4M3, 
#line 89
CUDA_R_8F_E5M2, 
#line 90
CUDA_R_8F_UE8M0, 
#line 91
CUDA_R_6F_E2M3, 
#line 92
CUDA_R_6F_E3M2, 
#line 93
CUDA_R_4F_E2M1
#line 94
} cudaDataType; 
#line 115
typedef 
#line 98
enum cudaEmulationStrategy_t { 
#line 104
CUDA_EMULATION_STRATEGY_DEFAULT, 
#line 109
CUDA_EMULATION_STRATEGY_PERFORMANT, 
#line 114
CUDA_EMULATION_STRATEGY_EAGER
#line 115
} cudaEmulationStrategy; 
#line 122
typedef 
#line 117
enum libraryPropertyType_t { 
#line 119
MAJOR_VERSION, 
#line 120
MINOR_VERSION, 
#line 121
PATCH_LEVEL
#line 122
} libraryPropertyType; 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_malloc.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 58 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_malloc.h"
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 59
_calloc_base(size_t _Count, size_t _Size); 
#line 65
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 66
calloc(size_t _Count, size_t _Size); 
#line 72
int __cdecl _callnewh(size_t _Size); 
#line 77
__declspec(allocator) void *__cdecl 
#line 78
_expand(void * _Block, size_t _Size); 
#line 84
void __cdecl _free_base(void * _Block); 
#line 89
void __cdecl free(void * _Block); 
#line 94
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 95
_malloc_base(size_t _Size); 
#line 100
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 101
malloc(size_t _Size); 
#line 107
size_t __cdecl _msize_base(void * _Block) noexcept; 
#line 113
size_t __cdecl _msize(void * _Block); 
#line 118
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 119
_realloc_base(void * _Block, size_t _Size); 
#line 125
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 126
realloc(void * _Block, size_t _Size); 
#line 132
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 133
_recalloc_base(void * _Block, size_t _Count, size_t _Size); 
#line 140
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 141
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 148
void __cdecl _aligned_free(void * _Block); 
#line 153
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 154
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 160
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 161
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 169
size_t __cdecl _aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 176
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 177
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 185
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 186
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 195
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 196
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 203
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 204
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 232 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 234
#pragma warning(pop)
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_search.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef int (__cdecl *_CoreCrtSecureSearchSortCompareFunction)(void *, const void *, const void *); 
#line 24
typedef int (__cdecl *_CoreCrtNonSecureSearchSortCompareFunction)(const void *, const void *); 
#line 30
void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 39
void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 52 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_search.h"
void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 60
void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 68
void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 78
void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 87
void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 97
void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 195 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_search.h"
void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 204
void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 216 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 218
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 54
errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 61
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itow_s ( int _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 77 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 84
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltow_s ( long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 99 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 106
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultow_s ( unsigned long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 113 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 127
double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 134
long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 141
long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 149
__int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 156
__int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 164
unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 171
unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 179
unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 186
unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 194
long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 200
long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 207
float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 213
float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 220
double __cdecl _wtof(const __wchar_t * _String); 
#line 225
double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 231
int __cdecl _wtoi(const __wchar_t * _String); 
#line 236
int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 242
long __cdecl _wtol(const __wchar_t * _String); 
#line 247
long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 253
__int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 258
__int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 264
errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 272
__wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 279
errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 287
__wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 294
__int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 299
__int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 305
__int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 312
__int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 320
unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 327
unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 339
__declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 348
errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 357
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wmakepath_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _Drive, wchar_t const * _Dir, wchar_t const * _Filename, wchar_t const * _Ext ) throw ( ) { return _wmakepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 375 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 380
void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 388
errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 400
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _wsplitpath_s ( wchar_t const * _Path, wchar_t ( & _Drive ) [ _DriveSize ], wchar_t ( & _Dir ) [ _DirSize ], wchar_t ( & _Name ) [ _NameSize ], wchar_t ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _wsplitpath_s ( _Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 418
__wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 424
errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wgetenv_s ( size_t * _RequiredCount, wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _VarName ) throw ( ) { return _wgetenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 440 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 445
errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 450
errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 457
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wsearchenv_s ( wchar_t const * _Filename, wchar_t const * _VarName, wchar_t ( & _ResultPath ) [ _Size ] ) throw ( ) { return _wsearchenv_s ( _Filename, _VarName, _ResultPath, _Size ); }}
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 471 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wsystem(const __wchar_t * _Command); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 481
#pragma warning(pop)
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 22
__pragma( pack ( push, 8 )) extern "C" {
#line 38 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 56
__declspec(noreturn) void __cdecl exit(int _Code); 
#line 57
__declspec(noreturn) void __cdecl _exit(int _Code); 
#line 58
__declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 59
__declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 60
__declspec(noreturn) void __cdecl abort(); 
#line 67 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 77
typedef int (__cdecl *_onexit_t)(void); 
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 145
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 159
typedef void (__cdecl *_purecall_handler)(void); 
#line 162
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 171
_purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 175
_purecall_handler __cdecl _get_purecall_handler(); 
#line 178
_invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 182
_invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 184
_invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 188
_invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 212 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int __cdecl _set_error_mode(int _Mode); 
#line 217
int *__cdecl _errno(); 
#line 220
errno_t __cdecl _set_errno(int _Value); 
#line 221
errno_t __cdecl _get_errno(int * _Value); 
#line 223
unsigned long *__cdecl __doserrno(); 
#line 226
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 227
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 230
char **__cdecl __sys_errlist(); 
#line 233
int *__cdecl __sys_nerr(); 
#line 236
void __cdecl perror(const char * _ErrMsg); 
#line 242 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char **__cdecl __p__pgmptr(); 
#line 243
__wchar_t **__cdecl __p__wpgmptr(); 
#line 244
int *__cdecl __p__fmode(); 
#line 259 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 262
errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 264
errno_t __cdecl _set_fmode(int _Mode); 
#line 266
errno_t __cdecl _get_fmode(int * _PMode); 
#line 279
typedef 
#line 275
struct _div_t { 
#line 277
int quot; 
#line 278
int rem; 
#line 279
} div_t; 
#line 285
typedef 
#line 281
struct _ldiv_t { 
#line 283
long quot; 
#line 284
long rem; 
#line 285
} ldiv_t; 
#line 291
typedef 
#line 287
struct _lldiv_t { 
#line 289
__int64 quot; 
#line 290
__int64 rem; 
#line 291
} lldiv_t; 
#line 293
int __cdecl abs(int _Number); 
#line 294
long __cdecl labs(long _Number); 
#line 295
__int64 __cdecl llabs(__int64 _Number); 
#line 296
__int64 __cdecl _abs64(__int64 _Number); 
#line 298
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 299
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 300
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 302
div_t __cdecl div(int _Numerator, int _Denominator); 
#line 303
ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 304
lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 308
#pragma warning(push)
#pragma warning(disable: 6540)
#line 311
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 317
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 322
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 327
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 333
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 338
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 343
#pragma warning(pop)
#line 350
void __cdecl srand(unsigned _Seed); 
#line 352
int __cdecl rand(); 
#line 361 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
extern "C++" {
#line 363
inline long abs(const long _X) throw() 
#line 364
{ 
#line 365
return labs(_X); 
#line 366
} 
#line 368
inline __int64 abs(const __int64 _X) throw() 
#line 369
{ 
#line 370
return llabs(_X); 
#line 371
} 
#line 373
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 374
{ 
#line 375
return ldiv(_A1, _A2); 
#line 376
} 
#line 378
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 379
{ 
#line 380
return lldiv(_A1, _A2); 
#line 381
} 
#line 382
}
#line 394 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 398
typedef 
#line 396
struct { 
#line 397
unsigned char ld[10]; 
#line 398
} _LDOUBLE; 
#pragma pack ( pop )
#line 418 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
typedef 
#line 416
struct { 
#line 417
double x; 
#line 418
} _CRT_DOUBLE; 
#line 423
typedef 
#line 421
struct { 
#line 422
float f; 
#line 423
} _CRT_FLOAT; 
#line 432
typedef 
#line 430
struct { 
#line 431
long double x; 
#line 432
} _LONGDOUBLE; 
#line 436
#pragma pack ( push, 4 )
#line 440
typedef 
#line 438
struct { 
#line 439
unsigned char ld12[12]; 
#line 440
} _LDBL12; 
#pragma pack ( pop )
#line 450
double __cdecl atof(const char * _String); 
#line 451
int __cdecl atoi(const char * _String); 
#line 452
long __cdecl atol(const char * _String); 
#line 453
__int64 __cdecl atoll(const char * _String); 
#line 454
__int64 __cdecl _atoi64(const char * _String); 
#line 456
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 457
int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 458
long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 459
__int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 460
__int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 462
int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 463
int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 464
int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 467
int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 474
int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 482
int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 489
float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 495
float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 502
double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 508
double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 515
long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 521
long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 528
long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 535
long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 543
__int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 550
__int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 558
unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 565
unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 573
unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 580
unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 588
__int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 595
__int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 603
unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 610
unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 626
errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 633
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itoa_s ( int _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 650 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 657
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltoa_s ( long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 664 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 673 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 680
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultoa_s ( unsigned long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 705
char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 713
errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 721
char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 741
errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 750
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ecvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _ecvt_s ( _Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign ); }}
#line 760 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 769
errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 778
extern "C++" {template < size_t _Size > inline errno_t __cdecl _fcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _fcvt_s ( _Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign ); }}
#line 790 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 798
errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 805
extern "C++" {template < size_t _Size > inline errno_t __cdecl _gcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount ) throw ( ) { return _gcvt_s ( _Buffer, _Size, _Value, _DigitCount ); }}
#line 814 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 852
int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 858
int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 866
size_t __cdecl _mbstrlen(const char * _String); 
#line 872
size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 879
size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 886
size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 893
int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 900
int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 908
errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 916
extern "C++" {template < size_t _Size > inline errno_t __cdecl mbstowcs_s ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount ) throw ( ) { return mbstowcs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 924 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 932 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 941
extern "C++" {template < size_t _Size > inline errno_t __cdecl _mbstowcs_s_l ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _mbstowcs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 950 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 963 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 969
int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 978
errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 988 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 996
errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 1004
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcstombs_s ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount ) throw ( ) { return wcstombs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 1012 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1020 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1029
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcstombs_s_l ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _wcstombs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 1038 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1068 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
__declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1077
errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1086
extern "C++" {template < size_t _Size > inline errno_t __cdecl _makepath_s ( char ( & _Buffer ) [ _Size ], char const * _Drive, char const * _Dir, char const * _Filename, char const * _Ext ) throw ( ) { return _makepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 1095 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1105 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1114
errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1126
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _splitpath_s ( char const * _Dest, char ( & _Drive ) [ _DriveSize ], char ( & _Dir ) [ _DirSize ], char ( & _Name ) [ _NameSize ], char ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _splitpath_s ( _Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 1132
errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int *__cdecl __p___argc(); 
#line 1145
char ***__cdecl __p___argv(); 
#line 1146
__wchar_t ***__cdecl __p___wargv(); 
#line 1158 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char ***__cdecl __p__environ(); 
#line 1159
__wchar_t ***__cdecl __p__wenviron(); 
#line 1184 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
char *__cdecl getenv(const char * _VarName); 
#line 1188
extern "C++" {template < size_t _Size > inline errno_t __cdecl getenv_s ( size_t * _RequiredCount, char ( & _Buffer ) [ _Size ], char const * _VarName ) throw ( ) { return getenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 1201 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1211 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
int __cdecl system(const char * _Command); 
#line 1217
#pragma warning(push)
#pragma warning(disable: 6540)
#line 1221
int __cdecl _putenv(const char * _EnvString); 
#line 1226
errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1231
#pragma warning(pop)
#line 1233
errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1240
extern "C++" {template < size_t _Size > inline errno_t __cdecl _searchenv_s ( char const * _Filename, char const * _VarName, char ( & _Buffer ) [ _Size ] ) throw ( ) { return _searchenv_s ( _Filename, _VarName, _Buffer, _Size ); }}
#line 1247 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) void __cdecl 
#line 1256
_seterrormode(int _Mode); 
#line 1260
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) void __cdecl 
#line 1261
_beep(unsigned _Frequency, unsigned _Duration); 
#line 1266
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) void __cdecl 
#line 1267
_sleep(unsigned long _Duration); 
#line 1289 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1293
char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1301
char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1309
char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1316
char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1323
char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1331
void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1338
char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1347
int __cdecl putenv(const char * _EnvString); 
#line 1351
#pragma warning(pop)
#line 1353
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1359 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\stdlib.h"
}__pragma( pack ( pop )) 
#line 1361
#pragma warning(pop)
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 191
extern cudaError_t __stdcall __cudaDeviceSynchronizeDeprecationAvoidance(); 
#line 244
extern cudaError_t __stdcall __cudaCDP2DeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 245
extern cudaError_t __stdcall __cudaCDP2DeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 246
extern cudaError_t __stdcall __cudaCDP2DeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 247
extern cudaError_t __stdcall __cudaCDP2DeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 248
extern cudaError_t __stdcall __cudaCDP2GetLastError(); 
#line 249
extern cudaError_t __stdcall __cudaCDP2PeekAtLastError(); 
#line 250
extern const char *__stdcall __cudaCDP2GetErrorString(cudaError_t error); 
#line 251
extern const char *__stdcall __cudaCDP2GetErrorName(cudaError_t error); 
#line 252
extern cudaError_t __stdcall __cudaCDP2GetDeviceCount(int * count); 
#line 253
extern cudaError_t __stdcall __cudaCDP2GetDevice(int * device); 
#line 254
extern cudaError_t __stdcall __cudaCDP2StreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 255
extern cudaError_t __stdcall __cudaCDP2StreamDestroy(cudaStream_t stream); 
#line 256
extern cudaError_t __stdcall __cudaCDP2StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 257
extern cudaError_t __stdcall __cudaCDP2StreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 258
extern cudaError_t __stdcall __cudaCDP2EventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 259
extern cudaError_t __stdcall __cudaCDP2EventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 260
extern cudaError_t __stdcall __cudaCDP2EventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 261
extern cudaError_t __stdcall __cudaCDP2EventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 262
extern cudaError_t __stdcall __cudaCDP2EventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 263
extern cudaError_t __stdcall __cudaCDP2EventDestroy(cudaEvent_t event); 
#line 264
extern cudaError_t __stdcall __cudaCDP2FuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 265
extern cudaError_t __stdcall __cudaCDP2Free(void * devPtr); 
#line 266
extern cudaError_t __stdcall __cudaCDP2Malloc(void ** devPtr, size_t size); 
#line 267
extern cudaError_t __stdcall __cudaCDP2MemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 268
extern cudaError_t __stdcall __cudaCDP2MemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 269
extern cudaError_t __stdcall __cudaCDP2Memcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 270
extern cudaError_t __stdcall __cudaCDP2Memcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 271
extern cudaError_t __stdcall __cudaCDP2Memcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 272
extern cudaError_t __stdcall __cudaCDP2Memcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 273
extern cudaError_t __stdcall __cudaCDP2MemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 274
extern cudaError_t __stdcall __cudaCDP2MemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 275
extern cudaError_t __stdcall __cudaCDP2Memset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 276
extern cudaError_t __stdcall __cudaCDP2Memset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 277
extern cudaError_t __stdcall __cudaCDP2Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 278
extern cudaError_t __stdcall __cudaCDP2Memset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 279
extern cudaError_t __stdcall __cudaCDP2RuntimeGetVersion(int * runtimeVersion); 
#line 280
extern void *__stdcall __cudaCDP2GetParameterBuffer(size_t alignment, size_t size); 
#line 281
extern void *__stdcall __cudaCDP2GetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 282
extern cudaError_t __stdcall __cudaCDP2LaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 283
extern cudaError_t __stdcall __cudaCDP2LaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 284
extern cudaError_t __stdcall __cudaCDP2LaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 285
extern cudaError_t __stdcall __cudaCDP2LaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 286
extern cudaError_t __stdcall __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 287
extern cudaError_t __stdcall __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 290
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static inline cudaGraphExec_t __stdcall cudaGetCurrentGraphExec() {int volatile ___ = 1;::exit(___);}
#if 0
#line 312
{ 
#line 313
unsigned __int64 current_graph_exec; 
#line 314
__asm mov.u64 %0, %%current_graph_exec;
return (cudaGraphExec_t)current_graph_exec; 
#line 316
} 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const void * value, size_t size); 
#line 374
extern cudaError_t __stdcall cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t node, bool enable); 
#line 401
extern cudaError_t __stdcall cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t node, dim3 gridDim); 
#line 430
extern cudaError_t __stdcall cudaGraphKernelNodeUpdatesApply(const cudaGraphKernelNodeUpdate * updates, size_t updateCount); 
#line 448
static inline void __stdcall cudaTriggerProgrammaticLaunchCompletion() {int volatile ___ = 1;::exit(___);}
#if 0
#line 449
{ 
#line 450
__asm griddepcontrol.launch_dependents;
} 
#endif
#line 464 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static inline void __stdcall cudaGridDependencySynchronize() {int volatile ___ = 1;::exit(___);}
#if 0
#line 465
{ 
#line 466
__asm griddepcontrol.wait;
} 
#endif
#line 480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
extern void __stdcall cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned value); 
#line 483
extern unsigned __int64 __stdcall cudaCGGetIntrinsicHandle(cudaCGScope scope); 
#line 484
extern cudaError_t __stdcall cudaCGSynchronize(unsigned __int64 handle, unsigned flags); 
#line 485
extern cudaError_t __stdcall cudaCGSynchronizeGrid(unsigned __int64 handle, unsigned flags); 
#line 486
extern cudaError_t __stdcall cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned __int64 handle); 
#line 487
extern cudaError_t __stdcall cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned __int64 handle); 
#line 715
static __inline void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size) {int volatile ___ = 1;(void)alignment;(void)size;::exit(___);}
#if 0
#line 716
{ 
#line 717
return __cudaCDP2GetParameterBuffer(alignment, size); 
#line 718
} 
#endif
#line 725 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static __inline void *__stdcall cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize) {int volatile ___ = 1;(void)func;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;::exit(___);}
#if 0
#line 726
{ 
#line 727
return __cudaCDP2GetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize); 
#line 728
} 
#endif
#line 735 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) {int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;::exit(___);}
#if 0
#line 736
{ 
#line 737
return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
#line 738
} 
#endif
#line 740 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream) {int volatile ___ = 1;(void)parameterBuffer;(void)stream;::exit(___);}
#if 0
#line 741
{ 
#line 742
return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream); 
#line 743
} 
#endif
#line 801 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) {int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;::exit(___);}
#if 0
#line 802
{ 
#line 803
return __cudaCDP2LaunchDevice(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
#line 804
} 
#endif
#line 806 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream) {int volatile ___ = 1;(void)parameterBuffer;(void)stream;::exit(___);}
#if 0
#line 807
{ 
#line 808
return __cudaCDP2LaunchDeviceV2(parameterBuffer, stream); 
#line 809
} 
#endif
#line 863 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_device_runtime_api.h"
}
#line 869
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 870
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 871
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 872
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 902
template< class T> static __inline ::cudaError_t __stdcall 
#line 903
cudaGraphKernelNodeSetParam(::cudaGraphDeviceNode_t node, ::size_t offset, const T &value) {int volatile ___ = 1;(void)node;(void)offset;(void)value;::exit(___);}
#if 0
#line 904
{ 
#line 905
return cudaGraphKernelNodeSetParam(node, offset, &value, sizeof(T)); 
#line 906
} 
#endif
#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern "C" {
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 352
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 438
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 474
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 497
extern cudaError_t __stdcall cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
#line 531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 568
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 612
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 639
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 669
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 720
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 764
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 809
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 876
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 915
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 947
extern cudaError_t __stdcall cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
#line 985 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void * userData, cudaAsyncCallbackHandle_t * callback); 
#line 1008
extern cudaError_t __stdcall cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback); 
#line 1055
__declspec(deprecated) extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 1101
__declspec(deprecated) extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 1165
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1216
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1232
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1248
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1277
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1299
extern cudaError_t __stdcall cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
#line 1324
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 1357
extern cudaError_t __stdcall cudaDeviceGetHostAtomicCapabilities(unsigned * capabilities, const cudaAtomicOperation * operations, unsigned count, int device); 
#line 1375
extern cudaError_t __stdcall cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
#line 1399
extern cudaError_t __stdcall cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
#line 1419
extern cudaError_t __stdcall cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
#line 1481
extern cudaError_t __stdcall cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
#line 1525
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 1561
extern cudaError_t __stdcall cudaDeviceGetP2PAtomicCapabilities(unsigned * capabilities, const cudaAtomicOperation * operations, unsigned count, int srcDevice, int dstDevice); 
#line 1584
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 1613
extern cudaError_t __stdcall cudaInitDevice(int device, unsigned deviceFlags, unsigned flags); 
#line 1659
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 1681
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 1712
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 1782
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 1827
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 1871
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 1907
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 1959
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 1987
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 2013
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2050
extern cudaError_t __stdcall cudaStreamGetId(cudaStream_t hStream, unsigned __int64 * streamId); 
#line 2076
extern cudaError_t __stdcall cudaStreamGetDevice(cudaStream_t hStream, int * device); 
#line 2091
extern cudaError_t __stdcall cudaCtxResetPersistingL2Cache(); 
#line 2111
extern cudaError_t __stdcall cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
#line 2132
extern cudaError_t __stdcall cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 2156
extern cudaError_t __stdcall cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 2190
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2221
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
#line 2229
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2296
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2320
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2345
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 2429
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 2468 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
#line 2509
extern cudaError_t __stdcall cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaStreamCaptureMode mode); 
#line 2560
extern cudaError_t __stdcall cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
#line 2589
extern cudaError_t __stdcall cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
#line 2627
extern cudaError_t __stdcall cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
#line 2686
extern cudaError_t __stdcall cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned __int64 * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, const cudaGraphEdgeData ** edgeData_out = 0, size_t * numDependencies_out = 0); 
#line 2724
extern cudaError_t __stdcall cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, unsigned flags = 0); 
#line 2761
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 2798
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 2839
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 2887
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
#line 2920 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 2951
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 2981
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 3029
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 3210
extern cudaError_t __stdcall cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
#line 3265
extern cudaError_t __stdcall cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
#line 3325
extern cudaError_t __stdcall cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
#line 3349
extern cudaError_t __stdcall cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
#line 3503
extern cudaError_t __stdcall cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
#line 3599
extern cudaError_t __stdcall cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3675
extern cudaError_t __stdcall cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3698
extern cudaError_t __stdcall cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
#line 3766
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 3829
extern cudaError_t __stdcall cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
#line 3886
extern cudaError_t __stdcall cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 3936
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 3971
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 4030
extern cudaError_t __stdcall cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
#line 4056
extern cudaError_t __stdcall cudaFuncGetName(const char ** name, const void * func); 
#line 4079
extern cudaError_t __stdcall cudaFuncGetParamInfo(const void * func, size_t paramIndex, size_t * paramOffset, size_t * paramSize); 
#line 4145
extern cudaError_t __stdcall cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
#line 4219
__declspec(deprecated) extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 4276
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 4306
extern cudaError_t __stdcall cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
#line 4352
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 4388
extern cudaError_t __stdcall cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 4428
extern cudaError_t __stdcall cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 4548
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 4581 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 4618
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 4661
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 4713
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 4752
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 4775
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 4798
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 4821
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 4887
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 4984
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 5007
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 5052
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 5074
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 5113
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 5258
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 5403
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 5436
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 5541
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 5573
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 5691
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 5718
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 5752
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 5778
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 5807
extern cudaError_t __stdcall cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
#line 5830
extern cudaError_t __stdcall cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
#line 5854
extern cudaError_t __stdcall cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
#line 5882
extern cudaError_t __stdcall cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
#line 5912 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
#line 5957 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 5992
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 6041
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6091
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6141
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6188
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 6231
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 6275
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 6332
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6367
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 6435
extern cudaError_t __stdcall cudaMemcpyBatchAsync(void *const * dsts, const void *const * srcs, const size_t * sizes, size_t count, cudaMemcpyAttributes * attrs, size_t * attrsIdxs, size_t numAttrs, cudaStream_t stream); 
#line 6499
extern cudaError_t __stdcall cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp * opList, unsigned __int64 flags, cudaStream_t stream); 
#line 6561
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6619
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6676
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6727
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6778
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6807
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 6841
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 6887
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 6923
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 6964
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 7017
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 7045
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 7072
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 7152
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, cudaMemLocation location, unsigned flags, cudaStream_t stream = 0); 
#line 7193
extern cudaError_t __stdcall cudaMemPrefetchBatchAsync(void ** dptrs, size_t * sizes, size_t count, cudaMemLocation * prefetchLocs, size_t * prefetchLocIdxs, size_t numPrefetchLocs, unsigned __int64 flags, cudaStream_t stream); 
#line 7226
extern cudaError_t __stdcall cudaMemDiscardBatchAsync(void ** dptrs, size_t * sizes, size_t count, unsigned __int64 flags, cudaStream_t stream); 
#line 7274
extern cudaError_t __stdcall cudaMemDiscardAndPrefetchBatchAsync(void ** dptrs, size_t * sizes, size_t count, cudaMemLocation * prefetchLocs, size_t * prefetchLocIdxs, size_t numPrefetchLocs, unsigned __int64 flags, cudaStream_t stream); 
#line 7399
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location); 
#line 7481
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 7524
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 7584
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 7626
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 7669
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 7720
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7770
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7839
extern cudaError_t __stdcall cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
#line 7865
extern cudaError_t __stdcall cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
#line 7890
extern cudaError_t __stdcall cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
#line 7934
extern cudaError_t __stdcall cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 7982
extern cudaError_t __stdcall cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 7997
extern cudaError_t __stdcall cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
#line 8010
extern cudaError_t __stdcall cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
#line 8061
extern cudaError_t __stdcall cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
#line 8083
extern cudaError_t __stdcall cudaMemPoolDestroy(cudaMemPool_t memPool); 
#line 8102
extern cudaError_t __stdcall cudaMemGetDefaultMemPool(cudaMemPool_t * memPool, cudaMemLocation * location, cudaMemAllocationType type); 
#line 8125
extern cudaError_t __stdcall cudaMemGetMemPool(cudaMemPool_t * memPool, cudaMemLocation * location, cudaMemAllocationType type); 
#line 8153
extern cudaError_t __stdcall cudaMemSetMemPool(cudaMemLocation * location, cudaMemAllocationType type, cudaMemPool_t memPool); 
#line 8189
extern cudaError_t __stdcall cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
#line 8214
extern cudaError_t __stdcall cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 8241
extern cudaError_t __stdcall cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 8264
extern cudaError_t __stdcall cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
#line 8293
extern cudaError_t __stdcall cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
#line 8446
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 8487
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 8529
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 8551
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 8615
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 8650
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 8689
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 8724
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 8756
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 8794
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 8823
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 8858
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 8888
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 9113
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 9133
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 9153
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 9173
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 9194
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 9239
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 9259
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 9278
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 9312
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 9341
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 9365
typedef void (__stdcall *cudaLogsCallback_t)(void * data, cudaLogLevel logLevel, char * message, size_t length); 
#line 9378
extern cudaError_t __stdcall cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void * userData, cudaLogsCallbackHandle * callback_out); 
#line 9389
extern cudaError_t __stdcall cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback); 
#line 9401
extern cudaError_t __stdcall cudaLogsCurrent(cudaLogIterator * iterator_out, unsigned flags); 
#line 9425
extern cudaError_t __stdcall cudaLogsDumpToFile(cudaLogIterator * iterator, const char * pathToFile, unsigned flags); 
#line 9461
extern cudaError_t __stdcall cudaLogsDumpToMemory(cudaLogIterator * iterator, char * buffer, size_t * size, unsigned flags); 
#line 9508
extern cudaError_t __stdcall cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
#line 9607
extern cudaError_t __stdcall cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
#line 9640
extern cudaError_t __stdcall cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
#line 9667
extern cudaError_t __stdcall cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 9687
extern cudaError_t __stdcall cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc); 
#line 9710
extern cudaError_t __stdcall cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 9734
extern cudaError_t __stdcall cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 9785
extern cudaError_t __stdcall cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
#line 9844
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9913 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9981 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10013 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
#line 10040
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 10079
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
#line 10242
extern cudaError_t __stdcall cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
#line 10266
extern cudaError_t __stdcall cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 10308
extern cudaError_t __stdcall cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
#line 10331
extern cudaError_t __stdcall cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
#line 10355
extern cudaError_t __stdcall cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 10397
extern cudaError_t __stdcall cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
#line 10424
extern cudaError_t __stdcall cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
#line 10462
extern cudaError_t __stdcall cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
#line 10506
extern cudaError_t __stdcall cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 10533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 10561 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 10608 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 10635 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 10663 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 10713 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 10746 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
#line 10774 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 10824 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 10857 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
#line 10885 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 10963 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
#line 10990 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
#line 11051 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
#line 11075 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
#line 11103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGraphMemTrim(int device); 
#line 11140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
#line 11233
extern cudaError_t __stdcall cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
#line 11264
extern cudaError_t __stdcall cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
#line 11295
extern cudaError_t __stdcall cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
#line 11326
extern cudaError_t __stdcall cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
#line 11366
extern cudaError_t __stdcall cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, cudaGraphEdgeData * edgeData, size_t * numEdges); 
#line 11403
extern cudaError_t __stdcall cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, cudaGraphEdgeData * edgeData, size_t * pNumDependencies); 
#line 11441
extern cudaError_t __stdcall cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, cudaGraphEdgeData * edgeData, size_t * pNumDependentNodes); 
#line 11473
extern cudaError_t __stdcall cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
#line 11508
extern cudaError_t __stdcall cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
#line 11538
extern cudaError_t __stdcall cudaGraphDestroyNode(cudaGraphNode_t node); 
#line 11609
extern cudaError_t __stdcall cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned __int64 flags = 0); 
#line 11682
extern cudaError_t __stdcall cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned __int64 flags = 0); 
#line 11789 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams * instantiateParams); 
#line 11814
extern cudaError_t __stdcall cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned __int64 * flags); 
#line 11874
extern cudaError_t __stdcall cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 11925
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 11980
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 12043 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 12104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 12163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 12203
extern cudaError_t __stdcall cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 12250
extern cudaError_t __stdcall cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
#line 12295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 12340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 12388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 12436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 12476 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
#line 12510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
#line 12604 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo * resultInfo); 
#line 12629
extern cudaError_t __stdcall cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 12660 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 12683
extern cudaError_t __stdcall cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
#line 12704
extern cudaError_t __stdcall cudaGraphDestroy(cudaGraph_t graph); 
#line 12723
extern cudaError_t __stdcall cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
#line 12759
extern cudaError_t __stdcall cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
#line 12783
extern cudaError_t __stdcall cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
#line 12811
extern cudaError_t __stdcall cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
#line 12839
extern cudaError_t __stdcall cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
#line 12864
extern cudaError_t __stdcall cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
#line 12908
extern cudaError_t __stdcall cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaGraphNodeParams * nodeParams); 
#line 12937
extern cudaError_t __stdcall cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
#line 12986
extern cudaError_t __stdcall cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
#line 13013
extern cudaError_t __stdcall cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle * pHandle_out, cudaGraph_t graph, unsigned defaultLaunchValue = 0, unsigned flags = 0); 
#line 13101
__declspec(deprecated) extern cudaError_t __stdcall cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned __int64 flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
#line 13178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGetDriverEntryPointByVersion(const char * symbol, void ** funcPtr, unsigned cudaVersion, unsigned __int64 flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
#line 13253 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaLibraryLoadData(cudaLibrary_t * library, const void * code, cudaJitOption * jitOptions, void ** jitOptionsValues, unsigned numJitOptions, cudaLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned numLibraryOptions); 
#line 13313
extern cudaError_t __stdcall cudaLibraryLoadFromFile(cudaLibrary_t * library, const char * fileName, cudaJitOption * jitOptions, void ** jitOptionsValues, unsigned numJitOptions, cudaLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned numLibraryOptions); 
#line 13334
extern cudaError_t __stdcall cudaLibraryUnload(cudaLibrary_t library); 
#line 13359
extern cudaError_t __stdcall cudaLibraryGetKernel(cudaKernel_t * pKernel, cudaLibrary_t library, const char * name); 
#line 13393
extern cudaError_t __stdcall cudaLibraryGetGlobal(void ** dptr, size_t * bytes, cudaLibrary_t library, const char * name); 
#line 13426
extern cudaError_t __stdcall cudaLibraryGetManaged(void ** dptr, size_t * bytes, cudaLibrary_t library, const char * name); 
#line 13453
extern cudaError_t __stdcall cudaLibraryGetUnifiedFunction(void ** fptr, cudaLibrary_t library, const char * symbol); 
#line 13475
extern cudaError_t __stdcall cudaLibraryGetKernelCount(unsigned * count, cudaLibrary_t lib); 
#line 13497
extern cudaError_t __stdcall cudaLibraryEnumerateKernels(cudaKernel_t * kernels, unsigned numKernels, cudaLibrary_t lib); 
#line 13566
extern cudaError_t __stdcall cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device); 
#line 13571
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 13757
extern cudaError_t __stdcall cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
#line 13781
extern cudaError_t __stdcall cudaGetKernel(cudaKernel_t * kernelPtr, const void * entryFuncAddr); 
#line 13948 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\cuda_runtime_api.h"
}
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 121
{ 
#line 122
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 123
} 
#line 125
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 126
{ 
#line 127
int e = (((int)sizeof(unsigned short)) * 8); 
#line 129
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 130
} 
#line 132
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 133
{ 
#line 134
int e = (((int)sizeof(unsigned short)) * 8); 
#line 136
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 137
} 
#line 139
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 140
{ 
#line 141
int e = (((int)sizeof(unsigned short)) * 8); 
#line 143
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 144
} 
#line 146
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 147
{ 
#line 148
int e = (((int)sizeof(unsigned short)) * 8); 
#line 150
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 151
} 
#line 153
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 154
{ 
#line 155
int e = (((int)sizeof(char)) * 8); 
#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\channel_descriptor.h"
} 
#line 164
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 165
{ 
#line 166
int e = (((int)sizeof(signed char)) * 8); 
#line 168
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 169
} 
#line 171
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 172
{ 
#line 173
int e = (((int)sizeof(unsigned char)) * 8); 
#line 175
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 176
} 
#line 178
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 179
{ 
#line 180
int e = (((int)sizeof(signed char)) * 8); 
#line 182
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 183
} 
#line 185
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 186
{ 
#line 187
int e = (((int)sizeof(unsigned char)) * 8); 
#line 189
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 190
} 
#line 192
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 193
{ 
#line 194
int e = (((int)sizeof(signed char)) * 8); 
#line 196
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 197
} 
#line 199
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 200
{ 
#line 201
int e = (((int)sizeof(unsigned char)) * 8); 
#line 203
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 204
} 
#line 206
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 207
{ 
#line 208
int e = (((int)sizeof(signed char)) * 8); 
#line 210
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 211
} 
#line 213
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 214
{ 
#line 215
int e = (((int)sizeof(unsigned char)) * 8); 
#line 217
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 218
} 
#line 220
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 221
{ 
#line 222
int e = (((int)sizeof(short)) * 8); 
#line 224
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 225
} 
#line 227
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 228
{ 
#line 229
int e = (((int)sizeof(unsigned short)) * 8); 
#line 231
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 232
} 
#line 234
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 235
{ 
#line 236
int e = (((int)sizeof(short)) * 8); 
#line 238
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 239
} 
#line 241
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 242
{ 
#line 243
int e = (((int)sizeof(unsigned short)) * 8); 
#line 245
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 246
} 
#line 248
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 249
{ 
#line 250
int e = (((int)sizeof(short)) * 8); 
#line 252
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 253
} 
#line 255
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 256
{ 
#line 257
int e = (((int)sizeof(unsigned short)) * 8); 
#line 259
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 260
} 
#line 262
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 263
{ 
#line 264
int e = (((int)sizeof(short)) * 8); 
#line 266
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 267
} 
#line 269
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 270
{ 
#line 271
int e = (((int)sizeof(unsigned short)) * 8); 
#line 273
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 274
} 
#line 276
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 277
{ 
#line 278
int e = (((int)sizeof(int)) * 8); 
#line 280
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 281
} 
#line 283
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 284
{ 
#line 285
int e = (((int)sizeof(unsigned)) * 8); 
#line 287
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 288
} 
#line 290
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 291
{ 
#line 292
int e = (((int)sizeof(int)) * 8); 
#line 294
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 295
} 
#line 297
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 298
{ 
#line 299
int e = (((int)sizeof(unsigned)) * 8); 
#line 301
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 302
} 
#line 304
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 305
{ 
#line 306
int e = (((int)sizeof(int)) * 8); 
#line 308
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 309
} 
#line 311
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 312
{ 
#line 313
int e = (((int)sizeof(unsigned)) * 8); 
#line 315
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 316
} 
#line 318
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 319
{ 
#line 320
int e = (((int)sizeof(int)) * 8); 
#line 322
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 323
} 
#line 325
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 326
{ 
#line 327
int e = (((int)sizeof(unsigned)) * 8); 
#line 329
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 330
} 
#line 334
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 335
{ 
#line 336
int e = (((int)sizeof(long)) * 8); 
#line 338
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 339
} 
#line 341
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 342
{ 
#line 343
int e = (((int)sizeof(unsigned long)) * 8); 
#line 345
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 346
} 
#line 348
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 349
{ 
#line 350
int e = (((int)sizeof(long)) * 8); 
#line 352
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 353
} 
#line 355
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 356
{ 
#line 357
int e = (((int)sizeof(unsigned long)) * 8); 
#line 359
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 360
} 
#line 362
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 363
{ 
#line 364
int e = (((int)sizeof(long)) * 8); 
#line 366
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 367
} 
#line 369
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 370
{ 
#line 371
int e = (((int)sizeof(unsigned long)) * 8); 
#line 373
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 374
} 
#line 376
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 377
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 378
{ 
#line 379
int e = (((int)sizeof(long)) * 8); 
#line 381
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 382
} 
#line 384
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 385
{ 
#line 386
int e = (((int)sizeof(unsigned long)) * 8); 
#line 388
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 389
} 
#line 390
__pragma( warning(pop)) 
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 395
{ 
#line 396
int e = (((int)sizeof(float)) * 8); 
#line 398
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 399
} 
#line 401
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 402
{ 
#line 403
int e = (((int)sizeof(float)) * 8); 
#line 405
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 406
} 
#line 408
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 409
{ 
#line 410
int e = (((int)sizeof(float)) * 8); 
#line 412
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 413
} 
#line 415
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 416
{ 
#line 417
int e = (((int)sizeof(float)) * 8); 
#line 419
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 420
} 
#line 422
static __inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
#line 423
{ 
#line 424
int e = (((int)sizeof(char)) * 8); 
#line 426
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
#line 427
} 
#line 429
template< cudaChannelFormatKind > __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 430
{ 
#line 431
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 432
} 
#line 435
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
#line 436
{ 
#line 437
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
#line 438
} 
#line 440
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
#line 441
{ 
#line 442
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
#line 443
} 
#line 445
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
#line 446
{ 
#line 447
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
#line 448
} 
#line 451
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
#line 452
{ 
#line 453
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
#line 454
} 
#line 456
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
#line 457
{ 
#line 458
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
#line 459
} 
#line 461
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
#line 462
{ 
#line 463
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
#line 464
} 
#line 467
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
#line 468
{ 
#line 469
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
#line 470
} 
#line 472
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
#line 473
{ 
#line 474
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
#line 475
} 
#line 477
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
#line 478
{ 
#line 479
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
#line 480
} 
#line 483
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
#line 484
{ 
#line 485
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
#line 486
} 
#line 488
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
#line 489
{ 
#line 490
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
#line 491
} 
#line 493
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
#line 494
{ 
#line 495
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
#line 496
} 
#line 499
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
#line 500
{ 
#line 501
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
#line 502
} 
#line 505
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized1010102> () 
#line 506
{ 
#line 507
return cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102); 
#line 508
} 
#line 511
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
#line 512
{ 
#line 513
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
#line 514
} 
#line 517
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
#line 518
{ 
#line 519
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
#line 520
} 
#line 523
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
#line 524
{ 
#line 525
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
#line 526
} 
#line 529
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
#line 530
{ 
#line 531
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
#line 532
} 
#line 535
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
#line 536
{ 
#line 537
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
#line 538
} 
#line 541
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
#line 542
{ 
#line 543
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
#line 544
} 
#line 547
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
#line 548
{ 
#line 549
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
#line 550
} 
#line 553
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
#line 554
{ 
#line 555
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
#line 556
} 
#line 559
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
#line 560
{ 
#line 561
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
#line 562
} 
#line 565
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
#line 566
{ 
#line 567
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
#line 568
} 
#line 571
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
#line 572
{ 
#line 573
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
#line 574
} 
#line 577
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
#line 578
{ 
#line 579
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
#line 580
} 
#line 583
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
#line 584
{ 
#line 585
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
#line 586
} 
#line 589
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
#line 590
{ 
#line 591
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
#line 592
} 
#line 79 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 77 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_functions.h"
static __inline char1 make_char1(signed char x); 
#line 79
static __inline uchar1 make_uchar1(unsigned char x); 
#line 81
static __inline char2 make_char2(signed char x, signed char y); 
#line 83
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
#line 85
static __inline char3 make_char3(signed char x, signed char y, signed char z); 
#line 87
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
#line 89
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
#line 91
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
#line 93
static __inline short1 make_short1(short x); 
#line 95
static __inline ushort1 make_ushort1(unsigned short x); 
#line 97
static __inline short2 make_short2(short x, short y); 
#line 99
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
#line 101
static __inline short3 make_short3(short x, short y, short z); 
#line 103
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
#line 105
static __inline short4 make_short4(short x, short y, short z, short w); 
#line 107
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
#line 109
static __inline int1 make_int1(int x); 
#line 111
static __inline uint1 make_uint1(unsigned x); 
#line 113
static __inline int2 make_int2(int x, int y); 
#line 115
static __inline uint2 make_uint2(unsigned x, unsigned y); 
#line 117
static __inline int3 make_int3(int x, int y, int z); 
#line 119
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
#line 121
static __inline int4 make_int4(int x, int y, int z, int w); 
#line 123
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
#line 125
static __inline long1 make_long1(long x); 
#line 127
static __inline ulong1 make_ulong1(unsigned long x); 
#line 129
static __inline long2 make_long2(long x, long y); 
#line 131
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
#line 133
static __inline long3 make_long3(long x, long y, long z); 
#line 135
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
#line 137
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 138
static __inline long4 make_long4(long x, long y, long z, long w); 
#line 140
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 141
__pragma( warning(pop)) 
#line 143
static __inline long4_16a make_long4_16a(long x, long y, long z, long w); 
#line 145
static __inline long4_32a make_long4_32a(long x, long y, long z, long w); 
#line 147
static __inline ulong4_16a make_ulong4_16a(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 149
static __inline ulong4_32a make_ulong4_32a(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 151
static __inline float1 make_float1(float x); 
#line 153
static __inline float2 make_float2(float x, float y); 
#line 155
static __inline float3 make_float3(float x, float y, float z); 
#line 157
static __inline float4 make_float4(float x, float y, float z, float w); 
#line 159
static __inline longlong1 make_longlong1(__int64 x); 
#line 161
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x); 
#line 163
static __inline longlong2 make_longlong2(__int64 x, __int64 y); 
#line 165
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y); 
#line 167
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z); 
#line 169
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z); 
#line 171
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 172
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w); 
#line 174
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w); 
#line 175
__pragma( warning(pop)) 
#line 177
static __inline double1 make_double1(double x); 
#line 179
static __inline double2 make_double2(double x, double y); 
#line 181
static __inline double3 make_double3(double x, double y, double z); 
#line 183
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 184
static __inline double4 make_double4(double x, double y, double z, double w); 
#line 185
__pragma( warning(pop)) 
#line 187
static __inline double4_16a make_double4_16a(double x, double y, double z, double w); 
#line 189
static __inline double4_32a make_double4_32a(double x, double y, double z, double w); 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 74
{ 
#line 75
char1 t; (t.x) = x; return t; 
#line 76
} 
#line 78
static __inline uchar1 make_uchar1(unsigned char x) 
#line 79
{ 
#line 80
uchar1 t; (t.x) = x; return t; 
#line 81
} 
#line 83
static __inline char2 make_char2(signed char x, signed char y) 
#line 84
{ 
#line 85
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 86
} 
#line 88
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 89
{ 
#line 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 91
} 
#line 93
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 94
{ 
#line 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 96
} 
#line 98
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 99
{ 
#line 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 101
} 
#line 103
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 104
{ 
#line 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 106
} 
#line 108
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 109
{ 
#line 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 111
} 
#line 113
static __inline short1 make_short1(short x) 
#line 114
{ 
#line 115
short1 t; (t.x) = x; return t; 
#line 116
} 
#line 118
static __inline ushort1 make_ushort1(unsigned short x) 
#line 119
{ 
#line 120
ushort1 t; (t.x) = x; return t; 
#line 121
} 
#line 123
static __inline short2 make_short2(short x, short y) 
#line 124
{ 
#line 125
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 126
} 
#line 128
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 129
{ 
#line 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 131
} 
#line 133
static __inline short3 make_short3(short x, short y, short z) 
#line 134
{ 
#line 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 136
} 
#line 138
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 139
{ 
#line 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 141
} 
#line 143
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 144
{ 
#line 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 146
} 
#line 148
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 149
{ 
#line 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 151
} 
#line 153
static __inline int1 make_int1(int x) 
#line 154
{ 
#line 155
int1 t; (t.x) = x; return t; 
#line 156
} 
#line 158
static __inline uint1 make_uint1(unsigned x) 
#line 159
{ 
#line 160
uint1 t; (t.x) = x; return t; 
#line 161
} 
#line 163
static __inline int2 make_int2(int x, int y) 
#line 164
{ 
#line 165
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 166
} 
#line 168
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 169
{ 
#line 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 171
} 
#line 173
static __inline int3 make_int3(int x, int y, int z) 
#line 174
{ 
#line 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 176
} 
#line 178
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 179
{ 
#line 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 181
} 
#line 183
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 184
{ 
#line 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 186
} 
#line 188
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 189
{ 
#line 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 191
} 
#line 193
static __inline long1 make_long1(long x) 
#line 194
{ 
#line 195
long1 t; (t.x) = x; return t; 
#line 196
} 
#line 198
static __inline ulong1 make_ulong1(unsigned long x) 
#line 199
{ 
#line 200
ulong1 t; (t.x) = x; return t; 
#line 201
} 
#line 203
static __inline long2 make_long2(long x, long y) 
#line 204
{ 
#line 205
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 206
} 
#line 208
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 209
{ 
#line 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 211
} 
#line 213
static __inline long3 make_long3(long x, long y, long z) 
#line 214
{ 
#line 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 216
} 
#line 218
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 219
{ 
#line 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 221
} 
#line 223
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 224
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 225
{ 
#line 226
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 227
} 
#line 229
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 230
{ 
#line 231
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 232
} 
#line 233
__pragma( warning(pop)) 
#line 235
static __inline long4_16a make_long4_16a(long x, long y, long z, long w) 
#line 236
{ 
#line 237
long4_16a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 238
} 
#line 240
static __inline long4_32a make_long4_32a(long x, long y, long z, long w) 
#line 241
{ 
#line 242
long4_32a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 243
} 
#line 245
static __inline ulong4_16a make_ulong4_16a(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 246
{ 
#line 247
ulong4_16a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 248
} 
#line 250
static __inline ulong4_32a make_ulong4_32a(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 251
{ 
#line 252
ulong4_32a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 253
} 
#line 255
static __inline float1 make_float1(float x) 
#line 256
{ 
#line 257
float1 t; (t.x) = x; return t; 
#line 258
} 
#line 260
static __inline float2 make_float2(float x, float y) 
#line 261
{ 
#line 262
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 263
} 
#line 265
static __inline float3 make_float3(float x, float y, float z) 
#line 266
{ 
#line 267
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 268
} 
#line 270
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 271
{ 
#line 272
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 273
} 
#line 275
static __inline longlong1 make_longlong1(__int64 x) 
#line 276
{ 
#line 277
longlong1 t; (t.x) = x; return t; 
#line 278
} 
#line 280
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 281
{ 
#line 282
ulonglong1 t; (t.x) = x; return t; 
#line 283
} 
#line 285
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 286
{ 
#line 287
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 288
} 
#line 290
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 291
{ 
#line 292
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 293
} 
#line 295
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 296
{ 
#line 297
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 298
} 
#line 300
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 301
{ 
#line 302
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 303
} 
#line 305
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 306
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 307
{ 
#line 308
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 309
} 
#line 311
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 312
{ 
#line 313
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 314
} 
#line 315
__pragma( warning(pop)) 
#line 317
static __inline longlong4_16a make_longlong4_16a(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 318
{ 
#line 319
longlong4_16a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 320
} 
#line 322
static __inline longlong4_32a make_longlong4_32a(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 323
{ 
#line 324
longlong4_32a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 325
} 
#line 327
static __inline ulonglong4_16a make_ulonglong4_16a(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 328
{ 
#line 329
ulonglong4_16a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 330
} 
#line 332
static __inline ulonglong4_32a make_ulonglong4_32a(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 333
{ 
#line 334
ulonglong4_32a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 335
} 
#line 336
static __inline double1 make_double1(double x) 
#line 337
{ 
#line 338
double1 t; (t.x) = x; return t; 
#line 339
} 
#line 341
static __inline double2 make_double2(double x, double y) 
#line 342
{ 
#line 343
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 344
} 
#line 346
static __inline double3 make_double3(double x, double y, double z) 
#line 347
{ 
#line 348
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 349
} 
#line 351
__pragma( warning(push)) __pragma( warning(disable:4996)) 
#line 352
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 353
{ 
#line 354
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 355
} 
#line 356
__pragma( warning(pop)) 
#line 358
static __inline double4_16a make_double4_16a(double x, double y, double z, double w) 
#line 359
{ 
#line 360
double4_16a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 361
} 
#line 363
static __inline double4_32a make_double4_32a(double x, double y, double z, double w) 
#line 364
{ 
#line 365
double4_32a t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 366
} 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\errno.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
int *__cdecl _errno(); 
#line 26
errno_t __cdecl _set_errno(int _Value); 
#line 27
errno_t __cdecl _get_errno(int * _Value); 
#line 29
unsigned long *__cdecl __doserrno(); 
#line 32
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 33
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 134 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 136
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime_string.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 21
[[nodiscard]] const void *__cdecl 
#line 22
memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 28
[[nodiscard]] int __cdecl 
#line 29
memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 43 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 50
void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 63 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 69
[[nodiscard]] const char *__cdecl 
#line 70
strchr(const char * _Str, int _Val); 
#line 75
[[nodiscard]] const char *__cdecl 
#line 76
strrchr(const char * _Str, int _Ch); 
#line 81
[[nodiscard]] const char *__cdecl 
#line 82
strstr(const char * _Str, const char * _SubStr); 
#line 87
[[nodiscard]] const __wchar_t *__cdecl 
#line 89
wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 94
[[nodiscard]] const __wchar_t *__cdecl 
#line 95
wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 100
[[nodiscard]] const __wchar_t *__cdecl 
#line 102
wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 109
}__pragma( pack ( pop )) 
#line 113 "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC\\Tools\\MSVC\\14.51.36231\\include\\vcruntime_string.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memcpy_s.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 40 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 41
_Destination, const rsize_t 
#line 42
_DestinationSize, const void *const 
#line 43
_Source, const rsize_t 
#line 44
_SourceSize) 
#line 46
{ 
#line 47
if (_SourceSize == (0)) 
#line 48
{ 
#line 49
return 0; 
#line 50
}  
#line 52
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 53
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 54
{ 
#line 55
memset(_Destination, 0, _DestinationSize); 
#line 57
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 58
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 61
return 22; 
#line 62
}  
#line 63
memcpy(_Destination, _Source, _SourceSize); 
#line 64
return 0; 
#line 65
} 
#line 69
static __inline errno_t __cdecl memmove_s(void *const 
#line 70
_Destination, const rsize_t 
#line 71
_DestinationSize, const void *const 
#line 72
_Source, const rsize_t 
#line 73
_SourceSize) 
#line 75
{ 
#line 76
if (_SourceSize == (0)) 
#line 77
{ 
#line 78
return 0; 
#line 79
}  
#line 81
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 82
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 83
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 85
memmove(_Destination, _Source, _SourceSize); 
#line 86
return 0; 
#line 87
} 
#line 95 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memcpy_s.h"
}
#line 94
#pragma warning(pop)
__pragma( pack ( pop )) 
#line 17 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memory.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 28
int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 35
int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memory.h"
void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 91
int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 104 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 105
_Pv, int 
#line 106
_C, size_t 
#line 107
_N) 
#line 109
{ 
#line 110
const void *const _Pvc = _Pv; 
#line 111
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 112
} 
#line 118 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 122 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_memory.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 39
errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 46
errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 54
errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 62
__wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscat_s ( _Destination, _Size, _Source ); }}
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 113
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscpy_s ( _Destination, _Size, _Source ); }}
#line 119 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 126 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 132
size_t __cdecl wcslen(const __wchar_t * _String); 
#line 145 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 161 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 162
_Source, size_t 
#line 163
_MaxCount) 
#line 165
{ 
#line 166
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 167
} 
#line 171 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 187 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 193
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 200 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 209 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 215
size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 221
__wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 240 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 241
_String, const __wchar_t *const 
#line 242
_Delimiter) 
#line 244
{ 
#line 245
return wcstok(_String, _Delimiter, 0); 
#line 246
} 
#line 254 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 253
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 254
wcstok(__wchar_t *
#line 255
_String, const __wchar_t *
#line 256
_Delimiter) throw() 
#line 258
{ 
#line 259
return wcstok(_String, _Delimiter, 0); 
#line 260
} 
#line 269 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 274
errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 280
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], int _Error ) throw ( ) { return _wcserror_s ( _Buffer, _Size, _Error ); }}
#line 289 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 293
errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 299
extern "C++" {template < size_t _Size > inline errno_t __cdecl __wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _ErrorMessage ) throw ( ) { return __wcserror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 305 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 310
int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 316
int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 322
int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 329
errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 336
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsnset_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t _Value, size_t _MaxCount ) throw ( ) { return _wcsnset_s ( _Destination, _Size, _Value, _MaxCount ); }}
#line 343 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 351 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 355
errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 361
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsset_s ( wchar_t ( & _String ) [ _Size ], wchar_t _Value ) throw ( ) { return _wcsset_s ( _String, _Size, _Value ); }}
#line 367 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 374 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 379
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcslwr_s ( _String, _Size ); }}
#line 384 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 390 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 396
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcslwr_s_l ( _String, _Size, _Locale ); }}
#line 402 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 410 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 415
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcsupr_s ( _String, _Size ); }}
#line 420 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 426 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 432
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcsupr_s_l ( _String, _Size, _Locale ); }}
#line 438 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 447 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 455
size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 463
int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 469
int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 476
int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 482
int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 489
int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 496
int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 504
int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 511
int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 526
extern "C++" {
#line 530
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 531
{ 
#line 532
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 533
} 
#line 536
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 537
{ 
#line 538
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 539
} 
#line 542
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 543
{ 
#line 544
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 545
} 
#line 549
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 550
{ 
#line 551
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 552
} 
#line 554
}
#line 571 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 583 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 589
int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 597
__wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 605
__wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 611
__wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 618
__wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 624
__wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 629
int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 638 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 642 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 39
errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 46
errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 52
errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 60
errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 68
char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 76 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 83
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcat_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcat_s ( _Destination, _Size, _Source ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 106
int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 112
int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 118
int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 124
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcpy_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcpy_s ( _Destination, _Size, _Source ); }}
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
__declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strerror(const char * _ErrorMessage); 
#line 164
errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 170
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strerror_s ( char ( & _Buffer ) [ _Size ], char const * _ErrorMessage ) throw ( ) { return _strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strerror(int _ErrorMessage); 
#line 182
extern "C++" {template < size_t _Size > inline errno_t __cdecl strerror_s ( char ( & _Buffer ) [ _Size ], int _ErrorMessage ) throw ( ) { return strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 189 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 195
int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 201
int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 208
int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 215
size_t __cdecl strlen(const char * _Str); 
#line 220
errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 225
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strlwr_s ( _String, _Size ); }}
#line 230 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strlwr(char * _String); 
#line 236 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 242
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strlwr_s_l ( _String, _Size, _Locale ); }}
#line 248 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncat_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 262 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 271 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 278
int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 285
int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 293
int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 300
int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 308
int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 315
int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 322
size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 327
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncpy_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 334 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 351 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 367 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 368
_String, size_t 
#line 369
_MaxCount) 
#line 371
{ 
#line 372
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 373
} 
#line 378 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 385
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strnset_s ( char ( & _Destination ) [ _Size ], int _Value, size_t _Count ) throw ( ) { return _strnset_s ( _Destination, _Size, _Value, _Count ); }}
#line 392 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 401 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 406
char *__cdecl _strrev(char * _Str); 
#line 411
errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 417
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strset_s ( char ( & _Destination ) [ _Size ], int _Value ) throw ( ) { return _strset_s ( _Destination, _Size, _Value ); }}
#line 423 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 430 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 436
char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 442
errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 447
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strupr_s ( _String, _Size ); }}
#line 452 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strupr(char * _String); 
#line 458 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 464
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strupr_s_l ( _String, _Size, _Locale ); }}
#line 470 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 487
size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 497
extern "C++" {
#line 500
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 501
{ 
#line 502
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 503
} 
#line 506
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 507
{ 
#line 508
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 509
} 
#line 512
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 513
{ 
#line 514
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 515
} 
#line 518
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 519
{ 
#line 520
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 521
} 
#line 522
}
#line 532 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
char *__cdecl strdup(const char * _String); 
#line 539
int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 545
int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 551
char *__cdecl strlwr(char * _String); 
#line 556
int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 563
char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 570
char *__cdecl strrev(char * _String); 
#line 575
char *__cdecl strset(char * _String, int _Value); 
#line 580
char *__cdecl strupr(char * _String); 
#line 588 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 590
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 26
struct tm { 
#line 28
int tm_sec; 
#line 29
int tm_min; 
#line 30
int tm_hour; 
#line 31
int tm_mday; 
#line 32
int tm_mon; 
#line 33
int tm_year; 
#line 34
int tm_wday; 
#line 35
int tm_yday; 
#line 36
int tm_isdst; 
#line 37
}; 
#line 48
__wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 54
errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 60
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wasctime_s ( wchar_t ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return _wasctime_s ( _Buffer, _Size, _Time ); }}
#line 69 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 78
size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 88
__wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 93
errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 99
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime32_s ( wchar_t ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _wctime32_s ( _Buffer, _Size, _Time ); }}
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 113
errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 118
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime64_s ( wchar_t ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _wctime64_s ( _Buffer, _Size, _Time ); }}
#line 125 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 130
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrdate_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrdate_s ( _Buffer, _Size ); }}
#line 135 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 141 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 146
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrtime_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrtime_s ( _Buffer, _Size ); }}
#line 151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 186 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 187
_Time) 
#line 188
{ 
#line 189
return _wctime64(_Time); 
#line 190
} 
#line 193
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 194
_Buffer, const size_t 
#line 195
_SizeInWords, const time_t *const 
#line 196
_Time) 
#line 198
{ 
#line 199
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 200
} 
#line 205 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_wtime.h"
}__pragma( pack ( pop )) 
#line 207
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 30 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
typedef long clock_t; 
#line 32
struct _timespec32 { 
#line 34
__time32_t tv_sec; 
#line 35
long tv_nsec; 
#line 36
}; 
#line 38
struct _timespec64 { 
#line 40
__time64_t tv_sec; 
#line 41
long tv_nsec; 
#line 42
}; 
#line 45
struct timespec { 
#line 47
time_t tv_sec; 
#line 48
long tv_nsec; 
#line 49
}; 
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
int *__cdecl __daylight(); 
#line 74
long *__cdecl __dstbias(); 
#line 80
long *__cdecl __timezone(); 
#line 86
char **__cdecl __tzname(); 
#line 91
errno_t __cdecl _get_daylight(int * _Daylight); 
#line 96
errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 101
errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 106
errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 123
char *__cdecl asctime(const tm * _Tm); 
#line 130
errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl asctime_s ( char ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return asctime_s ( _Buffer, _Size, _Time ); }}
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
clock_t __cdecl clock(); 
#line 149
char *__cdecl _ctime32(const __time32_t * _Time); 
#line 154
errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 160
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime32_s ( char ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _ctime32_s ( _Buffer, _Size, _Time ); }}
#line 169 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
char *__cdecl _ctime64(const __time64_t * _Time); 
#line 174
errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 180
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime64_s ( char ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _ctime64_s ( _Buffer, _Size, _Time ); }}
#line 187 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 193
double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 200
tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 205
errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 212
tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 217
errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 224
tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 229
errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 236
tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 241
errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 247
__time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 252
__time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 257
__time32_t __cdecl _mktime32(tm * _Tm); 
#line 262
__time64_t __cdecl _mktime64(tm * _Tm); 
#line 268
size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 277
size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 286
errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 291
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strdate_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strdate_s ( _Buffer, _Size ); }}
#line 296 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
char *__cdecl _strdate(char * _Buffer); 
#line 302 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 307
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strtime_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strtime_s ( _Buffer, _Size ); }}
#line 312 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
char *__cdecl _strtime(char * _Buffer); 
#line 317 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
__time32_t __cdecl _time32(__time32_t * _Time); 
#line 321
__time64_t __cdecl _time64(__time64_t * _Time); 
#line 327
int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 334
int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 348
void __cdecl _tzset(); 
#line 351
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 352
_getsystime(tm * _Tm); 
#line 356
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 357
_setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 501 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 502
_Time) 
#line 504
{ 
#line 505
return _ctime64(_Time); 
#line 506
} 
#line 509
static __inline double __cdecl difftime(const time_t 
#line 510
_Time1, const time_t 
#line 511
_Time2) 
#line 513
{ 
#line 514
return _difftime64(_Time1, _Time2); 
#line 515
} 
#line 518
static __inline tm *__cdecl gmtime(const time_t *const 
#line 519
_Time) 
#line 520
{ 
#line 521
return _gmtime64(_Time); 
#line 522
} 
#line 525
static __inline tm *__cdecl localtime(const time_t *const 
#line 526
_Time) 
#line 528
{ 
#line 529
return _localtime64(_Time); 
#line 530
} 
#line 533
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 534
_Tm) 
#line 536
{ 
#line 537
return _mkgmtime64(_Tm); 
#line 538
} 
#line 541
static __inline time_t __cdecl mktime(tm *const 
#line 542
_Tm) 
#line 544
{ 
#line 545
return _mktime64(_Tm); 
#line 546
} 
#line 548
static __inline time_t __cdecl time(time_t *const 
#line 549
_Time) 
#line 551
{ 
#line 552
return _time64(_Time); 
#line 553
} 
#line 556
static __inline int __cdecl timespec_get(timespec *const 
#line 557
_Ts, const int 
#line 558
_Base) 
#line 560
{ 
#line 561
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 562
} 
#line 566
static __inline errno_t __cdecl ctime_s(char *const 
#line 567
_Buffer, const size_t 
#line 568
_SizeInBytes, const time_t *const 
#line 569
_Time) 
#line 571
{ 
#line 572
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 573
} 
#line 603 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 604
_Tm, const time_t *const 
#line 605
_Time) 
#line 607
{ 
#line 608
return _gmtime64_s(_Tm, _Time); 
#line 609
} 
#line 612
static __inline errno_t __cdecl localtime_s(tm *const 
#line 613
_Tm, const time_t *const 
#line 614
_Time) 
#line 616
{ 
#line 617
return _localtime64_s(_Tm, _Time); 
#line 618
} 
#line 638 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
void __cdecl tzset(); 
#line 645 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 647
#pragma warning(pop)
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt/common_functions.h"
extern "C" {
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt/common_functions.h"
extern clock_t __cdecl clock(); 
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt/common_functions.h"
extern void *__cdecl memset(void *, int, size_t); 
#line 97
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 99
}
#line 158 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern "C" {
#line 265 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __cdecl abs(int a); 
#line 276
extern long __cdecl labs(long a); 
#line 287
extern __int64 llabs(__int64 a); 
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern __inline float fabsf(float x); 
#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern inline int min(const int a, const int b); 
#line 356
extern inline unsigned umin(const unsigned a, const unsigned b); 
#line 363
extern inline __int64 llmin(const __int64 a, const __int64 b); 
#line 370
extern inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 b); 
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl fminf(float x, float y); 
#line 413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl fmin(double x, double y); 
#line 424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern inline int max(const int a, const int b); 
#line 432
extern inline unsigned umax(const unsigned a, const unsigned b); 
#line 439
extern inline __int64 llmax(const __int64 a, const __int64 b); 
#line 446
extern inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 b); 
#line 469 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl fmaxf(float x, float y); 
#line 489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl fmax(double, double); 
#line 509 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl sin(double x); 
#line 527
extern double __cdecl cos(double x); 
#line 543 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern void sincos(double x, double * sptr, double * cptr); 
#line 556
extern void sincosf(float x, float * sptr, float * cptr); 
#line 579 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl tan(double x); 
#line 603
extern double __cdecl sqrt(double x); 
#line 629 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double rsqrt(double x); 
#line 653
extern float rsqrtf(float x); 
#line 682 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl log2(double x); 
#line 711 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl exp2(double x); 
#line 740 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl exp2f(float x); 
#line 769 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double exp10(double x); 
#line 796
extern float exp10f(float x); 
#line 832 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl expm1(double x); 
#line 865 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl expm1f(float x); 
#line 892 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl log2f(float x); 
#line 915 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl log10(double x); 
#line 941
extern double __cdecl log(double x); 
#line 970 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl log1p(double x); 
#line 1000 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl log1pf(float x); 
#line 1024 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl floor(double x); 
#line 1053
extern double __cdecl exp(double x); 
#line 1072
extern double __cdecl cosh(double x); 
#line 1092
extern double __cdecl sinh(double x); 
#line 1112
extern double __cdecl tanh(double x); 
#line 1138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl acosh(double x); 
#line 1165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl acoshf(float x); 
#line 1189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl asinh(double x); 
#line 1213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl asinhf(float x); 
#line 1238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl atanh(double x); 
#line 1263 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl atanhf(float x); 
#line 1279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl ldexp(double x, int exp); 
#line 1294
extern __inline float ldexpf(float x, int exp); 
#line 1317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl logb(double x); 
#line 1341 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl logbf(float x); 
#line 1365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __cdecl ilogb(double x); 
#line 1389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __cdecl ilogbf(float x); 
#line 1417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl scalbn(double x, int n); 
#line 1445 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl scalbnf(float x, int n); 
#line 1473 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl scalbln(double x, long n); 
#line 1501 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl scalblnf(float x, long n); 
#line 1531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl frexp(double x, int * nptr); 
#line 1560
extern __inline float frexpf(float x, int * nptr); 
#line 1585 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl round(double x); 
#line 1611 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl roundf(float x); 
#line 1629 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern long __cdecl lround(double x); 
#line 1647 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern long __cdecl lroundf(float x); 
#line 1665 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern __int64 __cdecl llround(double x); 
#line 1683 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern __int64 __cdecl llroundf(float x); 
#line 1753 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl rintf(float x); 
#line 1770 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern long __cdecl lrint(double x); 
#line 1787 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern long __cdecl lrintf(float x); 
#line 1804 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrint(double x); 
#line 1821 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrintf(float x); 
#line 1845 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl nearbyint(double x); 
#line 1869 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl nearbyintf(float x); 
#line 1891 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl ceil(double x); 
#line 1916 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl trunc(double x); 
#line 1942 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl truncf(float x); 
#line 1964 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl fdim(double x, double y); 
#line 1985 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl fdimf(float x, float y); 
#line 2066 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 2092
extern double __cdecl atan(double x); 
#line 2109
extern double __cdecl acos(double x); 
#line 2131
extern double __cdecl asin(double x); 
#line 2159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl hypot(double x, double y); 
#line 2217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
static __inline float __cdecl hypotf(float x, float y); 
#line 2493 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl cbrt(double x); 
#line 2520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl cbrtf(float x); 
#line 2544 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double rcbrt(double x); 
#line 2565
extern float rcbrtf(float x); 
#line 2594 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double sinpi(double x); 
#line 2617
extern float sinpif(float x); 
#line 2639
extern double cospi(double x); 
#line 2661
extern float cospif(float x); 
#line 2676
extern void sincospi(double x, double * sptr, double * cptr); 
#line 2689
extern void sincospif(float x, float * sptr, float * cptr); 
#line 2775 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl pow(double x, double y); 
#line 2799
extern double __cdecl modf(double x, double * iptr); 
#line 2826
extern double __cdecl fmod(double x, double y); 
#line 2858 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl remainder(double x, double y); 
#line 2891 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl remainderf(float x, float y); 
#line 2929 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl remquo(double x, double y, int * quo); 
#line 2967 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl remquof(float x, float y, int * quo); 
#line 2986 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl j0(double x); 
#line 3008 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float j0f(float x); 
#line 3035 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl j1(double x); 
#line 3062 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float j1f(float x); 
#line 3085 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl jn(int n, double x); 
#line 3108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float jnf(int n, float x); 
#line 3135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl y0(double x); 
#line 3162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float y0f(float x); 
#line 3189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl y1(double x); 
#line 3216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float y1f(float x); 
#line 3244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl yn(int n, double x); 
#line 3272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float ynf(int n, float x); 
#line 3370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl erf(double x); 
#line 3395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl erff(float x); 
#line 3423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double erfinv(double x); 
#line 3446
extern float erfinvf(float x); 
#line 3472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl erfc(double x); 
#line 3495 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl erfcf(float x); 
#line 3527 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl lgamma(double x); 
#line 3553 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double erfcinv(double x); 
#line 3574
extern float erfcinvf(float x); 
#line 3596
extern double normcdfinv(double x); 
#line 3618
extern float normcdfinvf(float x); 
#line 3637
extern double normcdf(double x); 
#line 3656
extern float normcdff(float x); 
#line 3676
extern double erfcx(double x); 
#line 3696
extern float erfcxf(float x); 
#line 3731 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl lgammaf(float x); 
#line 3760 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl tgamma(double x); 
#line 3789 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl tgammaf(float x); 
#line 3803 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl copysign(double x, double y); 
#line 3817 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl copysignf(float x, float y); 
#line 3836 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl nextafter(double x, double y); 
#line 3855 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl nextafterf(float x, float y); 
#line 3871 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl nan(const char * tagp); 
#line 3887 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl nanf(const char * tagp); 
#line 3892 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __isinff(float); 
#line 3893
extern int __isnanf(float); 
#line 3903 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __finite(double); 
#line 3904
extern int __finitef(float); 
#line 3905
extern int __signbit(double); 
#line 3906
extern int __isnan(double); 
#line 3907
extern int __isinf(double); 
#line 3910 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __signbitf(float); 
#line 3963 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern double __cdecl fma(double x, double y, double z); 
#line 4013 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl fmaf(float x, float y, float z); 
#line 4022 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __signbitl(long double); 
#line 4028 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern int __finitel(long double); 
#line 4029
extern int __isinfl(long double); 
#line 4030
extern int __isnanl(long double); 
#line 4034 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
extern float __cdecl acosf(float); 
#line 4035
extern float __cdecl asinf(float); 
#line 4036
extern float __cdecl atanf(float); 
#line 4037
extern float __cdecl atan2f(float, float); 
#line 4038
extern float __cdecl cosf(float); 
#line 4039
extern float __cdecl sinf(float); 
#line 4040
extern float __cdecl tanf(float); 
#line 4041
extern float __cdecl coshf(float); 
#line 4042
extern float __cdecl sinhf(float); 
#line 4043
extern float __cdecl tanhf(float); 
#line 4044
extern float __cdecl expf(float); 
#line 4045
extern float __cdecl logf(float); 
#line 4046
extern float __cdecl log10f(float); 
#line 4047
extern float __cdecl modff(float, float *); 
#line 4048
extern float __cdecl powf(float, float); 
#line 4049
extern float __cdecl sqrtf(float); 
#line 4050
extern float __cdecl ceilf(float); 
#line 4051
extern float __cdecl floorf(float); 
#line 4052
extern float __cdecl fmodf(float, float); 
#line 4614 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include\\crt\\math_functions.h"
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
struct _exception { 
#line 25
int type; 
#line 26
char *name; 
#line 27
double arg1; 
#line 28
double arg2; 
#line 29
double retval; 
#line 30
}; 
#line 37
struct _complex { 
#line 39
double x, y; 
#line 40
}; 
#line 59 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
typedef float float_t; 
#line 60
typedef double double_t; 
#line 78 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
extern const double _HUGE; 
#line 189 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
void __cdecl _fperrraise(int _Except); 
#line 191
short __cdecl _dclass(double _X); 
#line 192
short __cdecl _ldclass(long double _X); 
#line 193
short __cdecl _fdclass(float _X); 
#line 195
int __cdecl _dsign(double _X); 
#line 196
int __cdecl _ldsign(long double _X); 
#line 197
int __cdecl _fdsign(float _X); 
#line 199
int __cdecl _dpcomp(double _X, double _Y); 
#line 200
int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 201
int __cdecl _fdpcomp(float _X, float _Y); 
#line 203
short __cdecl _dtest(double * _Px); 
#line 204
short __cdecl _ldtest(long double * _Px); 
#line 205
short __cdecl _fdtest(float * _Px); 
#line 207
short __cdecl _d_int(double * _Px, short _Xexp); 
#line 208
short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 209
short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 211
short __cdecl _dscale(double * _Px, long _Lexp); 
#line 212
short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 213
short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 215
short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 216
short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 217
short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 219
short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 220
short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 221
short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 223
short __cdecl _dnorm(unsigned short * _Ps); 
#line 224
short __cdecl _fdnorm(unsigned short * _Ps); 
#line 226
double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 227
long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 228
float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 230
double __cdecl _dlog(double _X, int _Baseflag); 
#line 231
long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 232
float __cdecl _fdlog(float _X, int _Baseflag); 
#line 234
double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 235
long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 236
float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 243
typedef 
#line 240
union { 
#line 241
unsigned short _Sh[4]; 
#line 242
double _Val; 
#line 243
} _double_val; 
#line 250
typedef 
#line 247
union { 
#line 248
unsigned short _Sh[2]; 
#line 249
float _Val; 
#line 250
} _float_val; 
#line 257
typedef 
#line 254
union { 
#line 255
unsigned short _Sh[4]; 
#line 256
long double _Val; 
#line 257
} _ldouble_val; 
#line 265
typedef 
#line 260
union { 
#line 261
unsigned short _Word[4]; 
#line 262
float _Float; 
#line 263
double _Double; 
#line 264
long double _Long_double; 
#line 265
} _float_const; 
#line 267
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 268
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 269
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 271
extern const _float_const _Eps_C, _Rteps_C; 
#line 272
extern const _float_const _FEps_C, _FRteps_C; 
#line 273
extern const _float_const _LEps_C, _LRteps_C; 
#line 275
extern const double _Zero_C, _Xbig_C; 
#line 276
extern const float _FZero_C, _FXbig_C; 
#line 277
extern const long double _LZero_C, _LXbig_C; 
#line 310
extern "C++" {
#line 312
inline int fpclassify(float _X) throw() 
#line 313
{ 
#line 317 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
return _fdtest(&_X); 
#line 319 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
} 
#line 321
inline int fpclassify(double _X) throw() 
#line 322
{ 
#line 326 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
return _dtest(&_X); 
#line 328 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
} 
#line 330
inline int fpclassify(long double _X) throw() 
#line 331
{ 
#line 335 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
return _ldtest(&_X); 
#line 337 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
} 
#line 339
inline bool signbit(float _X) throw() 
#line 340
{ 
#line 341
return _fdsign(_X) != 0; 
#line 342
} 
#line 344
inline bool signbit(double _X) throw() 
#line 345
{ 
#line 346
return _dsign(_X) != 0; 
#line 347
} 
#line 349
inline bool signbit(long double _X) throw() 
#line 350
{ 
#line 351
return _ldsign(_X) != 0; 
#line 352
} 
#line 354
inline int _fpcomp(float _X, float _Y) throw() 
#line 355
{ 
#line 356
return _fdpcomp(_X, _Y); 
#line 357
} 
#line 359
inline int _fpcomp(double _X, double _Y) throw() 
#line 360
{ 
#line 361
return _dpcomp(_X, _Y); 
#line 362
} 
#line 364
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 365
{ 
#line 366
return _ldpcomp(_X, _Y); 
#line 367
} 
#line 369
template< class _Trc, class _Tre> struct _Combined_type { 
#line 371
typedef float _Type; 
#line 372
}; 
#line 374
template<> struct _Combined_type< float, double>  { 
#line 376
typedef double _Type; 
#line 377
}; 
#line 379
template<> struct _Combined_type< float, long double>  { 
#line 381
typedef long double _Type; 
#line 382
}; 
#line 384
template< class _Ty, class _T2> struct _Real_widened { 
#line 386
typedef long double _Type; 
#line 387
}; 
#line 389
template<> struct _Real_widened< float, float>  { 
#line 391
typedef float _Type; 
#line 392
}; 
#line 394
template<> struct _Real_widened< float, double>  { 
#line 396
typedef double _Type; 
#line 397
}; 
#line 399
template<> struct _Real_widened< double, float>  { 
#line 401
typedef double _Type; 
#line 402
}; 
#line 404
template<> struct _Real_widened< double, double>  { 
#line 406
typedef double _Type; 
#line 407
}; 
#line 409
template< class _Ty> struct _Real_type { 
#line 411
typedef double _Type; 
#line 412
}; 
#line 414
template<> struct _Real_type< float>  { 
#line 416
typedef float _Type; 
#line 417
}; 
#line 419
template<> struct _Real_type< long double>  { 
#line 421
typedef long double _Type; 
#line 422
}; 
#line 424
template < class _T1, class _T2 >
      inline int _fpcomp ( _T1 _X, _T2 _Y ) throw ( )
    {
        typedef typename _Combined_type < float,
            typename _Real_widened <
            typename _Real_type < _T1 > :: _Type,
            typename _Real_type < _T2 > :: _Type > :: _Type > :: _Type _Tw;
        return _fpcomp ( ( _Tw ) _X, ( _Tw ) _Y );
    }
#line 434
template < class _Ty >
      inline bool isfinite ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) <= 0;
    }
#line 440
template < class _Ty >
      inline bool isinf ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 1;
    }
#line 446
template < class _Ty >
      inline bool isnan ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 2;
    }
#line 452
template < class _Ty >
      inline bool isnormal ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == ( - 1 );
    }
#line 458
template < class _Ty1, class _Ty2 >
      inline bool isgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 4 ) != 0;
    }
#line 464
template < class _Ty1, class _Ty2 >
      inline bool isgreaterequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 2 | 4 ) ) != 0;
    }
#line 470
template < class _Ty1, class _Ty2 >
      inline bool isless ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 1 ) != 0;
    }
#line 476
template < class _Ty1, class _Ty2 >
      inline bool islessequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 2 ) ) != 0;
    }
#line 482
template < class _Ty1, class _Ty2 >
      inline bool islessgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 4 ) ) != 0;
    }
#line 488
template < class _Ty1, class _Ty2 >
      inline bool isunordered ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return _fpcomp ( _X, _Y ) == 0;
    }
#line 493
}
#line 500 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
int __cdecl abs(int _X); 
#line 501
long __cdecl labs(long _X); 
#line 502
__int64 __cdecl llabs(__int64 _X); 
#line 504
double __cdecl acos(double _X); 
#line 505
double __cdecl asin(double _X); 
#line 506
double __cdecl atan(double _X); 
#line 507
double __cdecl atan2(double _Y, double _X); 
#line 509
double __cdecl cos(double _X); 
#line 510
double __cdecl cosh(double _X); 
#line 511
double __cdecl exp(double _X); 
#line 512
double __cdecl fabs(double _X); 
#line 513
double __cdecl fmod(double _X, double _Y); 
#line 514
double __cdecl log(double _X); 
#line 515
double __cdecl log10(double _X); 
#line 516
double __cdecl pow(double _X, double _Y); 
#line 517
double __cdecl sin(double _X); 
#line 518
double __cdecl sinh(double _X); 
#line 519
double __cdecl sqrt(double _X); 
#line 520
double __cdecl tan(double _X); 
#line 521
double __cdecl tanh(double _X); 
#line 523
double __cdecl acosh(double _X); 
#line 524
double __cdecl asinh(double _X); 
#line 525
double __cdecl atanh(double _X); 
#line 526
double __cdecl atof(const char * _String); 
#line 527
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 528
double __cdecl _cabs(_complex _Complex_value); 
#line 529
double __cdecl cbrt(double _X); 
#line 530
double __cdecl ceil(double _X); 
#line 531
double __cdecl _chgsign(double _X); 
#line 532
double __cdecl copysign(double _Number, double _Sign); 
#line 533
double __cdecl _copysign(double _Number, double _Sign); 
#line 534
double __cdecl erf(double _X); 
#line 535
double __cdecl erfc(double _X); 
#line 536
double __cdecl exp2(double _X); 
#line 537
double __cdecl expm1(double _X); 
#line 538
double __cdecl fdim(double _X, double _Y); 
#line 539
double __cdecl floor(double _X); 
#line 540
double __cdecl fma(double _X, double _Y, double _Z); 
#line 541
double __cdecl fmax(double _X, double _Y); 
#line 542
double __cdecl fmin(double _X, double _Y); 
#line 543
double __cdecl frexp(double _X, int * _Y); 
#line 544
double __cdecl hypot(double _X, double _Y); 
#line 545
double __cdecl _hypot(double _X, double _Y); 
#line 546
int __cdecl ilogb(double _X); 
#line 547
double __cdecl ldexp(double _X, int _Y); 
#line 548
double __cdecl lgamma(double _X); 
#line 549
__int64 __cdecl llrint(double _X); 
#line 550
__int64 __cdecl llround(double _X); 
#line 551
double __cdecl log1p(double _X); 
#line 552
double __cdecl log2(double _X); 
#line 553
double __cdecl logb(double _X); 
#line 554
long __cdecl lrint(double _X); 
#line 555
long __cdecl lround(double _X); 
#line 557
int __cdecl _matherr(_exception * _Except); 
#line 559
double __cdecl modf(double _X, double * _Y); 
#line 560
double __cdecl nan(const char * _X); 
#line 561
double __cdecl nearbyint(double _X); 
#line 562
double __cdecl nextafter(double _X, double _Y); 
#line 563
double __cdecl nexttoward(double _X, long double _Y); 
#line 564
double __cdecl remainder(double _X, double _Y); 
#line 565
double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 566
double __cdecl rint(double _X); 
#line 567
double __cdecl round(double _X); 
#line 568
double __cdecl scalbln(double _X, long _Y); 
#line 569
double __cdecl scalbn(double _X, int _Y); 
#line 570
double __cdecl tgamma(double _X); 
#line 571
double __cdecl trunc(double _X); 
#line 572
double __cdecl _j0(double _X); 
#line 573
double __cdecl _j1(double _X); 
#line 574
double __cdecl _jn(int _X, double _Y); 
#line 575
double __cdecl _y0(double _X); 
#line 576
double __cdecl _y1(double _X); 
#line 577
double __cdecl _yn(int _X, double _Y); 
#line 579
float __cdecl acoshf(float _X); 
#line 580
float __cdecl asinhf(float _X); 
#line 581
float __cdecl atanhf(float _X); 
#line 582
float __cdecl cbrtf(float _X); 
#line 583
float __cdecl _chgsignf(float _X); 
#line 584
float __cdecl copysignf(float _Number, float _Sign); 
#line 585
float __cdecl _copysignf(float _Number, float _Sign); 
#line 586
float __cdecl erff(float _X); 
#line 587
float __cdecl erfcf(float _X); 
#line 588
float __cdecl expm1f(float _X); 
#line 589
float __cdecl exp2f(float _X); 
#line 590
float __cdecl fdimf(float _X, float _Y); 
#line 591
float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 592
float __cdecl fmaxf(float _X, float _Y); 
#line 593
float __cdecl fminf(float _X, float _Y); 
#line 594
float __cdecl _hypotf(float _X, float _Y); 
#line 595
int __cdecl ilogbf(float _X); 
#line 596
float __cdecl lgammaf(float _X); 
#line 597
__int64 __cdecl llrintf(float _X); 
#line 598
__int64 __cdecl llroundf(float _X); 
#line 599
float __cdecl log1pf(float _X); 
#line 600
float __cdecl log2f(float _X); 
#line 601
float __cdecl logbf(float _X); 
#line 602
long __cdecl lrintf(float _X); 
#line 603
long __cdecl lroundf(float _X); 
#line 604
float __cdecl nanf(const char * _X); 
#line 605
float __cdecl nearbyintf(float _X); 
#line 606
float __cdecl nextafterf(float _X, float _Y); 
#line 607
float __cdecl nexttowardf(float _X, long double _Y); 
#line 608
float __cdecl remainderf(float _X, float _Y); 
#line 609
float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 610
float __cdecl rintf(float _X); 
#line 611
float __cdecl roundf(float _X); 
#line 612
float __cdecl scalblnf(float _X, long _Y); 
#line 613
float __cdecl scalbnf(float _X, int _Y); 
#line 614
float __cdecl tgammaf(float _X); 
#line 615
float __cdecl truncf(float _X); 
#line 625 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
float __cdecl _logbf(float _X); 
#line 626
float __cdecl _nextafterf(float _X, float _Y); 
#line 627
int __cdecl _finitef(float _X); 
#line 628
int __cdecl _isnanf(float _X); 
#line 629
int __cdecl _fpclassf(float _X); 
#line 631
int __cdecl _set_FMA3_enable(int _Flag); 
#line 632
int __cdecl _get_FMA3_enable(); 
#line 645 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
float __cdecl acosf(float _X); 
#line 646
float __cdecl asinf(float _X); 
#line 647
float __cdecl atan2f(float _Y, float _X); 
#line 648
float __cdecl atanf(float _X); 
#line 649
float __cdecl ceilf(float _X); 
#line 650
float __cdecl cosf(float _X); 
#line 651
float __cdecl coshf(float _X); 
#line 652
float __cdecl expf(float _X); 
#line 709 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
__inline float __cdecl fabsf(float _X) 
#line 710
{ 
#line 711
return (float)fabs(_X); 
#line 712
} 
#line 718 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
float __cdecl floorf(float _X); 
#line 719
float __cdecl fmodf(float _X, float _Y); 
#line 735 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 736
{ 
#line 737
return (float)frexp(_X, _Y); 
#line 738
} 
#line 740
__inline float __cdecl hypotf(float _X, float _Y) 
#line 741
{ 
#line 742
return _hypotf(_X, _Y); 
#line 743
} 
#line 745
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 746
{ 
#line 747
return (float)ldexp(_X, _Y); 
#line 748
} 
#line 752
float __cdecl log10f(float _X); 
#line 753
float __cdecl logf(float _X); 
#line 754
float __cdecl modff(float _X, float * _Y); 
#line 755
float __cdecl powf(float _X, float _Y); 
#line 756
float __cdecl sinf(float _X); 
#line 757
float __cdecl sinhf(float _X); 
#line 758
float __cdecl sqrtf(float _X); 
#line 759
float __cdecl tanf(float _X); 
#line 760
float __cdecl tanhf(float _X); 
#line 814 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
long double __cdecl acoshl(long double _X); 
#line 816
__inline long double __cdecl acosl(long double _X) 
#line 817
{ 
#line 818
return acos((double)_X); 
#line 819
} 
#line 821
long double __cdecl asinhl(long double _X); 
#line 823
__inline long double __cdecl asinl(long double _X) 
#line 824
{ 
#line 825
return asin((double)_X); 
#line 826
} 
#line 828
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 829
{ 
#line 830
return atan2((double)_Y, (double)_X); 
#line 831
} 
#line 833
long double __cdecl atanhl(long double _X); 
#line 835
__inline long double __cdecl atanl(long double _X) 
#line 836
{ 
#line 837
return atan((double)_X); 
#line 838
} 
#line 840
long double __cdecl cbrtl(long double _X); 
#line 842
__inline long double __cdecl ceill(long double _X) 
#line 843
{ 
#line 844
return ceil((double)_X); 
#line 845
} 
#line 847
__inline long double __cdecl _chgsignl(long double _X) 
#line 848
{ 
#line 849
return _chgsign((double)_X); 
#line 850
} 
#line 852
long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 854
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 855
{ 
#line 856
return _copysign((double)_Number, (double)_Sign); 
#line 857
} 
#line 859
__inline long double __cdecl coshl(long double _X) 
#line 860
{ 
#line 861
return cosh((double)_X); 
#line 862
} 
#line 864
__inline long double __cdecl cosl(long double _X) 
#line 865
{ 
#line 866
return cos((double)_X); 
#line 867
} 
#line 869
long double __cdecl erfl(long double _X); 
#line 870
long double __cdecl erfcl(long double _X); 
#line 872
__inline long double __cdecl expl(long double _X) 
#line 873
{ 
#line 874
return exp((double)_X); 
#line 875
} 
#line 877
long double __cdecl exp2l(long double _X); 
#line 878
long double __cdecl expm1l(long double _X); 
#line 880
__inline long double __cdecl fabsl(long double _X) 
#line 881
{ 
#line 882
return fabs((double)_X); 
#line 883
} 
#line 885
long double __cdecl fdiml(long double _X, long double _Y); 
#line 887
__inline long double __cdecl floorl(long double _X) 
#line 888
{ 
#line 889
return floor((double)_X); 
#line 890
} 
#line 892
long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 893
long double __cdecl fmaxl(long double _X, long double _Y); 
#line 894
long double __cdecl fminl(long double _X, long double _Y); 
#line 896
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 897
{ 
#line 898
return fmod((double)_X, (double)_Y); 
#line 899
} 
#line 901
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 902
{ 
#line 903
return frexp((double)_X, _Y); 
#line 904
} 
#line 906
int __cdecl ilogbl(long double _X); 
#line 908
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 909
{ 
#line 910
return _hypot((double)_X, (double)_Y); 
#line 911
} 
#line 913
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 914
{ 
#line 915
return _hypot((double)_X, (double)_Y); 
#line 916
} 
#line 918
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 919
{ 
#line 920
return ldexp((double)_X, _Y); 
#line 921
} 
#line 923
long double __cdecl lgammal(long double _X); 
#line 924
__int64 __cdecl llrintl(long double _X); 
#line 925
__int64 __cdecl llroundl(long double _X); 
#line 927
__inline long double __cdecl logl(long double _X) 
#line 928
{ 
#line 929
return log((double)_X); 
#line 930
} 
#line 932
__inline long double __cdecl log10l(long double _X) 
#line 933
{ 
#line 934
return log10((double)_X); 
#line 935
} 
#line 937
long double __cdecl log1pl(long double _X); 
#line 938
long double __cdecl log2l(long double _X); 
#line 939
long double __cdecl logbl(long double _X); 
#line 940
long __cdecl lrintl(long double _X); 
#line 941
long __cdecl lroundl(long double _X); 
#line 943
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 944
{ 
#line 945
double _F, _I; 
#line 946
_F = modf((double)_X, &_I); 
#line 947
(*_Y) = _I; 
#line 948
return _F; 
#line 949
} 
#line 951
long double __cdecl nanl(const char * _X); 
#line 952
long double __cdecl nearbyintl(long double _X); 
#line 953
long double __cdecl nextafterl(long double _X, long double _Y); 
#line 954
long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 956
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 957
{ 
#line 958
return pow((double)_X, (double)_Y); 
#line 959
} 
#line 961
long double __cdecl remainderl(long double _X, long double _Y); 
#line 962
long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 963
long double __cdecl rintl(long double _X); 
#line 964
long double __cdecl roundl(long double _X); 
#line 965
long double __cdecl scalblnl(long double _X, long _Y); 
#line 966
long double __cdecl scalbnl(long double _X, int _Y); 
#line 968
__inline long double __cdecl sinhl(long double _X) 
#line 969
{ 
#line 970
return sinh((double)_X); 
#line 971
} 
#line 973
__inline long double __cdecl sinl(long double _X) 
#line 974
{ 
#line 975
return sin((double)_X); 
#line 976
} 
#line 978
__inline long double __cdecl sqrtl(long double _X) 
#line 979
{ 
#line 980
return sqrt((double)_X); 
#line 981
} 
#line 983
__inline long double __cdecl tanhl(long double _X) 
#line 984
{ 
#line 985
return tanh((double)_X); 
#line 986
} 
#line 988
__inline long double __cdecl tanl(long double _X) 
#line 989
{ 
#line 990
return tan((double)_X); 
#line 991
} 
#line 993
long double __cdecl tgammal(long double _X); 
#line 994
long double __cdecl truncl(long double _X); 
#line 1015 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
extern double HUGE; 
#line 1020 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.26100.0\\ucrt\\corecrt_math.h"
double __cdecl j0(double _X); 
#line 1021
double __cdecl j1(double _X); 
#line 1022
dou