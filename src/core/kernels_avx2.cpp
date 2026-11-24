// SPDX-License-Identifier: GPL-2.0-or-later
// AVX2 kernels (32 samples per step at 8 bit, 16 at 9-16 bit, 8 in the warp and in float). Built with /arch:AVX2 or
// -mavx2 in its own library and called only after the CPU check (cpu.cpp).
#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "kernels.hpp"
#include "scalar.hpp"

namespace awarp {
namespace {

struct S {
  using V8 = __m256i;
  using V16 = __m256i;
  using VI = __m256i;
  using VF = __m256;
  using MI = __m256i;
  using MF = __m256;
  static constexpr int N8 = 32, N16 = 16, NI = 8;

  static V8 ld8(const uint8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void st8(uint8_t* p, V8 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V8 set8(int v) { return _mm256_set1_epi8(char(v)); }
  static V8 avg8(V8 a, V8 b) { return _mm256_avg_epu8(a, b); }
  static V8 absd8(V8 a, V8 b) { return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)); }
  static V8 adds8(V8 a, V8 b) { return _mm256_adds_epu8(a, b); }
  static V8 add8(V8 a, V8 b) { return _mm256_add_epi8(a, b); }
  static V8 max8(V8 a, V8 b) { return _mm256_max_epu8(a, b); }
  static V8 min8(V8 a, V8 b) { return _mm256_min_epu8(a, b); }

  static V16 ld16(const uint16_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void st16(uint16_t* p, V16 v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static V16 set16(int v) { return _mm256_set1_epi16(short(v)); }
  static V16 avg16(V16 a, V16 b) { return _mm256_avg_epu16(a, b); }
  static V16 absd16(V16 a, V16 b) { return _mm256_or_si256(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a)); }
  static V16 adds16(V16 a, V16 b) { return _mm256_adds_epu16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm256_add_epi16(a, b); }
  static V16 and16(V16 a, V16 b) { return _mm256_and_si256(a, b); }
  static V16 max16(V16 a, V16 b) { return _mm256_max_epu16(a, b); }
  static V16 min16(V16 a, V16 b) { return _mm256_min_epu16(a, b); }

  static VF ldf(const float* p) { return _mm256_loadu_ps(p); }
  static void stf(float* p, VF v) { _mm256_storeu_ps(p, v); }
  static VF setf(float v) { return _mm256_set1_ps(v); }
  static VF addf(VF a, VF b) { return _mm256_add_ps(a, b); }
  static VF subf(VF a, VF b) { return _mm256_sub_ps(a, b); }
  static VF mulf(VF a, VF b) { return _mm256_mul_ps(a, b); }
  static VF minf(VF a, VF b) { return _mm256_min_ps(a, b); }
  static VF maxf(VF a, VF b) { return _mm256_max_ps(a, b); }
  static VF absf(VF v) { return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))); }
  static VF floorf_(VF v) { return _mm256_floor_ps(v); }
  static MF ltf(VF a, VF b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
  static MF gef(VF a, VF b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
  static VF self(MF m, VF a, VF b) { return _mm256_blendv_ps(b, a, m); }
  static MF andMF(MF a, MF b) { return _mm256_and_ps(a, b); }
  static MI mfToMi(MF m) { return _mm256_castps_si256(m); }

  static VI setI(int v) { return _mm256_set1_epi32(v); }
  static VI iotaI() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
  static VI ld8I(const uint8_t* p) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
  }
  static VI ld16I(const uint16_t* p) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  }
  static VI ldI(const int32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static void stI(int32_t* p, VI v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
  static VI addI(VI a, VI b) { return _mm256_add_epi32(a, b); }
  static VI subI(VI a, VI b) { return _mm256_sub_epi32(a, b); }
  static VI mulI(VI a, VI b) { return _mm256_mullo_epi32(a, b); }
  template <int K>
  static VI sraI(VI v) { return _mm256_srai_epi32(v, K); }
  template <int K>
  static VI sllI(VI v) { return _mm256_slli_epi32(v, K); }
  static VI andI(VI a, VI b) { return _mm256_and_si256(a, b); }
  static VI minI(VI a, VI b) { return _mm256_min_epi32(a, b); }
  static VI maxI(VI a, VI b) { return _mm256_max_epi32(a, b); }
  static MI ltI(VI a, VI b) { return _mm256_cmpgt_epi32(b, a); }
  static VI selI(MI m, VI a, VI b) { return _mm256_blendv_epi8(b, a, m); }
  static MI andMI(MI a, MI b) { return _mm256_and_si256(a, b); }
  static MI notMI(MI a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
  static __m128i pack16(VI v) { return _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)); }
  static void stI8(uint8_t* p, VI v) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packus_epi16(pack16(v), _mm_setzero_si128()));
  }
  static void stI16(uint16_t* p, VI v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), pack16(v)); }
  static constexpr bool kGather = true;
  template <int K>
  static VI srlI(VI v) { return _mm256_srli_epi32(v, K); }
  static bool allMI(MI m) { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) == 0xFF; }
  template <int SC, typename T>
  static VI gather32(const T* base, VI idx, MI mask) {
    return _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), reinterpret_cast<const int*>(base), idx, mask, SC);
  }
  static VF gatherF(const float* base, VI idx) { return _mm256_i32gather_ps(base, idx, 4); }
  static VI orI(VI a, VI b) { return _mm256_or_si256(a, b); }
  static VI xorI(VI a, VI b) { return _mm256_xor_si256(a, b); }
  static VI madd16(VI a, VI b) { return _mm256_madd_epi16(a, b); }
  using VW = __m256i;
  static constexpr int NW = 16;
  static VW setw(int v) { return _mm256_set1_epi16(short(v)); }
  static VW ldw(const int16_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static VW ldw8(const uint8_t* p) { return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
  static VW addw(VW a, VW b) { return _mm256_add_epi16(a, b); }
  static VW subw(VW a, VW b) { return _mm256_sub_epi16(a, b); }
  static VW mullow(VW a, VW b) { return _mm256_mullo_epi16(a, b); }
  static VW mulhsw(VW a, VW b) { return _mm256_mulhi_epi16(a, b); }
  static VW andw(VW a, VW b) { return _mm256_and_si256(a, b); }
  template <int K>
  static VW sllw(VW v) { return _mm256_slli_epi16(v, K); }
  template <int K>
  static VW sraw(VW v) { return _mm256_srai_epi16(v, K); }
  template <int K>
  static VW srlw(VW v) { return _mm256_srli_epi16(v, K); }
  static VW minsw(VW a, VW b) { return _mm256_min_epi16(a, b); }
  static VW maxsw(VW a, VW b) { return _mm256_max_epi16(a, b); }
  // packs works per 128-bit half; the permute restores the lane order.
  static VW packsw(VI a, VI b) { return _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8); }
  static VI widenLo(VW v) { return _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)); }
  static VI widenHi(VW v) { return _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)); }
  static void stw8(uint8_t* p, VW v) {
    const __m256i b = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0xD8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(b));
  }
  static VW gatherw8(const uint8_t* base, VI i0, VI i1) {
    const __m256i m = _mm256_set1_epi32(0xFFFF);
    const __m256i a = _mm256_and_si256(_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), i0, 1), m);
    const __m256i b = _mm256_and_si256(_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), i1, 1), m);
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
  }
  static VW iotaw() { return _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
  static VW ltw(VW a, VW b) { return _mm256_cmpgt_epi16(b, a); }
  static VW notw(VW a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
  static VW pairw(int lo, int hi) { return _mm256_set1_epi32(int((unsigned(lo) & 0xFFFFu) | (unsigned(hi) << 16))); }
  // In-lane unpacks: lo holds samples 0-3 and 8-11, hi 4-7 and 12-15; gatherw8u packs them back in order.
  static void maddPairs(VW a, VW b, VW w, VI& lo, VI& hi) {
    lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w);
    hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w);
  }
  static VW gatherw8u(const uint8_t* base, VI i0, VI i1) {
    const __m256i m = _mm256_set1_epi32(0xFFFF);
    const __m256i a = _mm256_and_si256(_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), i0, 1), m);
    const __m256i b = _mm256_and_si256(_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), i1, 1), m);
    return _mm256_packus_epi32(a, b);
  }
  static VF cvtIF(VI v) { return _mm256_cvtepi32_ps(v); }
  static VI cvtFI(VF v) { return _mm256_cvttps_epi32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsAVX2() { return simdKernels(); }

}  // namespace awarp
