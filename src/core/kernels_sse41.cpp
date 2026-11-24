// SPDX-License-Identifier: GPL-2.0-or-later
// SSE4.1 kernels (16 samples per step at 8 bit, 8 at 9-16 bit, 4 in the warp and in float).
#include <smmintrin.h>

#include <algorithm>
#include <cstring>

#include "kernels.hpp"
#include "scalar.hpp"

namespace awarp {
namespace {

struct S {
  using V8 = __m128i;
  using V16 = __m128i;
  using VI = __m128i;
  using VF = __m128;
  using MI = __m128i;
  using MF = __m128;
  static constexpr int N8 = 16, N16 = 8, NI = 4;

  static V8 ld8(const uint8_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void st8(uint8_t* p, V8 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V8 set8(int v) { return _mm_set1_epi8(char(v)); }
  static V8 avg8(V8 a, V8 b) { return _mm_avg_epu8(a, b); }
  static V8 absd8(V8 a, V8 b) { return _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a)); }
  static V8 adds8(V8 a, V8 b) { return _mm_adds_epu8(a, b); }
  static V8 add8(V8 a, V8 b) { return _mm_add_epi8(a, b); }
  static V8 max8(V8 a, V8 b) { return _mm_max_epu8(a, b); }
  static V8 min8(V8 a, V8 b) { return _mm_min_epu8(a, b); }

  static V16 ld16(const uint16_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void st16(uint16_t* p, V16 v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static V16 set16(int v) { return _mm_set1_epi16(short(v)); }
  static V16 avg16(V16 a, V16 b) { return _mm_avg_epu16(a, b); }
  static V16 absd16(V16 a, V16 b) { return _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a)); }
  static V16 adds16(V16 a, V16 b) { return _mm_adds_epu16(a, b); }
  static V16 add16(V16 a, V16 b) { return _mm_add_epi16(a, b); }
  static V16 and16(V16 a, V16 b) { return _mm_and_si128(a, b); }
  static V16 max16(V16 a, V16 b) { return _mm_max_epu16(a, b); }
  static V16 min16(V16 a, V16 b) { return _mm_min_epu16(a, b); }

  static VF ldf(const float* p) { return _mm_loadu_ps(p); }
  static void stf(float* p, VF v) { _mm_storeu_ps(p, v); }
  static VF setf(float v) { return _mm_set1_ps(v); }
  static VF addf(VF a, VF b) { return _mm_add_ps(a, b); }
  static VF subf(VF a, VF b) { return _mm_sub_ps(a, b); }
  static VF mulf(VF a, VF b) { return _mm_mul_ps(a, b); }
  static VF minf(VF a, VF b) { return _mm_min_ps(a, b); }
  static VF maxf(VF a, VF b) { return _mm_max_ps(a, b); }
  static VF absf(VF v) { return _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))); }
  static VF floorf_(VF v) { return _mm_floor_ps(v); }
  static MF ltf(VF a, VF b) { return _mm_cmplt_ps(a, b); }
  static MF gef(VF a, VF b) { return _mm_cmpge_ps(a, b); }
  static VF self(MF m, VF a, VF b) { return _mm_blendv_ps(b, a, m); }
  static MF andMF(MF a, MF b) { return _mm_and_ps(a, b); }
  static MI mfToMi(MF m) { return _mm_castps_si128(m); }

  static VI setI(int v) { return _mm_set1_epi32(v); }
  static VI iotaI() { return _mm_setr_epi32(0, 1, 2, 3); }
  static VI ld8I(const uint8_t* p) {
    int v;
    std::memcpy(&v, p, 4);
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(v));
  }
  static VI ld16I(const uint16_t* p) {
    return _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
  }
  static VI ldI(const int32_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static void stI(int32_t* p, VI v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }
  static VI addI(VI a, VI b) { return _mm_add_epi32(a, b); }
  static VI subI(VI a, VI b) { return _mm_sub_epi32(a, b); }
  static VI mulI(VI a, VI b) { return _mm_mullo_epi32(a, b); }
  template <int K>
  static VI sraI(VI v) { return _mm_srai_epi32(v, K); }
  template <int K>
  static VI sllI(VI v) { return _mm_slli_epi32(v, K); }
  static VI andI(VI a, VI b) { return _mm_and_si128(a, b); }
  static VI minI(VI a, VI b) { return _mm_min_epi32(a, b); }
  static VI maxI(VI a, VI b) { return _mm_max_epi32(a, b); }
  static MI ltI(VI a, VI b) { return _mm_cmplt_epi32(a, b); }
  static VI selI(MI m, VI a, VI b) { return _mm_blendv_epi8(b, a, m); }
  static MI andMI(MI a, MI b) { return _mm_and_si128(a, b); }
  static MI notMI(MI a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
  static void stI8(uint8_t* p, VI v) {
    const __m128i b = _mm_packus_epi16(_mm_packus_epi32(v, v), _mm_setzero_si128());
    const int x = _mm_cvtsi128_si32(b);
    std::memcpy(p, &x, 4);
  }
  static void stI16(uint16_t* p, VI v) { _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packus_epi32(v, v)); }
  static constexpr bool kGather = false;
  template <int K>
  static VI srlI(VI v) { return _mm_srli_epi32(v, K); }
  static bool allMI(MI m) { return _mm_movemask_ps(_mm_castsi128_ps(m)) == 0xF; }
  template <int SC, typename T>
  static VI gather32(const T*, VI, MI) { return _mm_setzero_si128(); }
  static VF gatherF(const float*, VI) { return _mm_setzero_ps(); }
  static VI orI(VI a, VI b) { return _mm_or_si128(a, b); }
  static VI xorI(VI a, VI b) { return _mm_xor_si128(a, b); }
  static VI madd16(VI a, VI b) { return _mm_madd_epi16(a, b); }
  using VW = __m128i;
  static constexpr int NW = 8;
  static VW setw(int v) { return _mm_set1_epi16(short(v)); }
  static VW ldw(const int16_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static VW ldw8(const uint8_t* p) { return _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))); }
  static VW addw(VW a, VW b) { return _mm_add_epi16(a, b); }
  static VW subw(VW a, VW b) { return _mm_sub_epi16(a, b); }
  static VW mullow(VW a, VW b) { return _mm_mullo_epi16(a, b); }
  static VW mulhsw(VW a, VW b) { return _mm_mulhi_epi16(a, b); }
  static VW andw(VW a, VW b) { return _mm_and_si128(a, b); }
  template <int K>
  static VW sllw(VW v) { return _mm_slli_epi16(v, K); }
  template <int K>
  static VW sraw(VW v) { return _mm_srai_epi16(v, K); }
  template <int K>
  static VW srlw(VW v) { return _mm_srli_epi16(v, K); }
  static VW minsw(VW a, VW b) { return _mm_min_epi16(a, b); }
  static VW maxsw(VW a, VW b) { return _mm_max_epi16(a, b); }
  static VW packsw(VI a, VI b) { return _mm_packs_epi32(a, b); }
  static VI widenLo(VW v) { return _mm_cvtepi16_epi32(v); }
  static VI widenHi(VW v) { return _mm_cvtepi16_epi32(_mm_srli_si128(v, 8)); }
  static void stw8(uint8_t* p, VW v) { _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packus_epi16(v, v)); }
  static int ld16at(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
  }
  static VW gatherw8(const uint8_t* b, VI i0, VI i1) {
    __m128i r = _mm_cvtsi32_si128(ld16at(b + _mm_cvtsi128_si32(i0)));
    r = _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i0, 1)), 1);
    r = _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i0, 2)), 2);
    r = _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i0, 3)), 3);
    r = _mm_insert_epi16(r, ld16at(b + _mm_cvtsi128_si32(i1)), 4);
    r = _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i1, 1)), 5);
    r = _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i1, 2)), 6);
    return _mm_insert_epi16(r, ld16at(b + _mm_extract_epi32(i1, 3)), 7);
  }
  static VW iotaw() { return _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7); }
  static VW ltw(VW a, VW b) { return _mm_cmplt_epi16(a, b); }
  static VW notw(VW a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
  static VW pairw(int lo, int hi) { return _mm_set1_epi32(int((unsigned(lo) & 0xFFFFu) | (unsigned(hi) << 16))); }
  static void maddPairs(VW a, VW b, VW w, VI& lo, VI& hi) {
    lo = _mm_madd_epi16(_mm_unpacklo_epi16(a, b), w);
    hi = _mm_madd_epi16(_mm_unpackhi_epi16(a, b), w);
  }
  static VW gatherw8u(const uint8_t* b, VI i0, VI i1) { return gatherw8(b, i0, i1); }
  static VF cvtIF(VI v) { return _mm_cvtepi32_ps(v); }
  static VI cvtFI(VF v) { return _mm_cvttps_epi32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsSSE41() { return simdKernels(); }

}  // namespace awarp
