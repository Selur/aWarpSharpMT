// SPDX-License-Identifier: GPL-2.0-or-later
// NEON kernels for arm64 (16 samples per step at 8 bit, 8 at 9-16 bit, 4 in the warp and in float).
#include <arm_neon.h>

#include <algorithm>
#include <cstring>

#include "kernels.hpp"
#include "scalar.hpp"

namespace awarp {
namespace {

struct S {
  using V8 = uint8x16_t;
  using V16 = uint16x8_t;
  using VI = int32x4_t;
  using VF = float32x4_t;
  using MI = uint32x4_t;
  using MF = uint32x4_t;
  static constexpr int N8 = 16, N16 = 8, NI = 4;

  static V8 ld8(const uint8_t* p) { return vld1q_u8(p); }
  static void st8(uint8_t* p, V8 v) { vst1q_u8(p, v); }
  static V8 set8(int v) { return vdupq_n_u8(uint8_t(v)); }
  static V8 avg8(V8 a, V8 b) { return vrhaddq_u8(a, b); }
  static V8 absd8(V8 a, V8 b) { return vabdq_u8(a, b); }
  static V8 adds8(V8 a, V8 b) { return vqaddq_u8(a, b); }
  static V8 max8(V8 a, V8 b) { return vmaxq_u8(a, b); }
  static V8 min8(V8 a, V8 b) { return vminq_u8(a, b); }

  static V16 ld16(const uint16_t* p) { return vld1q_u16(p); }
  static void st16(uint16_t* p, V16 v) { vst1q_u16(p, v); }
  static V16 set16(int v) { return vdupq_n_u16(uint16_t(v)); }
  static V16 avg16(V16 a, V16 b) { return vrhaddq_u16(a, b); }
  static V16 absd16(V16 a, V16 b) { return vabdq_u16(a, b); }
  static V16 adds16(V16 a, V16 b) { return vqaddq_u16(a, b); }
  static V16 max16(V16 a, V16 b) { return vmaxq_u16(a, b); }
  static V16 min16(V16 a, V16 b) { return vminq_u16(a, b); }

  static VF ldf(const float* p) { return vld1q_f32(p); }
  static void stf(float* p, VF v) { vst1q_f32(p, v); }
  static VF setf(float v) { return vdupq_n_f32(v); }
  static VF addf(VF a, VF b) { return vaddq_f32(a, b); }
  static VF subf(VF a, VF b) { return vsubq_f32(a, b); }
  static VF mulf(VF a, VF b) { return vmulq_f32(a, b); }
  static VF minf(VF a, VF b) { return vminq_f32(a, b); }
  static VF maxf(VF a, VF b) { return vmaxq_f32(a, b); }
  static VF absf(VF v) { return vabsq_f32(v); }
  static VF floorf_(VF v) { return vrndmq_f32(v); }
  static MF ltf(VF a, VF b) { return vcltq_f32(a, b); }
  static MF gef(VF a, VF b) { return vcgeq_f32(a, b); }
  static VF self(MF m, VF a, VF b) { return vbslq_f32(m, a, b); }
  static MF andMF(MF a, MF b) { return vandq_u32(a, b); }
  static MI mfToMi(MF m) { return m; }

  static VI setI(int v) { return vdupq_n_s32(v); }
  static VI iotaI() {
    static const int32_t i[4] = {0, 1, 2, 3};
    return vld1q_s32(i);
  }
  static VI ld8I(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    const uint16x8_t w = vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(v)));
    return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w)));
  }
  static VI ld16I(const uint16_t* p) { return vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p))); }
  static VI ldI(const int32_t* p) { return vld1q_s32(p); }
  static void stI(int32_t* p, VI v) { vst1q_s32(p, v); }
  static VI addI(VI a, VI b) { return vaddq_s32(a, b); }
  static VI subI(VI a, VI b) { return vsubq_s32(a, b); }
  static VI mulI(VI a, VI b) { return vmulq_s32(a, b); }
  template <int K>
  static VI sraI(VI v) {
    if constexpr (K == 0)
      return v;
    else
      return vshrq_n_s32(v, K);
  }
  template <int K>
  static VI sllI(VI v) {
    if constexpr (K == 0)
      return v;
    else
      return vshlq_n_s32(v, K);
  }
  static VI andI(VI a, VI b) { return vandq_s32(a, b); }
  static VI minI(VI a, VI b) { return vminq_s32(a, b); }
  static VI maxI(VI a, VI b) { return vmaxq_s32(a, b); }
  static MI ltI(VI a, VI b) { return vcltq_s32(a, b); }
  static VI selI(MI m, VI a, VI b) { return vbslq_s32(m, a, b); }
  static MI andMI(MI a, MI b) { return vandq_u32(a, b); }
  static MI notMI(MI a) { return vmvnq_u32(a); }
  static void stI8(uint8_t* p, VI v) {
    const uint16x4_t h = vqmovun_s32(v);
    const uint8x8_t b = vqmovn_u16(vcombine_u16(h, h));
    const uint32_t x = vget_lane_u32(vreinterpret_u32_u8(b), 0);
    std::memcpy(p, &x, 4);
  }
  static void stI16(uint16_t* p, VI v) { vst1_u16(p, vqmovun_s32(v)); }
  static constexpr bool kGather = false;
  template <int K>
  static VI srlI(VI v) { return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(v), K)); }
  static bool allMI(MI m) { return vminvq_u32(m) == 0xFFFFFFFFu; }
  template <int SC, typename T>
  static VI gather32(const T*, VI, MI) { return vdupq_n_s32(0); }
  static VF gatherF(const float*, VI) { return vdupq_n_f32(0.0f); }
  static VI orI(VI a, VI b) { return vorrq_s32(a, b); }
  static VI xorI(VI a, VI b) { return veorq_s32(a, b); }
  static VI madd16(VI a, VI b) {
    const int16x8_t x = vreinterpretq_s16_s32(a), y = vreinterpretq_s16_s32(b);
    return vpaddq_s32(vmull_s16(vget_low_s16(x), vget_low_s16(y)), vmull_high_s16(x, y));
  }
  using VW = int16x8_t;
  static constexpr int NW = 8;
  static VW setw(int v) { return vdupq_n_s16(int16_t(v)); }
  static VW ldw(const int16_t* p) { return vld1q_s16(p); }
  static VW ldw8(const uint8_t* p) { return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p))); }
  static VW addw(VW a, VW b) { return vaddq_s16(a, b); }
  static VW subw(VW a, VW b) { return vsubq_s16(a, b); }
  static VW mullow(VW a, VW b) { return vmulq_s16(a, b); }
  // pmulhw: the high 16 bits of the 32-bit products (vqdmulh would double and saturate).
  static VW mulhsw(VW a, VW b) {
    const int32x4_t l = vmull_s16(vget_low_s16(a), vget_low_s16(b)), h = vmull_high_s16(a, b);
    return vuzp2q_s16(vreinterpretq_s16_s32(l), vreinterpretq_s16_s32(h));
  }
  static VW andw(VW a, VW b) { return vandq_s16(a, b); }
  template <int K>
  static VW sllw(VW v) {
    if constexpr (K == 0)
      return v;
    else
      return vshlq_n_s16(v, K);
  }
  template <int K>
  static VW sraw(VW v) {
    if constexpr (K == 0)
      return v;
    else
      return vshrq_n_s16(v, K);
  }
  template <int K>
  static VW srlw(VW v) {
    return vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(v), K));
  }
  static VW minsw(VW a, VW b) { return vminq_s16(a, b); }
  static VW maxsw(VW a, VW b) { return vmaxq_s16(a, b); }
  static VW packsw(VI a, VI b) { return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)); }
  static VW packsw(MI a, MI b) { return packsw(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b)); }
  static VI widenLo(VW v) { return vmovl_s16(vget_low_s16(v)); }
  static VI widenHi(VW v) { return vmovl_high_s16(v); }
  static void stw8(uint8_t* p, VW v) { vst1_u8(p, vqmovun_s16(v)); }
  static int16_t ld16at(const uint8_t* p) {
    int16_t v;
    std::memcpy(&v, p, 2);
    return v;
  }
  static VW gatherw8(const uint8_t* b, VI i0, VI i1) {
    VW r = vdupq_n_s16(ld16at(b + vgetq_lane_s32(i0, 0)));
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i0, 1)), r, 1);
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i0, 2)), r, 2);
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i0, 3)), r, 3);
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i1, 0)), r, 4);
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i1, 1)), r, 5);
    r = vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i1, 2)), r, 6);
    return vsetq_lane_s16(ld16at(b + vgetq_lane_s32(i1, 3)), r, 7);
  }
  static VW iotaw() {
    static const int16_t i[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    return vld1q_s16(i);
  }
  static VW ltw(VW a, VW b) { return vreinterpretq_s16_u16(vcltq_s16(a, b)); }
  static VW notw(VW a) { return vmvnq_s16(a); }
  static VW pairw(int lo, int hi) {
    return vreinterpretq_s16_u32(vdupq_n_u32((unsigned(lo) & 0xFFFFu) | (unsigned(hi) << 16)));
  }
  static void maddPairs(VW a, VW b, VW w, VI& lo, VI& hi) {
    lo = madd16(vreinterpretq_s32_s16(vzip1q_s16(a, b)), vreinterpretq_s32_s16(w));
    hi = madd16(vreinterpretq_s32_s16(vzip2q_s16(a, b)), vreinterpretq_s32_s16(w));
  }
  static VW gatherw8u(const uint8_t* b, VI i0, VI i1) { return gatherw8(b, i0, i1); }
  static VF cvtIF(VI v) { return vcvtq_f32_s32(v); }
  static VI cvtFI(VF v) { return vcvtq_s32_f32(v); }
};

#include "kernels_simd.hpp"

}  // namespace

const Kernels& kernelsNEON() { return simdKernels(); }

}  // namespace awarp
