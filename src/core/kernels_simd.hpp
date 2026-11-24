// SPDX-License-Identifier: GPL-2.0-or-later
// Generic SIMD kernels of aWarpSharpMT, written once against a small vector interface S (kernels_sse41.cpp,
// kernels_avx2.cpp, kernels_neon.cpp). Include inside an anonymous namespace after defining S.
//
// Bit-exact to the C reference (scalar.hpp): the same operations in the same order. Sobel and blur run in 8/16-bit
// lanes (rounded averages as pavgb/pavgw; the saturating adds at 255/65535 give min(value, thresh) like the C clamps
// at pixel_max, see sobelPx), float in float lanes. The warp computes the displacement in 32-bit lanes, gathers the
// four source samples per pixel with scalar loads and interpolates in 32-bit (or float) lanes. Each range runs in
// vectors, the last one overlapping the previous; ranges shorter than one vector run the scalar code.
//
// Interface S:
//   V8 (N8 x u8):  ld8, st8, set8, avg8, absd8, adds8, max8, min8
//   V16 (N16 x u16): ld16, st16, set16, avg16, absd16, adds16, max16, min16
//   VF (NI x f32): ldf, stf, setf, addf, subf, mulf, minf, maxf, absf, floorf_, ltf (mask MF), gef, self (m ? a : b)
//   VI (NI x i32): setI, iotaI (0, 1, ...), ld8I / ld16I (widening loads of NI samples), ldI / stI (int32 arrays),
//                  addI, subI, mulI, sraI<k>, sllI<k>, andI, minI, maxI, ltI (mask MI), selI (m ? a : b),
//                  stI8 / stI16 (narrowing stores of values in range), cvtIF, cvtFI (truncating), mfToMi, srlI<k>,
//                  allMI (every lane set), andMI, notMI, orI, xorI, madd16 (pmaddwd: signed 16-bit pairs)
//   VW (NW = 2 * NI x i16): setw, ldw, ldw8 (widening load of NW bytes), addw, subw, mullow, mulhsw (signed high
//                  half), andw, sllw<k>, sraw<k>, srlw<k>, minsw, maxsw, packsw (two int32 masks or values, signed
//                  saturation), widenLo / widenHi (sign extension to VI), stw8 (NW bytes, unsigned saturation),
//                  gatherw8 (16-bit reads at base + idx bytes for both halves; may read up to 4 bytes), iotaw,
//                  ltw / notw (word masks), pairw (lo | hi << 16 in every 32-bit lane), maddPairs (pmaddwd of the
//                  interleaved low and high halves of two VW), gatherw8u (gatherw8 for indices in maddPairs order)
//   kGather: true = gather32<bytes>(base, idx, mask) (32-bit reads at base + idx samples, masked lanes 0) and
//            gatherF(base, idx) are hardware gathers and used; false = scalar loads
#pragma once

template <typename F>
inline void forRange(int x0, int x1, int n, F&& step) {
  int x = x0;
  for (; x + n <= x1; x += n) step(x);
  if (x < x1) step(x1 - n);
}

// ---------------------------------------------------------------------------------------------------------------- sobel

template <typename V, typename AVG, typename ABSD, typename ADDS, typename MAX, typename MIN>
inline V sobelVec(V ul, V uc, V ur, V ml, V mr, V dl, V dc, V dr, V thr, AVG avg, ABSD absd, ADDS adds, MAX vmax, MIN vmin) {
  const V v = absd(avg(uc, avg(ul, ur)), avg(dc, avg(dl, dr)));
  const V h = absd(avg(ml, avg(dl, ul)), avg(mr, avg(dr, ur)));
  V a = adds(v, h);
  a = adds(a, vmax(h, v));
  a = adds(adds(a, a), a);
  a = adds(a, a);
  return vmin(a, thr);
}

void sobel8(const uint8_t* u, const uint8_t* m, const uint8_t* d, uint8_t* dst, int x0, int x1, int thresh, int pmax) {
  constexpr int N = S::N8;
  if (x1 - x0 < N) return sobelRowScalar(u, m, d, dst, x0, x1, thresh, pmax);
  const typename S::V8 thr = S::set8(thresh);
  forRange(x0, x1, N, [&](int x) {
    S::st8(dst + x, sobelVec(S::ld8(u + x - 1), S::ld8(u + x), S::ld8(u + x + 1), S::ld8(m + x - 1), S::ld8(m + x + 1),
                             S::ld8(d + x - 1), S::ld8(d + x), S::ld8(d + x + 1), thr, S::avg8, S::absd8, S::adds8,
                             S::max8, S::min8));
  });
}

void sobel16(const uint16_t* u, const uint16_t* m, const uint16_t* d, uint16_t* dst, int x0, int x1, int thresh,
             int pmax) {
  constexpr int N = S::N16;
  if (x1 - x0 < N) return sobelRowScalar(u, m, d, dst, x0, x1, thresh, pmax);
  const typename S::V16 thr = S::set16(thresh);
  forRange(x0, x1, N, [&](int x) {
    S::st16(dst + x, sobelVec(S::ld16(u + x - 1), S::ld16(u + x), S::ld16(u + x + 1), S::ld16(m + x - 1),
                              S::ld16(m + x + 1), S::ld16(d + x - 1), S::ld16(d + x), S::ld16(d + x + 1), thr,
                              S::avg16, S::absd16, S::adds16, S::max16, S::min16));
  });
}

inline typename S::VF avgF(typename S::VF a, typename S::VF b) { return S::mulf(S::addf(a, b), S::setf(0.5f)); }
inline typename S::VF absdF(typename S::VF a, typename S::VF b) { return S::absf(S::subf(a, b)); }

void sobelF(const float* u, const float* m, const float* d, float* dst, int x0, int x1, float thresh, float pmax) {
  using VF = typename S::VF;
  constexpr int N = S::NI;
  if (x1 - x0 < N) return sobelRowScalar(u, m, d, dst, x0, x1, thresh, pmax);
  const VF thr = S::setf(thresh), pm = S::setf(pmax);
  // The C formula with min(.., pixel_max) in place of the saturating adds.
  auto adds = [&](VF a, VF b) { return S::minf(S::addf(a, b), pm); };
  forRange(x0, x1, N, [&](int x) {
    S::stf(dst + x, sobelVec(S::ldf(u + x - 1), S::ldf(u + x), S::ldf(u + x + 1), S::ldf(m + x - 1), S::ldf(m + x + 1),
                             S::ldf(d + x - 1), S::ldf(d + x), S::ldf(d + x + 1), thr, avgF, absdF, adds, S::maxf,
                             S::minf));
  });
}

// ----------------------------------------------------------------------------------------------------------------- blur

template <typename V, typename AVG>
inline V blur6Vec(V c, const V* p, const V* q, AVG avg) {
  const V a12 = avg(avg(p[0], q[0]), avg(p[1], q[1]));
  const V a34 = avg(avg(p[2], q[2]), avg(p[3], q[3]));
  const V a56 = avg(avg(p[4], q[4]), avg(p[5], q[5]));
  const V a012 = avg(c, a12);
  const V a3456 = avg(a34, a56);
  return avg(a012, avg(a012, a3456));
}

template <typename V, typename AVG>
inline V blur2Vec(V c, V p1, V q1, V p2, V q2, AVG avg) {
  V a = avg(avg(p2, q2), c);
  a = avg(a, c);
  return avg(a, avg(p1, q1));
}

// Per sample type: vector width, load, store, average.
template <typename T>
struct Lanes;
template <>
struct Lanes<uint8_t> {
  static constexpr int N = S::N8;
  static typename S::V8 ld(const uint8_t* p) { return S::ld8(p); }
  static void st(uint8_t* p, typename S::V8 v) { S::st8(p, v); }
  static typename S::V8 avg(typename S::V8 a, typename S::V8 b) { return S::avg8(a, b); }
};
template <>
struct Lanes<uint16_t> {
  static constexpr int N = S::N16;
  static typename S::V16 ld(const uint16_t* p) { return S::ld16(p); }
  static void st(uint16_t* p, typename S::V16 v) { S::st16(p, v); }
  static typename S::V16 avg(typename S::V16 a, typename S::V16 b) { return S::avg16(a, b); }
};
template <>
struct Lanes<float> {
  static constexpr int N = S::NI;
  static typename S::VF ld(const float* p) { return S::ldf(p); }
  static void st(float* p, typename S::VF v) { S::stf(p, v); }
  static typename S::VF avg(typename S::VF a, typename S::VF b) { return avgF(a, b); }
};

template <typename T>
void blurH6(const T* s, T* d, int x0, int x1) {
  using L = Lanes<T>;
  if (x1 - x0 < L::N) return blurH6RowScalar(s, d, x0, x1);
  forRange(x0, x1, L::N, [&](int x) {
    decltype(L::ld(s)) p[6], q[6];
    for (int k = 0; k < 6; k++) {
      p[k] = L::ld(s + x - k - 1);
      q[k] = L::ld(s + x + k + 1);
    }
    L::st(d + x, blur6Vec(L::ld(s + x), p, q, L::avg));
  });
}

template <typename T>
void blurH2(const T* s, T* d, int x0, int x1) {
  using L = Lanes<T>;
  if (x1 - x0 < L::N) return blurH2RowScalar(s, d, x0, x1);
  forRange(x0, x1, L::N, [&](int x) {
    L::st(d + x, blur2Vec(L::ld(s + x), L::ld(s + x - 1), L::ld(s + x + 1), L::ld(s + x - 2), L::ld(s + x + 2), L::avg));
  });
}

template <typename T>
void blurV6(const T* const* rows, T* d, int w) {
  using L = Lanes<T>;
  if (w < L::N) return blurV6RowScalar(rows, d, 0, w);
  forRange(0, w, L::N, [&](int x) {
    decltype(L::ld(d)) p[6], q[6];
    for (int k = 0; k < 6; k++) {
      p[k] = L::ld(rows[5 - k] + x);
      q[k] = L::ld(rows[7 + k] + x);
    }
    L::st(d + x, blur6Vec(L::ld(rows[6] + x), p, q, L::avg));
  });
}

template <typename T>
void blurV2(const T* const* rows, T* d, int w) {
  using L = Lanes<T>;
  if (w < L::N) return blurV2RowScalar(rows, d, 0, w);
  forRange(0, w, L::N, [&](int x) {
    L::st(d + x, blur2Vec(L::ld(rows[2] + x), L::ld(rows[1] + x), L::ld(rows[3] + x), L::ld(rows[0] + x),
                          L::ld(rows[4] + x), L::avg));
  });
}

// ----------------------------------------------------------------------------------------------------------------- warp

template <typename T>
inline typename S::VI ldWiden(const T* p);
template <>
inline typename S::VI ldWiden<uint8_t>(const uint8_t* p) { return S::ld8I(p); }
template <>
inline typename S::VI ldWiden<uint16_t>(const uint16_t* p) { return S::ld16I(p); }

inline void stNarrow(uint8_t* p, typename S::VI v) { S::stI8(p, v); }
inline void stNarrow(uint16_t* p, typename S::VI v) { S::stI16(p, v); }

// Both horizontal neighbours of every lane with one load: s0 | s1 << (8 * sizeof(T)). The pair at the last sample of
// the source plane would read one sample past it; there (where the fraction is 0, so s1 does not count) s1 = s0.
// With hardware gathers (S::kGather) a 32-bit gather reads 4 / sizeof(T) samples; lanes whose read would leave the
// plane are masked out and read here.
template <typename T>
inline int32_t pairScalar(const T* p, bool both) {
  const int32_t a = p[0], b = both ? p[1] : p[0];
  return a | (b << (8 * int(sizeof(T))));
}

template <typename T>
inline void gatherPairs(const T* src, typename S::VI idxv, ptrdiff_t st, ptrdiff_t lim, typename S::VI& top,
                        typename S::VI& bot) {
  using VI = typename S::VI;
  constexpr int N = S::NI;
  alignas(64) int32_t idx[N], t[N], b[N];
  if constexpr (S::kGather) {
    constexpr int span = 4 / int(sizeof(T)) - 1;
    const VI limv = S::setI(int(lim - span) + 1);
    const auto okT = S::ltI(idxv, limv);
    const auto okB = S::ltI(S::addI(idxv, S::setI(int(st))), limv);
    top = S::template gather32<int(sizeof(T))>(src, idxv, okT);
    bot = S::template gather32<int(sizeof(T))>(src + st, idxv, okB);
    if (!S::allMI(S::andMI(okT, okB))) {
      S::stI(idx, idxv);
      S::stI(t, top);
      S::stI(b, bot);
      for (int i = 0; i < N; i++) {
        if (idx[i] + span > lim) t[i] = pairScalar(src + idx[i], idx[i] + 1 <= lim);
        if (idx[i] + st + span > lim) b[i] = pairScalar(src + idx[i] + st, idx[i] + st + 1 <= lim);
      }
      top = S::ldI(t);
      bot = S::ldI(b);
    }
    // A 32-bit read of 8-bit samples carries two more samples above the pair.
    if constexpr (sizeof(T) == 1) {
      top = S::andI(top, S::setI(0xFFFF));
      bot = S::andI(bot, S::setI(0xFFFF));
    }
  } else {
    S::stI(idx, idxv);
    for (int i = 0; i < N; i++) {
      const T* p = src + idx[i];
      t[i] = pairScalar(p, idx[i] + 1 <= lim);
      b[i] = pairScalar(p + st, idx[i] + st + 1 <= lim);
    }
    top = S::ldI(t);
    bot = S::ldI(b);
  }
}

// 8-bit warp in 16-bit lanes as SEt's code: displacement with pmulhw ((d << 7) * (depth << 8) >> 16 is
// (d * (depth << 13)) >> 14), weights and interpolation with pmullw (every sum stays below 2^15). Narrow: source rows
// below 2^15 samples, positions in 16-bit lanes and the offsets as one pmaddwd (vd * stride + hd) like the original;
// otherwise the positions run in 32-bit lanes. A vector whose reads would reach the last sample of the source plane
// runs the scalar formula. Columns [1, w-1) in vectors, the first and last column scalar.
template <int SMAGL, bool Narrow>
void warp8Row(const WarpRow<uint8_t>& r, int lo, int hi) {
  using VW = typename S::VW;
  using VI = typename S::VI;
  constexpr int N = S::NW, NI = S::NI;
  auto sat16 = [](int v) { return std::max(-32768, std::min(32767, v)); };
  const VW dH = S::setw(r.depthH >> 5), dV = S::setw(r.depthV >> 5);
  const VW vmin = S::setw(sat16(-r.y * 128)), vmax = S::setw(sat16((r.h - 1 - r.y) * 128 - 1));
  const VW m127 = S::setw(127), c128 = S::setw(128), c64 = S::setw(64), mFF = S::setw(255);
  const VW xmaxw = S::setw(Narrow ? (r.w - 1) << SMAGL : 0), zw = S::setw(0), iotaw = S::sllw<SMAGL>(S::iotaw());
  const VW so = S::pairw(Narrow ? int(r.src_stride) : 0, 1);
  const VI xmax = S::setI((r.w - 1) << SMAGL), zero = S::setI(0);
  const VI stride = S::setI(int(r.src_stride)), rowbase = S::setI(r.y << SMAGL);
  const VI iota = S::sllI<SMAGL>(S::iotaI());
  const ptrdiff_t st = r.src_stride;
  // Narrow offsets are relative to the source row of y.
  const uint8_t* base = Narrow ? r.src + ptrdiff_t(r.y << SMAGL) * st : r.src;
  const ptrdiff_t lim = ptrdiff_t((r.h << SMAGL) - 1) * st + ((r.w << SMAGL) - 1) - (base - r.src);
  const VI limv = S::setI(int(lim - 3) + 1), stv = S::setI(int(st));
  forRange(lo, hi, N, [&](int x) {
    VW hd = S::mulhsw(S::sllw<7>(S::subw(S::ldw8(r.ec + x - 1), S::ldw8(r.ec + x + 1))), dH);
    VW vd = S::mulhsw(S::sllw<7>(S::subw(S::ldw8(r.eu + x), S::ldw8(r.ed + x))), dV);
    vd = S::maxsw(S::minsw(vd, vmax), vmin);
    VW rh = S::andw(S::sllw<SMAGL>(hd), m127);
    const VW rv = S::andw(S::sllw<SMAGL>(vd), m127);
    hd = S::sraw<7 - SMAGL>(hd);
    vd = S::sraw<7 - SMAGL>(vd);
    VI id[2];
    if constexpr (Narrow) {
      hd = S::addw(hd, S::addw(S::setw(x << SMAGL), iotaw));
      rh = S::andw(rh, S::andw(S::ltw(hd, xmaxw), S::notw(S::ltw(hd, zw))));
      hd = S::maxsw(S::minsw(hd, xmaxw), zw);
      S::maddPairs(vd, hd, so, id[0], id[1]);
    } else {
      VI h32[2] = {S::widenLo(hd), S::widenHi(hd)}, v32[2] = {S::widenLo(vd), S::widenHi(vd)}, in[2];
      for (int k = 0; k < 2; k++) {
        h32[k] = S::addI(h32[k], S::addI(S::setI((x + k * NI) << SMAGL), iota));
        in[k] = S::andMI(S::ltI(h32[k], xmax), S::notMI(S::ltI(h32[k], zero)));
        h32[k] = S::maxI(S::minI(h32[k], xmax), zero);
        id[k] = S::addI(S::mulI(S::addI(rowbase, v32[k]), stride), h32[k]);
      }
      rh = S::andw(rh, S::packsw(in[0], in[1]));
    }
    // Pairs s0 | s1 << 8 of the rows above and below the position.
    if (!S::allMI(S::andMI(S::ltI(S::addI(id[0], stv), limv), S::ltI(S::addI(id[1], stv), limv)))) {
      for (int i = 0; i < N; i++) r.dst[x + i] = warpPx<SMAGL>(r, x + i);
      return;
    }
    const VW top = Narrow ? S::gatherw8u(base, id[0], id[1]) : S::gatherw8(base, id[0], id[1]);
    const VW bot = Narrow ? S::gatherw8u(base + st, id[0], id[1]) : S::gatherw8(base + st, id[0], id[1]);
    const VW irh = S::subw(c128, rh), irv = S::subw(c128, rv);
    const VW t = S::srlw<7>(S::addw(S::addw(S::mullow(S::andw(top, mFF), irh), S::mullow(S::srlw<8>(top), rh)), c64));
    const VW b = S::srlw<7>(S::addw(S::addw(S::mullow(S::andw(bot, mFF), irh), S::mullow(S::srlw<8>(bot), rh)), c64));
    S::stw8(r.dst + x, S::srlw<7>(S::addw(S::addw(S::mullow(t, irv), S::mullow(b, rv)), c64)));
  });
}

template <int SMAGL>
void warp8(const WarpRow<uint8_t>& r, int x0, int x1) {
  const int lo = std::max(x0, 1), hi = std::min(x1, r.w - 1);
  if (hi - lo < S::NW || (r.h << SMAGL) == 1) return warpRowScalar<SMAGL>(r, x0, x1);
  for (int x = x0; x < lo; x++) r.dst[x] = warpPx<SMAGL>(r, x);
  for (int x = hi; x < x1; x++) r.dst[x] = warpPx<SMAGL>(r, x);
  // Narrow: positions up to (w << SMAGL) plus the largest displacement (127 << SMAGL) and strides fit 16 bits.
  if ((r.w << SMAGL) + (128 << SMAGL) < 32768 && r.src_stride < 32768)
    warp8Row<SMAGL, true>(r, lo, hi);
  else
    warp8Row<SMAGL, false>(r, lo, hi);
}

// Columns [1, w-1) in vectors (they read the mask at x-1 and x+1 inside the row), the first and last column scalar.
template <int SMAGL, typename T>
void warpInt(const WarpRow<T>& r, int x0, int x1) {
  using VI = typename S::VI;
  constexpr int N = S::NI;
  constexpr int SH = 8 * int(sizeof(T));
  const int lo = std::max(x0, 1), hi = std::min(x1, r.w - 1);
  if (hi - lo < N || (r.h << SMAGL) == 1) return warpRowScalar<SMAGL>(r, x0, x1);
  for (int x = x0; x < lo; x++) r.dst[x] = warpPx<SMAGL>(r, x);
  for (int x = hi; x < x1; x++) r.dst[x] = warpPx<SMAGL>(r, x);

  const VI dH = S::setI(r.depthH), dV = S::setI(r.depthV);
  const VI vmin = S::setI(-r.y * 128), vmax = S::setI((r.h - 1 - r.y) * 128 - 1);
  const VI m127 = S::setI(127), zero = S::setI(0), c128 = S::setI(128);
  const VI lowmask = S::setI((1 << SH) - 1), hibyte = S::setI(0xFF00);
  const VI flip = S::setI(int(0x80008000u)), bias = S::setI((1 << 22) + 64);
  const VI xmax = S::setI((r.w - 1) << SMAGL), pmax = S::setI(int(r.pixel_max));
  const VI stride = S::setI(int(r.src_stride)), rowbase = S::setI(r.y << SMAGL);
  const VI iota = S::sllI<SMAGL>(S::iotaI());
  const ptrdiff_t st = r.src_stride;
  const ptrdiff_t lim = ptrdiff_t((r.h << SMAGL) - 1) * st + ((r.w << SMAGL) - 1);  // last sample of the source
  forRange(lo, hi, N, [&](int x) {
    VI hd = S::sraI<14>(S::mulI(S::subI(ldWiden(r.ec + x - 1), ldWiden(r.ec + x + 1)), dH));
    VI vd = S::sraI<14>(S::mulI(S::subI(ldWiden(r.eu + x), ldWiden(r.ed + x)), dV));
    vd = S::maxI(S::minI(vd, vmax), vmin);
    VI rh = S::andI(S::sllI<SMAGL>(hd), m127);
    const VI rv = S::andI(S::sllI<SMAGL>(vd), m127);
    hd = S::sraI<7 - SMAGL>(hd);
    vd = S::sraI<7 - SMAGL>(vd);
    hd = S::addI(hd, S::addI(S::setI(x << SMAGL), iota));
    // rh stays where 0 <= hd < xmax.
    const auto inside = S::andMI(S::ltI(hd, xmax), S::notMI(S::ltI(hd, zero)));
    rh = S::selI(inside, rh, zero);
    hd = S::maxI(S::minI(hd, xmax), zero);
    VI top, bot;
    gatherPairs(r.src, S::addI(S::mulI(S::addI(rowbase, vd), stride), hd), st, lim, top, bot);
    // a * (128 - f) + b * f as one signed 16-bit multiply-add of the pair (a - 2^15, b - 2^15) with (128 - f, f),
    // plus 2^15 * 128: exact for any 16-bit a and b.
    const VI wh = S::orI(S::subI(c128, rh), S::sllI<16>(rh)), wv = S::orI(S::subI(c128, rv), S::sllI<16>(rv));
    if constexpr (sizeof(T) == 1) {
      top = S::orI(S::andI(top, lowmask), S::sllI<8>(S::andI(top, hibyte)));
      bot = S::orI(S::andI(bot, lowmask), S::sllI<8>(S::andI(bot, hibyte)));
    }
    const VI t = S::sraI<7>(S::addI(S::madd16(S::xorI(top, flip), wh), bias));
    const VI b = S::sraI<7>(S::addI(S::madd16(S::xorI(bot, flip), wh), bias));
    const VI v = S::sraI<7>(S::addI(S::madd16(S::xorI(S::orI(t, S::sllI<16>(b)), flip), wv), bias));
    stNarrow(r.dst + x, S::maxI(S::minI(v, pmax), zero));
  });
}

template <int SMAGL>
void warpF(const WarpRow<float>& r, int x0, int x1) {
  using VI = typename S::VI;
  using VF = typename S::VF;
  constexpr int N = S::NI;
  const int lo = std::max(x0, 1), hi = std::min(x1, r.w - 1);
  if (hi - lo < N || (r.h << SMAGL) == 1) return warpRowScalarF<SMAGL>(r, x0, x1);
  for (int x = x0; x < lo; x++) r.dst[x] = warpPxF<SMAGL>(r, x);
  for (int x = hi; x < x1; x++) r.dst[x] = warpPxF<SMAGL>(r, x);

  const float smag = float(1 << SMAGL);
  const VF dH = S::setf(r.depthHf), dV = S::setf(r.depthVf);
  const VF vmax = S::setf(float(r.h - 1 - r.y) - 1.0f / 128.0f), vmin = S::setf(-float(r.y));
  const VF vs = S::setf(smag), xmaxf = S::setf(float((r.w - 1) << SMAGL)), zf = S::setf(0.0f);
  const VI xmaxi = S::setI((r.w - 1) << SMAGL), zero = S::setI(0), one = S::setI(1);
  const VI stride = S::setI(int(r.src_stride)), rowbase = S::setI(r.y << SMAGL), stv = S::setI(int(r.src_stride));
  const VF iota = S::cvtIF(S::iotaI());
  alignas(64) int32_t idx[N], nxt[N];
  alignas(64) float v00[N], v01[N], v10[N], v11[N];
  const ptrdiff_t st = r.src_stride;
  forRange(lo, hi, N, [&](int x) {
    const VF hd = S::mulf(S::subf(S::ldf(r.ec + x - 1), S::ldf(r.ec + x + 1)), dH);
    VF vd = S::mulf(S::subf(S::ldf(r.eu + x), S::ldf(r.ed + x)), dV);
    vd = S::maxf(S::minf(vd, vmax), vmin);
    const VF hp = S::mulf(S::addf(S::addf(S::setf(float(x)), iota), hd), vs);
    const VF vp = S::mulf(vd, vs);
    const auto inside = S::andMF(S::gef(hp, zf), S::ltf(hp, xmaxf));
    const VF fl = S::floorf_(hp);
    const VF fx = S::self(inside, S::subf(hp, fl), zf);
    const VI ix = S::selI(S::mfToMi(inside), S::cvtFI(fl), S::selI(S::mfToMi(S::ltf(hp, zf)), zero, xmaxi));
    const VF fv = S::floorf_(vp);
    const VF fy = S::subf(vp, fv);
    const VI i00 = S::addI(S::mulI(S::addI(rowbase, S::cvtFI(fv)), stride), ix);
    const VI i01 = S::addI(i00, S::selI(S::mfToMi(inside), one, zero));
    VF a, s01, c, s11;
    if constexpr (S::kGather) {
      // Every index lies inside the plane (the right neighbour only where the fraction is not 0).
      a = S::gatherF(r.src, i00);
      s01 = S::gatherF(r.src, i01);
      c = S::gatherF(r.src, S::addI(i00, stv));
      s11 = S::gatherF(r.src, S::addI(i01, stv));
    } else {
      S::stI(idx, i00);
      S::stI(nxt, i01);
      for (int i = 0; i < N; i++) {
        v00[i] = r.src[idx[i]];
        v01[i] = r.src[nxt[i]];
        v10[i] = r.src[idx[i] + st];
        v11[i] = r.src[nxt[i] + st];
      }
      a = S::ldf(v00);
      s01 = S::ldf(v01);
      c = S::ldf(v10);
      s11 = S::ldf(v11);
    }
    const VF t = S::addf(a, S::mulf(S::subf(s01, a), fx));
    const VF b = S::addf(c, S::mulf(S::subf(s11, c), fx));
    S::stf(r.dst + x, S::addf(t, S::mulf(S::subf(b, t), fy)));
  });
}

KernelSet<uint8_t> set8() {
  KernelSet<uint8_t> k;
  k.sobel = sobel8;
  k.blurH6 = blurH6<uint8_t>;
  k.blurH2 = blurH2<uint8_t>;
  k.blurV6 = blurV6<uint8_t>;
  k.blurV2 = blurV2<uint8_t>;
  k.warp0 = warp8<0>;
  k.warp2 = warp8<2>;
  return k;
}

KernelSet<uint16_t> set16() {
  KernelSet<uint16_t> k;
  k.sobel = sobel16;
  k.blurH6 = blurH6<uint16_t>;
  k.blurH2 = blurH2<uint16_t>;
  k.blurV6 = blurV6<uint16_t>;
  k.blurV2 = blurV2<uint16_t>;
  k.warp0 = warpInt<0, uint16_t>;
  k.warp2 = warpInt<2, uint16_t>;
  return k;
}

KernelSet<float> setF() {
  KernelSet<float> k;
  k.sobel = sobelF;
  k.blurH6 = blurH6<float>;
  k.blurH2 = blurH2<float>;
  k.blurV6 = blurV6<float>;
  k.blurV2 = blurV2<float>;
  k.warp0 = warpF<0>;
  k.warp2 = warpF<2>;
  return k;
}

const Kernels& simdKernels() {
  static const Kernels k = {set8(), set16(), setF()};
  return k;
}
