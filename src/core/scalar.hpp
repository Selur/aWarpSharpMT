// SPDX-License-Identifier: GPL-2.0-or-later
// Scalar reference formulas, shared by the C kernels and the column tails of the SIMD kernels. The anonymous namespace
// keeps every translation unit (built with its own instruction set flags) on its own copy.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "kernels.hpp"

namespace awarp {
namespace {

// Rounded average (pavgb/pavgw); float: exact mean.
inline int avg(int a, int b) { return (a + b + 1) >> 1; }
inline float avg(float a, float b) { return (a + b) * 0.5f; }

inline int absd(int a, int b) { return std::abs(a - b); }
inline float absd(float a, float b) { return std::fabs(a - b); }

template <typename T>
inline Num<T> load(const T* p, int i) {
  return Num<T>(p[i]);
}

// Edge strength of one sample from its 3x3 neighbourhood (u = row above, m = row, d = row below; l, c, r = columns).
// The saturating steps of the original (paddusb) give min(value, thresh) for any thresh <= pixel_max.
template <typename N>
inline N sobelPx(N ul, N uc, N ur, N ml, N mr, N dl, N dc, N dr, N thresh, N pmax) {
  const N v = absd(avg(uc, avg(ul, ur)), avg(dc, avg(dl, dr)));
  const N h = absd(avg(ml, avg(dl, ul)), avg(mr, avg(dr, ur)));
  N a = std::min(v + h, pmax);
  a = std::min(a + std::max(h, v), pmax);
  a = std::min(std::min(a + a, pmax) + a, pmax);
  a = std::min(a + a, pmax);
  return std::min(a, thresh);
}

template <typename T>
void sobelRowScalar(const T* u, const T* m, const T* d, T* dst, int x0, int x1, Num<T> thresh, Num<T> pmax) {
  for (int x = x0; x < x1; x++)
    dst[x] = T(sobelPx(load(u, x - 1), load(u, x), load(u, x + 1), load(m, x - 1), load(m, x + 1), load(d, x - 1),
                       load(d, x), load(d, x + 1), thresh, pmax));
}

// Radius 6 blur of a centre sample c and its six pairs of neighbours at distance 1..6 (p[k], q[k]).
template <typename N>
inline N blur6(N c, const N* p, const N* q) {
  const N a12 = avg(avg(p[0], q[0]), avg(p[1], q[1]));
  const N a34 = avg(avg(p[2], q[2]), avg(p[3], q[3]));
  const N a56 = avg(avg(p[4], q[4]), avg(p[5], q[5]));
  const N a012 = avg(c, a12);
  const N a3456 = avg(a34, a56);
  return avg(a012, avg(a012, a3456));
}

// Radius 2 blur: c, pairs at distance 1 (p1, q1) and 2 (p2, q2).
template <typename N>
inline N blur2(N c, N p1, N q1, N p2, N q2) {
  N a = avg(avg(p2, q2), c);
  a = avg(a, c);
  return avg(a, avg(p1, q1));
}

// Horizontal radius 6 blur of column x; the six columns at each side are mirrored from the other side where they
// leave the row (the edge formula of aWarpSharp2).
template <typename T>
inline Num<T> blurH6At(const T* s, int x, int w) {
  Num<T> p[6], q[6];
  for (int k = 1; k <= 6; k++) {
    const int l = x - k >= 0 ? x - k : x + k;
    const int r = x + k < w ? x + k : x - k;
    p[k - 1] = load(s, l);
    q[k - 1] = load(s, r);
  }
  return blur6(load(s, x), p, q);
}

template <typename T>
void blurH6RowScalar(const T* s, T* d, int x0, int x1) {
  for (int x = x0; x < x1; x++) {
    Num<T> p[6], q[6];
    for (int k = 0; k < 6; k++) {
      p[k] = load(s, x - k - 1);
      q[k] = load(s, x + k + 1);
    }
    d[x] = T(blur6(load(s, x), p, q));
  }
}

template <typename T>
void blurH2RowScalar(const T* s, T* d, int x0, int x1) {
  for (int x = x0; x < x1; x++)
    d[x] = T(blur2(load(s, x), load(s, x - 1), load(s, x + 1), load(s, x - 2), load(s, x + 2)));
}

template <typename T>
void blurV6RowScalar(const T* const* rows, T* d, int x0, int x1) {
  for (int x = x0; x < x1; x++) {
    Num<T> p[6], q[6];
    for (int k = 0; k < 6; k++) {
      p[k] = load(rows[5 - k], x);
      q[k] = load(rows[7 + k], x);
    }
    d[x] = T(blur6(load(rows[6], x), p, q));
  }
}

template <typename T>
void blurV2RowScalar(const T* const* rows, T* d, int x0, int x1) {
  for (int x = x0; x < x1; x++)
    d[x] = T(blur2(load(rows[2], x), load(rows[1], x), load(rows[3], x), load(rows[0], x), load(rows[4], x)));
}

// Warp of one integer sample: displacement from the mask gradient in 1/128 pixel (SEt's fixed point), bilinear
// interpolation with 7-bit weights. SMAGL = 2 reads a source four times the size (aWarp4).
template <int SMAGL, typename T>
inline T warpPx(const WarpRow<T>& r, int x) {
  const int w = r.w;
  const int left = r.ec[x > 0 ? x - 1 : 0];
  const int right = r.ec[x < w - 1 ? x + 1 : w - 1];
  int hd = ((left - right) * r.depthH) >> 14;
  int vd = ((int(r.eu[x]) - int(r.ed[x])) * r.depthV) >> 14;
  vd = std::max(std::min(vd, (r.h - 1 - r.y) * 128 - 1), -r.y * 128);
  int rh = (hd << SMAGL) & 127;
  const int rv = (vd << SMAGL) & 127;
  hd >>= 7 - SMAGL;
  vd >>= 7 - SMAGL;
  hd += x << SMAGL;
  const int xmax = (w - 1) << SMAGL;
  if (!(hd >= 0 && hd < xmax)) rh = 0;
  hd = std::max(std::min(hd, xmax), 0);
  const T* s0 = r.src + (ptrdiff_t(r.y << SMAGL) + vd) * r.src_stride + hd;
  // A plane of one row (source at the mask size): no row below, its weight is 0 there.
  const T* s1 = (r.h << SMAGL) > 1 ? s0 + r.src_stride : s0;
  // Without a fraction the right neighbour (past the row end at the right edge) is not read.
  const int t = rh ? (s0[0] * (128 - rh) + s0[1] * rh + 64) >> 7 : s0[0];
  const int b = rh ? (s1[0] * (128 - rh) + s1[1] * rh + 64) >> 7 : s1[0];
  const int v = (t * (128 - rv) + b * rv + 64) >> 7;
  return T(std::min(std::max(v, 0), int(r.pixel_max)));
}

// Float warp: the same steps with exact fractions instead of 1/128 pixel.
template <int SMAGL>
inline float warpPxF(const WarpRow<float>& r, int x) {
  const int w = r.w;
  const float smag = float(1 << SMAGL);
  const float left = r.ec[x > 0 ? x - 1 : 0];
  const float right = r.ec[x < w - 1 ? x + 1 : w - 1];
  const float hd = (left - right) * r.depthHf;
  float vd = (r.eu[x] - r.ed[x]) * r.depthVf;
  vd = std::max(std::min(vd, float(r.h - 1 - r.y) - 1.0f / 128.0f), -float(r.y));
  const float hp = (float(x) + hd) * smag;
  const float vp = vd * smag;
  const float xmax = float((w - 1) << SMAGL);
  int ix;
  float fx;
  if (hp >= 0.0f && hp < xmax) {
    ix = int(std::floor(hp));
    fx = hp - float(ix);
  } else {
    ix = hp < 0.0f ? 0 : (w - 1) << SMAGL;
    fx = 0.0f;
  }
  const int iv = int(std::floor(vp));
  const float fy = vp - float(iv);
  const float* s0 = r.src + (ptrdiff_t(r.y << SMAGL) + iv) * r.src_stride + ix;
  const float* s1 = (r.h << SMAGL) > 1 ? s0 + r.src_stride : s0;
  const float t = fx != 0.0f ? s0[0] + (s0[1] - s0[0]) * fx : s0[0];
  const float b = fx != 0.0f ? s1[0] + (s1[1] - s1[0]) * fx : s1[0];
  return t + (b - t) * fy;
}

template <int SMAGL, typename T>
void warpRowScalar(const WarpRow<T>& r, int x0, int x1) {
  for (int x = x0; x < x1; x++) r.dst[x] = warpPx<SMAGL>(r, x);
}

template <int SMAGL>
void warpRowScalarF(const WarpRow<float>& r, int x0, int x1) {
  for (int x = x0; x < x1; x++) r.dst[x] = warpPxF<SMAGL>(r, x);
}

}  // namespace
}  // namespace awarp
