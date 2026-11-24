// SPDX-License-Identifier: GPL-2.0-or-later
// Row kernels of aWarpSharpMT. Every instruction set gives bit-identical results to the C reference (scalar.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace awarp {

// Integer samples carry thresholds and pixel_max as int, float samples as float.
template <typename T>
using Num = typename std::conditional<std::is_same<T, float>::value, float, int>::type;

// One output row of the warp. src is row 0 of the source plane (aWarp4: four times the size), eu/ec/ed the mask rows
// above, at and below y (clamped at the plane edges).
template <typename T>
struct WarpRow {
  const T* src = nullptr;
  ptrdiff_t src_stride = 0;  // samples
  const T* eu = nullptr;
  const T* ec = nullptr;
  const T* ed = nullptr;
  T* dst = nullptr;
  int y = 0, w = 0, h = 0;   // output plane
  int depthH = 0, depthV = 0;  // integer: depth << (21 - bits)
  float depthHf = 0.0f, depthVf = 0.0f;  // float: displacement in pixels per unit of mask difference
  Num<T> pixel_max = 0;
};

// Columns [x0, x1) of one row. The kernels read one sample (sobel) or two (blur r2) left of x0 and right of x1 - 1,
// which the caller keeps inside the row; the radius 6 blur and the warp handle the whole row.
template <typename T>
struct KernelSet {
  // Edge mask from the rows above, at and below.
  void (*sobel)(const T* up, const T* mid, const T* down, T* dst, int x0, int x1, Num<T> thresh, Num<T> pixel_max);
  // The same with MarcFD's formula (mask=1); wrap: 2^bits - 1 (integer), 256/255 (float).
  void (*sobelOrig)(const T* up, const T* mid, const T* down, T* dst, int x0, int x1, Num<T> thresh, Num<T> pixel_max,
                    Num<T> wrap);
  // Horizontal blur: radius 6 for a whole row of at least 12 samples, radius 2 for the symmetric part [x0, x1).
  void (*blurH6)(const T* src, T* dst, int w);
  void (*blurH2)(const T* src, T* dst, int x0, int x1);
  // Vertical blur from 13 / 5 row pointers (rows[6] / rows[2] is the output row), columns [0, w).
  void (*blurV6)(const T* const* rows, T* dst, int w);
  void (*blurV2)(const T* const* rows, T* dst, int w);
  // Warp, source at the size of the mask (warp0) or four times the size (warp2).
  void (*warp0)(const WarpRow<T>& r, int x0, int x1);
  void (*warp2)(const WarpRow<T>& r, int x0, int x1);
};

struct Kernels {
  KernelSet<uint8_t> u8;
  KernelSet<uint16_t> u16;
  KernelSet<float> f32;

  template <typename T>
  const KernelSet<T>& get() const;
};

template <>
inline const KernelSet<uint8_t>& Kernels::get<uint8_t>() const { return u8; }
template <>
inline const KernelSet<uint16_t>& Kernels::get<uint16_t>() const { return u16; }
template <>
inline const KernelSet<float>& Kernels::get<float>() const { return f32; }

enum class Isa { C, SSE41, AVX2, NEON };

const Kernels& kernelsC();
#if defined(_M_X64) || defined(__x86_64__)
#define AWARP_X86 1
const Kernels& kernelsSSE41();
const Kernels& kernelsAVX2();
#elif defined(__aarch64__) || defined(_M_ARM64)
#define AWARP_NEON 1
const Kernels& kernelsNEON();
#endif

// opt: -1 = best available, 0 = C, 1 = SSE4.1, 2 = AVX2, 3 = NEON. Returns false if not available.
bool selectIsa(int opt, Isa* isa);
const Kernels& kernelsFor(Isa isa);
const char* isaName(Isa isa);

}  // namespace awarp
