// SPDX-License-Identifier: GPL-2.0-or-later
// C reference kernels.
#include "kernels.hpp"
#include "scalar.hpp"

namespace awarp {
namespace {

template <typename T>
void sobel(const T* u, const T* m, const T* d, T* dst, int x0, int x1, Num<T> thresh, Num<T> pmax) {
  sobelRowScalar(u, m, d, dst, x0, x1, thresh, pmax);
}
template <typename T>
void sobelOrig(const T* u, const T* m, const T* d, T* dst, int x0, int x1, Num<T> thresh, Num<T> pmax, Num<T> wrap) {
  sobelOrigRowScalar(u, m, d, dst, x0, x1, thresh, pmax, wrap);
}
template <typename T>
void blurV6(const T* const* rows, T* d, int w) {
  blurV6RowScalar(rows, d, 0, w);
}
template <typename T>
void blurV2(const T* const* rows, T* d, int w) {
  blurV2RowScalar(rows, d, 0, w);
}

template <typename T>
KernelSet<T> intSet() {
  KernelSet<T> k;
  k.sobel = sobel<T>;
  k.sobelOrig = sobelOrig<T>;
  k.blurH6 = blurH6RowFull<T>;
  k.blurH2 = blurH2RowScalar<T>;
  k.blurV6 = blurV6<T>;
  k.blurV2 = blurV2<T>;
  k.warp0 = warpRowScalar<0, T>;
  k.warp2 = warpRowScalar<2, T>;
  return k;
}

KernelSet<float> floatSet() {
  KernelSet<float> k;
  k.sobel = sobel<float>;
  k.sobelOrig = sobelOrig<float>;
  k.blurH6 = blurH6RowFull<float>;
  k.blurH2 = blurH2RowScalar<float>;
  k.blurV6 = blurV6<float>;
  k.blurV2 = blurV2<float>;
  k.warp0 = warpRowScalarF<0>;
  k.warp2 = warpRowScalarF<2>;
  return k;
}

}  // namespace

const Kernels& kernelsC() {
  static const Kernels k = {intSet<uint8_t>(), intSet<uint16_t>(), floatSet()};
  return k;
}

}  // namespace awarp
