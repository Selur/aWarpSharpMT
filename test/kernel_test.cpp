// SPDX-License-Identifier: GPL-2.0-or-later
// Every SIMD tier of the CPU against the C kernels on random data: all kernels, 8/9-16 bit and float, many widths
// and ranges. Exit code 1 on the first difference.
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "core/kernels.hpp"

using namespace awarp;

namespace {

std::mt19937 rng(12345);

int rnd(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

template <typename T>
T sample(int bits) {
  if constexpr (std::is_same<T, float>::value) {
    (void)bits;
    return std::uniform_real_distribution<float>(-0.25f, 1.25f)(rng);
  } else {
    // Mostly random, sometimes the extremes (saturation, clamps).
    const int pmax = (1 << bits) - 1, r = rnd(0, 15);
    return T(r == 0 ? 0 : (r == 1 ? pmax : rnd(0, pmax)));
  }
}

// A row with a margin of 64 samples on both sides.
template <typename T>
struct Row {
  std::vector<T> v;
  Row(int w, int bits) : v(size_t(w) + 128) {
    for (auto& x : v) x = sample<T>(bits);
  }
  T* p() { return v.data() + 64; }
  void unit() {
    for (auto& x : v) x = T(std::uniform_real_distribution<float>(0.0f, 1.0f)(rng));
  }
};

template <typename T>
bool same(const T* a, const T* b, int n, const char* what, const char* isa, int w, int bits) {
  if (std::memcmp(a, b, size_t(n) * sizeof(T)) == 0) return true;
  for (int i = 0; i < n; i++)
    if (std::memcmp(a + i, b + i, sizeof(T)) != 0) {
      std::printf("FAIL %s %s width %d bits %d at x=%d: C %.9g, SIMD %.9g\n", isa, what, w, bits, i, double(a[i]),
                  double(b[i]));
      return false;
    }
  return true;
}

template <typename T>
bool testSet(const KernelSet<T>& c, const KernelSet<T>& s, const char* isa, int bits) {
  constexpr bool fl = std::is_same<T, float>::value;
  const Num<T> pmax = fl ? Num<T>(1) : Num<T>((1 << (fl ? 1 : bits)) - 1);
  // The widths past 8000 run the 8-bit warp with 32-bit positions (aWarp4 sources wider than 2^15 samples).
  for (int w = 1; w <= 8300; w = w < 80 ? w + 1 : (w < 300 ? w + 37 : (w < 8000 ? 8100 : w + 197))) {
    for (int rep = 0; rep < 4; rep++) {
      if (w >= 3) {
        Row<T> u(w, bits), m(w, bits), d(w, bits), oc(w, bits);
        Row<T> os = oc;
        const int x0 = rep == 0 ? 1 : rnd(1, w - 2), x1 = rep == 0 ? w - 1 : rnd(x0, w - 1);
        const Num<T> thr = fl ? Num<T>(float(rnd(0, 255)) / 255.0f) : Num<T>(rnd(0, 255) << (fl ? 0 : bits - 8));
        c.sobel(u.p(), m.p(), d.p(), oc.p(), x0, x1, thr, pmax);
        s.sobel(u.p(), m.p(), d.p(), os.p(), x0, x1, thr, pmax);
        if (!same(oc.p(), os.p(), w, "sobel", isa, w, bits)) return false;
      }
      if (w >= 12) {
        Row<T> src(w, bits), oc(w, bits);
        Row<T> os = oc;
        c.blurH6(src.p(), oc.p(), w);
        s.blurH6(src.p(), os.p(), w);
        if (!same(oc.p(), os.p(), w, "blurH6", isa, w, bits)) return false;
      }
      if (w >= 5) {
        Row<T> src(w, bits), oc(w, bits);
        Row<T> os = oc;
        c.blurH2(src.p(), oc.p(), 2, w - 2);
        s.blurH2(src.p(), os.p(), 2, w - 2);
        if (!same(oc.p(), os.p(), w, "blurH2", isa, w, bits)) return false;
      }
      {
        std::vector<Row<T>> rows;
        for (int i = 0; i < 13; i++) rows.emplace_back(w, bits);
        const T* rp[13];
        for (int i = 0; i < 13; i++) rp[i] = rows[size_t(rnd(0, 12))].p();
        Row<T> oc(w, bits);
        Row<T> os = oc;
        c.blurV6(rp, oc.p(), w);
        s.blurV6(rp, os.p(), w);
        if (!same(oc.p(), os.p(), w, "blurV6", isa, w, bits)) return false;
        c.blurV2(rp, oc.p(), w);
        s.blurV2(rp, os.p(), w);
        if (!same(oc.p(), os.p(), w, "blurV2", isa, w, bits)) return false;
      }
      for (int smagl = 0; smagl <= 2; smagl += 2) {
        const int h = rnd(1, 12), sw = w << smagl, sh = h << smagl, stride = sw + rnd(0, 40);
        std::vector<T> src(size_t(stride) * size_t(sh) + 64);
        for (auto& x : src) x = fl ? T(std::uniform_real_distribution<float>(0.0f, 1.0f)(rng)) : sample<T>(bits);
        Row<T> eu(w, bits), ec(w, bits), ed(w, bits), oc(w, bits);
        if (fl) {
          eu.unit();
          ec.unit();
          ed.unit();
        }
        Row<T> os = oc;
        WarpRow<T> r;
        r.src = src.data();
        r.src_stride = stride;
        r.eu = eu.p();
        r.ec = ec.p();
        r.ed = ed.p();
        r.w = w;
        r.h = h;
        r.y = rnd(0, h - 1);
        r.pixel_max = pmax;
        const int dh = rnd(-128, 127), dv = rnd(-128, 127);
        r.depthHf = float(dh);
        r.depthVf = float(dv);
        if (!fl) {
          r.depthH = dh * (1 << (21 - bits));
          r.depthV = dv * (1 << (21 - bits));
        }
        r.dst = oc.p();
        (smagl ? c.warp2 : c.warp0)(r, 0, w);
        r.dst = os.p();
        (smagl ? s.warp2 : s.warp0)(r, 0, w);
        if (!same(oc.p(), os.p(), w, smagl ? "warp2" : "warp0", isa, w, bits)) return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  const Kernels& c = kernelsC();
  int tested = 0;
  for (int opt = 1; opt <= 3; opt++) {
    Isa isa;
    if (!selectIsa(opt, &isa)) continue;
    const Kernels& s = kernelsFor(isa);
    const char* name = isaName(isa);
    bool ok = testSet(c.u8, s.u8, name, 8);
    for (int bits : {9, 10, 12, 14, 16}) ok = ok && testSet(c.u16, s.u16, name, bits);
    ok = ok && testSet(c.f32, s.f32, name, 32);
    if (!ok) return 1;
    std::printf("%s: all kernels identical to C\n", name);
    tested++;
  }
  if (!tested) std::printf("no SIMD tier on this CPU, nothing to compare\n");
  return 0;
}
