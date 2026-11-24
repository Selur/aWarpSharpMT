// SPDX-License-Identifier: GPL-2.0-or-later
// Runtime instruction set selection.
#include "kernels.hpp"

#if defined(AWARP_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace awarp {

#if defined(AWARP_X86)
namespace {

void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int t[4];
  __cpuidex(t, leaf, sub);
  for (int i = 0; i < 4; i++) r[i] = unsigned(t[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

struct Features {
  bool sse41 = false, avx2 = false;
};

Features detect() {
  Features f;
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  cpuid(1, 0, r);
  f.sse41 = (r[2] >> 19) & 1;
  const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
  const bool ymm = osxsave && avx && (xgetbv0() & 6) == 6;  // OS saves XMM and YMM state
  if (ymm && max_leaf >= 7) {
    cpuid(7, 0, r);
    f.avx2 = (r[1] >> 5) & 1;
  }
  return f;
}

}  // namespace
#endif

bool selectIsa(int opt, Isa* isa) {
  if (opt == 0) {
    *isa = Isa::C;
    return true;
  }
#if defined(AWARP_X86)
  static const Features f = detect();
  switch (opt) {
    case -1:
      *isa = f.avx2 ? Isa::AVX2 : (f.sse41 ? Isa::SSE41 : Isa::C);
      return true;
    case 1: *isa = Isa::SSE41; return f.sse41;
    case 2: *isa = Isa::AVX2; return f.avx2;
    default: return false;
  }
#elif defined(AWARP_NEON)
  if (opt == -1 || opt == 3) {
    *isa = Isa::NEON;
    return true;
  }
  return false;
#else
  if (opt == -1) {
    *isa = Isa::C;
    return true;
  }
  return false;
#endif
}

const Kernels& kernelsFor(Isa isa) {
  switch (isa) {
#if defined(AWARP_X86)
    case Isa::SSE41: return kernelsSSE41();
    case Isa::AVX2: return kernelsAVX2();
#elif defined(AWARP_NEON)
    case Isa::NEON: return kernelsNEON();
#endif
    default: return kernelsC();
  }
}

const char* isaName(Isa isa) {
  switch (isa) {
    case Isa::SSE41: return "SSE4.1";
    case Isa::AVX2: return "AVX2";
    case Isa::NEON: return "NEON";
    default: return "C";
  }
}

}  // namespace awarp
