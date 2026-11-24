// SPDX-License-Identifier: GPL-2.0-or-later
// Host-independent part of aWarpSharpMT: parameters, formats and the per-frame processing of the six filters.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "kernels.hpp"

namespace awarp {

enum class Kind { WarpSharp2, WarpSharp, Sobel, Blur, Warp, Warp4 };

// Chroma modes of aWarpSharp2/aSobel/aBlur/aWarp/aWarp4.
enum Chroma {
  kChromaZero = 0,     // fill U/V with the neutral value
  kChromaIgnore = 1,   // "don't care": U/V are copied (aWarp4: point-sampled from the 4x clip)
  kChromaCopy = 2,     // copy U/V
  kChromaProcess = 3,  // process U/V with their own masks
  kChromaGuide = 4,    // warp U/V along the luma mask (warp filters only; sobel/blur treat it as 3)
  kChromaOnly = 5,     // 3 without processing luma
  kChromaGuideOnly = 6 // 4 without processing luma
};

struct Params {
  Kind kind = Kind::WarpSharp2;
  int thresh = 128, threshC = 128;   // 0..255 (8-bit scale)
  int blur = 2, blurV = 2;           // passes
  int blurC = 1, blurVC = 1;
  int type = 0;                      // 0: radius 6 blur, 1: radius 2 blur, 2: HQ blur of the original
  int depth = 16, depthV = 16;       // -128..127
  int depthC = 8, depthVC = 8;
  int chroma = 4;
  bool mpeg2 = false;                // cplace "MPEG2"
  // aWarpSharp(original=true): MarcFD's aWarpSharp.
  bool original = false;
  int origDepth = 0;                 // depth * blurlevel * 128
  int origBlur = 2;                  // blurlevel
  int origBm = 2;                    // 0 HQ, 1 fast 3-pass, 2 fast 1-pass
  bool origShow = false;
  int opt = -1;
};

struct Format {
  int width = 0, height = 0;  // luma size of the output (aWarp4: of the edge mask clip)
  int bits = 8;               // 8..16, 32 = float
  int planes = 1;             // 1 (Y / Gray) or 3 (YUV)
  int ssw = 0, ssh = 0;       // chroma subsampling (log2)
  int bytes() const { return bits == 32 ? 4 : (bits > 8 ? 2 : 1); }
  int planeWidth(int p) const { return p ? width >> ssw : width; }
  int planeHeight(int p) const { return p ? height >> ssh : height; }
};

struct FrameView {
  const uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};  // bytes
};

struct FrameOut {
  uint8_t* data[3] = {nullptr, nullptr, nullptr};
  ptrdiff_t stride[3] = {0, 0, 0};
};

class Engine {
 public:
  // Throws std::runtime_error with the message for the host on invalid parameters or formats.
  Engine(const Params& p, const Format& f);

  // src: the clip (aWarp4: four times the size of fmt), edge: the mask clip of aWarp/aWarp4 (ignored otherwise).
  void process(const FrameView& src, const FrameView& edge, const FrameOut& out) const;
  Isa isa() const { return isa_; }

 private:
  template <typename T>
  void run(const FrameView& src, const FrameView& edge, const FrameOut& out) const;
  template <typename T>
  void runOriginal(const FrameView& src, const FrameOut& out) const;

  Params p_;
  Format fmt_;
  Isa isa_ = Isa::C;
  const Kernels* k_ = nullptr;
};

}  // namespace awarp
