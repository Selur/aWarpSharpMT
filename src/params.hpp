// SPDX-License-Identifier: GPL-2.0-or-later
// Arguments of the six filters, shared by both hosts: defaults and the mapping of aWarpSharp to the aWarpSharp2
// parameters as in aWarpSharpMT 2.2.0.
#pragma once

#include <stdexcept>
#include <string>

#include "core/awarp.hpp"

namespace awarp {

enum Fn { kWarpSharp2 = 0, kWarpSharp = 1, kSobel = 2, kBlur = 3, kWarp = 4, kWarp4 = 5 };

inline const char* fnName(int fn) {
  static const char* const names[] = {"aWarpSharp2", "aWarpSharp", "aSobel", "aBlur", "aWarp", "aWarp4"};
  return names[fn];
}

// Host argument access by parameter name.
struct Args {
  virtual ~Args() = default;
  virtual bool has(const char* name) const = 0;
  virtual int i(const char* name, int def) const = 0;
  virtual double f(const char* name, double def) const = 0;
  virtual bool b(const char* name, bool def) const = 0;
  virtual std::string s(const char* name) const = 0;
};

// cplace "MPEG2" in any case; everything else is MPEG1.
inline bool isMpeg2(const std::string& s) {
  if (s.size() != 5) return false;
  static const char ref[] = "mpeg2";
  for (int k = 0; k < 5; k++) {
    char c = s[k];
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    if (c != ref[k]) return false;
  }
  return true;
}

// Blur passes by type: radius 6, radius 2, and the original's HQ, fast 3-pass and fast 1-pass blur at blurlevel 2.
inline int defaultBlur(int type) {
  static const int passes[5] = {2, 3, 6, 6, 2};
  return type >= 0 && type <= 4 ? passes[type] : 2;
}

// full_chroma: 4:4:4 or Y (depthC defaults to depth there, else to depth / 2).
inline Params buildParams(int fn, const Args& a, bool full_chroma) {
  Params p;
  auto dc = [&](int depth) { return full_chroma ? depth : depth >> 1; };
  switch (fn) {
    case kWarpSharp2:
      p.kind = Kind::WarpSharp2;
      p.thresh = a.i("thresh", 128);
      p.type = a.i("type", 0);
      p.blur = a.i("blur", defaultBlur(p.type));
      p.depth = a.i("depth", 16);
      p.chroma = a.i("chroma", 4);
      p.depthC = a.i("depthC", dc(p.depth));
      p.mpeg2 = isMpeg2(a.s("cplace"));
      p.blurV = a.i("blurV", p.blur);
      p.depthV = a.i("depthV", p.depth);
      p.depthVC = a.i("depthVC", p.depthC);
      p.blurC = a.i("blurC", (p.blur + 1) >> 1);
      p.blurVC = a.i("blurVC", p.blurC);
      p.threshC = a.i("threshC", p.thresh);
      p.mask = a.i("mask", 0);
      p.wmode = a.i("wmode", 0);
      break;
    case kWarpSharp: {
      p.kind = Kind::WarpSharp;
      const bool original = a.b("original", true);
      const double d = a.f("depth", 16.0), t = a.f("thresh", 0.5);
      const int blurlevel = a.i("blurlevel", 2), cm = a.i("cm", 1), bm = a.i("bm", 2);
      if (original) {
        const char* e = nullptr;
        if (bm < 0 || bm > 2) e = "bm must be 0..2";
        else if (!(d >= 0.0 && d <= 64.0)) e = "depth must be 0..64";
        else if (!(t >= 0.0 && t <= 1.0)) e = "thresh must be 0..1";
        else if (blurlevel < 1 || blurlevel > 4) e = "blurlevel must be 1..4";
        else if (cm < 0 || cm > 2) e = "cm must be 0..2";
        if (e) throw std::runtime_error(std::string("aWarpSharp(original=true): ") + e);
      }
      p.thresh = int(t * 256.0);
      p.threshC = p.thresh;
      if (original) {
        // aWarpSharp2 with the original's mask, blur and warp; the warp takes origDepth (finer than depth) directly.
        static const int types[3] = {2, 3, 4}, chromas[3] = {2, 4, 3};
        const int k = bm == 2 ? 1 : 3;  // bm 0 and 1 run three passes per blur level
        p.thresh = p.threshC = p.thresh & 255;  // beta 1 passes the threshold as a byte
        p.mask = 1;
        p.wmode = 1;
        p.type = types[bm];
        p.blur = p.blurV = blurlevel * k;
        p.blurC = p.blurVC = (blurlevel / 2) * k;
        p.chroma = chromas[cm];
        p.original = true;
        p.origDepth = int(d * blurlevel * 128.0);
        p.depth = p.depthV = p.depthC = p.depthVC = 0;  // unused: origDepth
        p.origShow = a.b("show", false);
        break;
      }
      static const int map[4] = {1, 4, 3, 2};
      p.type = bm != 2 ? 1 : 0;
      p.blur = p.type == 1 ? blurlevel * 3 : blurlevel;
      p.depth = int(d * blurlevel * 0.5);
      p.depthC = dc(p.depth);
      p.blurV = p.blur;
      p.depthV = p.depth;
      p.depthVC = p.depthC;
      p.blurC = (p.blur + 1) >> 1;
      p.blurVC = p.blurC;
      p.chroma = cm >= 0 && cm < 4 ? map[cm] : -1;
      break;
    }
    case kSobel:
      p.kind = Kind::Sobel;
      p.thresh = a.i("thresh", 128);
      p.chroma = a.i("chroma", 1);
      p.threshC = a.i("threshC", p.thresh);
      p.mask = a.i("mask", 0);
      break;
    case kBlur:
      p.kind = Kind::Blur;
      p.type = a.i("type", 1);
      p.blur = a.i("blur", defaultBlur(p.type));
      p.chroma = a.i("chroma", 1);
      p.blurV = a.i("blurV", p.blur);
      p.blurC = a.i("blurC", (p.blur + 1) >> 1);
      p.blurVC = a.i("blurVC", p.blurC);
      break;
    default:
      p.kind = fn == kWarp ? Kind::Warp : Kind::Warp4;
      p.depth = a.i("depth", 3);
      p.chroma = a.i("chroma", 4);
      p.depthC = a.i("depthC", dc(p.depth));
      p.mpeg2 = isMpeg2(a.s("cplace"));
      p.depthV = a.i("depthV", p.depth);
      p.depthVC = a.i("depthVC", p.depthC);
      if (fn == kWarp) p.wmode = a.i("wmode", 0);
      break;
  }
  p.opt = a.i("opt", -1);
  return p;
}

}  // namespace awarp
