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

// Blur passes by type: radius 6, radius 2, HQ blur (6 = the original's blurlevel 2 x 3).
inline int defaultBlur(int type) { return type == 0 ? 2 : (type == 1 ? 3 : 6); }

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
      static const int map[4] = {1, 4, 3, 2};
      p.type = bm != 2 ? 1 : 0;
      p.thresh = int(t * 256.0);
      if (original) p.thresh &= 255;  // beta 1 passes the threshold as a byte
      p.blur = original ? blurlevel : (p.type == 1 ? blurlevel * 3 : blurlevel);
      p.depth = int(d * blurlevel * 0.5);
      if (original && p.depth > 127) p.depth = 127;
      p.depthC = dc(p.depth);
      p.blurV = p.blur;
      p.depthV = p.depth;
      p.depthVC = p.depthC;
      p.blurC = (p.blur + 1) >> 1;
      p.blurVC = p.blurC;
      p.threshC = p.thresh;
      p.chroma = (original && cm == 0) ? 2 : (cm >= 0 && cm < 4 ? map[cm] : -1);
      p.original = original;
      p.origDepth = original ? int(d * blurlevel * 128.0) : 0;
      p.origBlur = blurlevel;
      p.origBm = bm;
      p.origShow = a.b("show", false);
      break;
    }
    case kSobel:
      p.kind = Kind::Sobel;
      p.thresh = a.i("thresh", 128);
      p.chroma = a.i("chroma", 1);
      p.threshC = a.i("threshC", p.thresh);
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
      break;
  }
  p.opt = a.i("opt", -1);
  return p;
}

}  // namespace awarp
