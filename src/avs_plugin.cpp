// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth+ interface: aWarpSharp2, aWarpSharp, aSobel, aBlur, aWarp, aWarp4 with the argument lists of aWarpSharpMT
// 2.2.0. The thread pool arguments are accepted and ignored (AviSynth+ runs the filters in parallel); opt is new.
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "avisynth.h"
#include "core/awarp.hpp"
#include "params.hpp"
#include "version.hpp"

using namespace awarp;

namespace {

class AWarpAvs : public GenericVideoFilter {
 public:
  AWarpAvs(PClip child, PClip edges, const Params& p, const Format& fmt, const VideoInfo& out_vi,
           IScriptEnvironment* env)
      : GenericVideoFilter(child), edges_(edges), fmt_(fmt), src_frames_(child->GetVideoInfo().num_frames) {
    vi = out_vi;
    try {
      env->CheckVersion(8);
      has_v8_ = true;
    } catch (const AvisynthError&) {
      has_v8_ = false;
    }
    try {
      engine_ = std::make_unique<Engine>(p, fmt_);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    n = n < 0 ? 0 : (n >= vi.num_frames ? vi.num_frames - 1 : n);
    PVideoFrame src = child->GetFrame(n < src_frames_ ? n : src_frames_ - 1, env);
    PVideoFrame edg;
    if (edges_) {
      const int ef = edges_->GetVideoInfo().num_frames;
      edg = edges_->GetFrame(n < ef ? n : ef - 1, env);
    }
    PVideoFrame& props = edg ? (vi.width == child->GetVideoInfo().width ? src : edg) : src;
    PVideoFrame dst = has_v8_ ? env->NewVideoFrameP(vi, &props) : env->NewVideoFrame(vi);
    static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    FrameOut out;
    for (int i = 0; i < fmt_.planes; i++) {
      out.data[i] = dst->GetWritePtr(planes[i]);
      out.stride[i] = dst->GetPitch(planes[i]);
    }
    if (vi.IsYUVA()) copyAlpha(src, dst, env);
    try {
      engine_->process(view(src), edg ? view(edg) : FrameView(), out);
    } catch (const std::exception& e) {
      env->ThrowError("%s", e.what());
    }
    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    (void)frame_range;
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

 private:
  FrameView view(const PVideoFrame& f) const {
    FrameView v;
    static const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int i = 0; i < fmt_.planes; i++) {
      v.data[i] = f->GetReadPtr(planes[i]);
      v.stride[i] = f->GetPitch(planes[i]);
    }
    return v;
  }

  // aWarp4: the alpha plane of the large clip, every fourth sample.
  void copyAlpha(const PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env) const {
    const int rows = dst->GetHeight(PLANAR_A), row_size = dst->GetRowSize(PLANAR_A);
    if (src->GetRowSize(PLANAR_A) == row_size) {
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetReadPtr(PLANAR_A),
                  src->GetPitch(PLANAR_A), row_size, rows);
      return;
    }
    const int bytes = fmt_.bytes();
    for (int y = 0; y < rows; y++) {
      const uint8_t* s = src->GetReadPtr(PLANAR_A) + ptrdiff_t(y) * 4 * src->GetPitch(PLANAR_A);
      uint8_t* d = dst->GetWritePtr(PLANAR_A) + ptrdiff_t(y) * dst->GetPitch(PLANAR_A);
      for (int x = 0; x < row_size / bytes; x++) std::memcpy(d + x * bytes, s + x * 4 * bytes, size_t(bytes));
    }
  }

  PClip edges_;
  Format fmt_;
  int src_frames_;
  bool has_v8_ = false;
  std::unique_ptr<Engine> engine_;
};

Format formatOf(const VideoInfo& vi, const char* name, IScriptEnvironment* env) {
  Format f;
  f.width = vi.width;
  f.height = vi.height;
  f.bits = vi.BitsPerComponent();
  if (vi.IsY()) {
    f.planes = 1;
  } else if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    f.planes = 3;
    f.ssw = vi.GetPlaneWidthSubsampling(PLANAR_U);
    f.ssh = vi.GetPlaneHeightSubsampling(PLANAR_U);
  } else {
    env->ThrowError("%s: Planar YUV or Y input is required", name);
  }
  return f;
}

// Signatures of aWarpSharpMT 2.2.0 plus opt; the thread pool arguments are parsed and ignored.
#define AWARP_MT "[threads]i[logicalCores]b[MaxPhysCore]b[SetAffinity]b[sleep]b[prefetch]i[ThreadLevel]i[opt]i"
const char* const kSignatures[] = {
    "c[thresh]i[blur]i[type]i[depth]i[chroma]i[depthC]i[cplace]s[blurV]i[depthV]i[depthVC]i[blurC]i[blurVC]i"
    "[threshC]i" AWARP_MT,
    "c[depth]f[blurlevel]i[thresh]f[cm]i[bm]i[show]b[original]b" AWARP_MT,
    "c[thresh]i[chroma]i[threshC]i" AWARP_MT,
    "c[blur]i[type]i[chroma]i[blurV]i[blurC]i[blurVC]i" AWARP_MT,
    "cc[depth]i[chroma]i[depthC]i[cplace]s[depthV]i[depthVC]i" AWARP_MT,
    "cc[depth]i[chroma]i[depthC]i[cplace]s[depthV]i[depthVC]i" AWARP_MT,
};
#undef AWARP_MT

// Named access to the positional AVSValue array, indexed by the signature.
class AvsArgs : public Args {
 public:
  AvsArgs(const AVSValue& a, const char* sig) : a_(a), sig_(sig) {}
  bool has(const char* name) const override {
    const int k = index(name);
    return k >= 0 && a_[k].Defined();
  }
  int i(const char* name, int def) const override { return has(name) ? a_[index(name)].AsInt() : def; }
  double f(const char* name, double def) const override { return has(name) ? a_[index(name)].AsFloat() : def; }
  bool b(const char* name, bool def) const override { return has(name) ? a_[index(name)].AsBool() : def; }
  std::string s(const char* name) const override { return has(name) ? a_[index(name)].AsString("") : ""; }

 private:
  int index(const char* name) const {
    int k = 0;
    for (const char* c = sig_; *c; k++) {
      if (*c == '[') {
        const char* e = std::strchr(c, ']');
        if (size_t(e - c - 1) == std::strlen(name) && std::strncmp(c + 1, name, size_t(e - c - 1)) == 0) return k;
        c = e + 2;
      } else {
        c++;
      }
    }
    return -1;
  }
  const AVSValue& a_;
  const char* sig_;
};

bool isYuvOrY(const VideoInfo& v) { return v.IsY() || (v.IsPlanar() && (v.IsYUV() || v.IsYUVA())); }

AVSValue __cdecl Create(AVSValue a, void* user_data, IScriptEnvironment* env) {
  const int fn = int(reinterpret_cast<intptr_t>(user_data));
  const char* name = fnName(fn);
  PClip child = a[0].AsClip();
  const VideoInfo& vi = child->GetVideoInfo();
  Format fmt = formatOf(vi, name, env);
  VideoInfo out_vi = vi;
  PClip edges;
  if (fn == kWarp || fn == kWarp4) {
    edges = a[1].AsClip();
    const VideoInfo& ve = edges->GetVideoInfo();
    if (!isYuvOrY(ve)) env->ThrowError("%s: Planar YUV or Y input is required", name);
    if (fn == kWarp && (vi.width != ve.width || vi.height != ve.height))
      env->ThrowError("aWarp: both sources must have the same width and height");
    if (fn == kWarp4 && (vi.width != ve.width * 4 || vi.height != ve.height * 4))
      env->ThrowError("aWarp4: first source must be exactly 4 times width and height of second source");
    if (!vi.IsSameColorspace(ve)) env->ThrowError("%s: both sources must have the same colorspace", name);
    if (fn == kWarp4) {
      out_vi = ve;
      fmt = formatOf(ve, name, env);
    }
  }
  Params p;
  try {
    p = buildParams(fn, AvsArgs(a, kSignatures[fn]), vi.Is444() || vi.IsY());
  } catch (const std::exception& e) {
    env->ThrowError("%s", e.what());
  }
  return new AWarpAvs(child, edges, p, fmt, out_vi, env);
}

}  // namespace

#ifdef _WIN32
#define AWARP_AVS_EXPORT extern "C" __declspec(dllexport)
#else
#define AWARP_AVS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

const AVS_Linkage* AVS_linkage = nullptr;

AWARP_AVS_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  for (int fn = kWarpSharp2; fn <= kWarp4; fn++)
    env->AddFunction(fnName(fn), kSignatures[fn], Create, reinterpret_cast<void*>(intptr_t(fn)));
  return "aWarpSharpMT " AWARP_VERSION;
}
