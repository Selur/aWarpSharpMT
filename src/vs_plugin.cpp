// SPDX-License-Identifier: GPL-2.0-or-later
// VapourSynth (API 4) interface: awarpsharpmt.aWarpSharp2, aWarpSharp, aSobel, aBlur, aWarp, aWarp4 with the
// parameters of the AviSynth+ functions (without the thread pool arguments). Gray and YUV, 8-16 bit integer and
// 32 bit float.
#include <memory>
#include <stdexcept>
#include <string>

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "core/awarp.hpp"
#include "params.hpp"
#include "version.hpp"

using namespace awarp;

namespace {

struct FilterData {
  VSNode* node = nullptr;
  VSNode* edges = nullptr;
  int fn = 0;
  int src_frames = 0, edge_frames = 0;
  Format fmt;
  VSVideoInfo vi;
  std::unique_ptr<Engine> engine;
};

class VsArgs : public Args {
 public:
  VsArgs(const VSMap* in, const VSAPI* vsapi) : in_(in), vsapi_(vsapi) {}
  bool has(const char* name) const override { return vsapi_->mapNumElements(in_, name) > 0; }
  int i(const char* name, int def) const override {
    return has(name) ? int(vsapi_->mapGetInt(in_, name, 0, nullptr)) : def;
  }
  double f(const char* name, double def) const override {
    return has(name) ? vsapi_->mapGetFloat(in_, name, 0, nullptr) : def;
  }
  bool b(const char* name, bool def) const override {
    return has(name) ? vsapi_->mapGetInt(in_, name, 0, nullptr) != 0 : def;
  }
  std::string s(const char* name) const override {
    return has(name) ? std::string(vsapi_->mapGetData(in_, name, 0, nullptr)) : std::string();
  }

 private:
  const VSMap* in_;
  const VSAPI* vsapi_;
};

FrameView view(const VSFrame* f, int planes, const VSAPI* vsapi) {
  FrameView v;
  for (int i = 0; i < planes; i++) {
    v.data[i] = vsapi->getReadPtr(f, i);
    v.stride[i] = vsapi->getStride(f, i);
  }
  return v;
}

const VSFrame* VS_CC getFrame(int n, int reason, void* instance, void**, VSFrameContext* ctx, VSCore* core,
                              const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  const int sn = n < d->src_frames ? n : d->src_frames - 1;
  const int en = d->edges ? (n < d->edge_frames ? n : d->edge_frames - 1) : 0;
  if (reason == arInitial) {
    vsapi->requestFrameFilter(sn, d->node, ctx);
    if (d->edges) vsapi->requestFrameFilter(en, d->edges, ctx);
    return nullptr;
  }
  if (reason != arAllFramesReady) return nullptr;

  const VSFrame* src = vsapi->getFrameFilter(sn, d->node, ctx);
  const VSFrame* edg = d->edges ? vsapi->getFrameFilter(en, d->edges, ctx) : nullptr;
  const VSFrame* props = d->fn == kWarp4 ? edg : src;
  VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, props, core);
  FrameOut out;
  for (int i = 0; i < d->fmt.planes; i++) {
    out.data[i] = vsapi->getWritePtr(dst, i);
    out.stride[i] = vsapi->getStride(dst, i);
  }
  const VSFrame* result = dst;
  try {
    d->engine->process(view(src, d->fmt.planes, vsapi), edg ? view(edg, d->fmt.planes, vsapi) : FrameView(), out);
  } catch (const std::exception& e) {
    vsapi->setFilterError(e.what(), ctx);
    vsapi->freeFrame(dst);
    result = nullptr;
  }
  vsapi->freeFrame(src);
  vsapi->freeFrame(edg);
  return result;
}

void VS_CC freeFilter(void* instance, VSCore*, const VSAPI* vsapi) {
  auto* d = static_cast<FilterData*>(instance);
  vsapi->freeNode(d->node);
  vsapi->freeNode(d->edges);
  delete d;
}

bool formatOf(const VSVideoInfo* vi, Format* f) {
  const VSVideoFormat& vf = vi->format;
  f->width = vi->width;
  f->height = vi->height;
  const bool int_ok = vf.sampleType == stInteger && vf.bitsPerSample >= 8 && vf.bitsPerSample <= 16;
  const bool float_ok = vf.sampleType == stFloat && vf.bitsPerSample == 32;
  if (!int_ok && !float_ok) return false;
  f->bits = vf.bitsPerSample;
  if (vf.colorFamily == cfYUV) {
    f->planes = 3;
    f->ssw = vf.subSamplingW;
    f->ssh = vf.subSamplingH;
  } else if (vf.colorFamily == cfGray) {
    f->planes = 1;
  } else {
    return false;
  }
  return true;
}

void VS_CC create(const VSMap* in, VSMap* out, void* user_data, VSCore* core, const VSAPI* vsapi) {
  auto d = std::make_unique<FilterData>();
  d->fn = int(reinterpret_cast<intptr_t>(user_data));
  const std::string name = fnName(d->fn);
  d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  if (d->fn == kWarp || d->fn == kWarp4) d->edges = vsapi->mapGetNode(in, "edges", 0, nullptr);
  auto fail = [&](const std::string& msg) {
    vsapi->mapSetError(out, (msg.rfind(name, 0) == 0 ? msg : name + ": " + msg).c_str());
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->edges);
  };

  const VSVideoInfo* vi = vsapi->getVideoInfo(d->node);
  if (!vsh::isConstantVideoFormat(vi)) return fail("only constant format input is supported");
  if (!formatOf(vi, &d->fmt)) return fail("use Gray or YUV with 8-16 bit integer or 32 bit float samples");
  d->vi = *vi;
  d->src_frames = vi->numFrames;
  if (d->edges) {
    const VSVideoInfo* ve = vsapi->getVideoInfo(d->edges);
    if (!vsh::isConstantVideoFormat(ve) || !vsh::isSameVideoFormat(&ve->format, &vi->format))
      return fail("both sources must have the same constant format");
    if (d->fn == kWarp && (ve->width != vi->width || ve->height != vi->height))
      return fail("both sources must have the same width and height");
    if (d->fn == kWarp4) {
      if (vi->width != ve->width * 4 || vi->height != ve->height * 4)
        return fail("first source must be exactly 4 times width and height of second source");
      d->vi = *ve;
      formatOf(ve, &d->fmt);
    }
    d->edge_frames = ve->numFrames;
  }

  const bool full_chroma = vi->format.colorFamily == cfGray || (vi->format.subSamplingW == 0 && vi->format.subSamplingH == 0);
  try {
    const Params p = buildParams(d->fn, VsArgs(in, vsapi), full_chroma);
    d->engine = std::make_unique<Engine>(p, d->fmt);
  } catch (const std::exception& ex) {
    return fail(ex.what());
  }

  VSFilterDependency deps[2] = {{d->node, d->fn == kWarp4 ? rpGeneral : rpStrictSpatial}, {d->edges, rpStrictSpatial}};
  if (d->fn == kWarp4) deps[1].requestPattern = rpGeneral;
  const int ndeps = d->edges ? 2 : 1;
  const VSVideoInfo outvi = d->vi;
  vsapi->createVideoFilter(out, name.c_str(), &outvi, getFrame, freeFilter, fmParallel, deps, ndeps, d.release(), core);
}

#define AWARP_COMMON "opt:int:opt;"

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin("io.github.selur.awarpsharpmt", "awarpsharpmt", "aWarpSharpMT " AWARP_VERSION,
                       VS_MAKE_VERSION(AWARP_VERSION_MAJOR, AWARP_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0, plugin);
  const char* const args[] = {
      "clip:vnode;thresh:int:opt;blur:int:opt;type:int:opt;depth:int:opt;chroma:int:opt;depthC:int:opt;"
      "cplace:data:opt;blurV:int:opt;depthV:int:opt;depthVC:int:opt;blurC:int:opt;blurVC:int:opt;threshC:int:opt;"
      "mask:int:opt;wmode:int:opt;" AWARP_COMMON,
      "clip:vnode;depth:float:opt;blurlevel:int:opt;thresh:float:opt;cm:int:opt;bm:int:opt;show:int:opt;"
      "original:int:opt;" AWARP_COMMON,
      "clip:vnode;thresh:int:opt;chroma:int:opt;threshC:int:opt;mask:int:opt;" AWARP_COMMON,
      "clip:vnode;blur:int:opt;type:int:opt;chroma:int:opt;blurV:int:opt;blurC:int:opt;blurVC:int:opt;" AWARP_COMMON,
      "clip:vnode;edges:vnode;depth:int:opt;chroma:int:opt;depthC:int:opt;cplace:data:opt;depthV:int:opt;"
      "depthVC:int:opt;wmode:int:opt;" AWARP_COMMON,
      "clip:vnode;edges:vnode;depth:int:opt;chroma:int:opt;depthC:int:opt;cplace:data:opt;depthV:int:opt;"
      "depthVC:int:opt;" AWARP_COMMON,
  };
  for (int fn = kWarpSharp2; fn <= kWarp4; fn++)
    vspapi->registerFunction(fnName(fn), args[fn], "clip:vnode;", create, reinterpret_cast<void*>(intptr_t(fn)),
                             plugin);
}
