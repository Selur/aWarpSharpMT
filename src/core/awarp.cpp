// SPDX-License-Identifier: GPL-2.0-or-later
// aWarpSharpMT engine: sobel edge mask, mask blur, chroma guide and warp per plane, in the order of aWarpSharpMT 2.2.0
// (JPSDR). See README.md for the formulas and the differences to the original.
#include "awarp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "scalar.hpp"

namespace awarp {

namespace {

const char* kindName(Kind k) {
  switch (k) {
    case Kind::WarpSharp2: return "aWarpSharp2";
    case Kind::WarpSharp: return "aWarpSharp";
    case Kind::Sobel: return "aSobel";
    case Kind::Blur: return "aBlur";
    case Kind::Warp: return "aWarp";
    case Kind::Warp4: return "aWarp4";
  }
  return "aWarpSharp";
}

// A plane of samples of type T with a stride in samples.
template <typename T>
struct Plane {
  T* p = nullptr;
  ptrdiff_t stride = 0;
  int w = 0, h = 0;
  T* row(int y) const { return p + y * stride; }
};

template <typename T>
struct CPlane {
  const T* p = nullptr;
  ptrdiff_t stride = 0;
  int w = 0, h = 0;
  CPlane() = default;
  CPlane(const Plane<T>& o) : p(o.p), stride(o.stride), w(o.w), h(o.h) {}
  const T* row(int y) const { return p + y * stride; }
};

// Scratch memory kept per thread between frames, so the pages are not faulted in again for every frame.
struct Block {
  std::unique_ptr<uint8_t[]> mem;
  size_t size = 0;
};

std::vector<Block>& pool() {
  static thread_local std::vector<Block> p;
  return p;
}

// Scratch plane owned by one call (no initialisation; every sample is written before it is read).
template <typename T>
struct Buffer {
  Block block;
  Plane<T> plane;
  Buffer(int w, int h) {
    const ptrdiff_t stride = (ptrdiff_t(w) + 63) & ~ptrdiff_t(63);
    const size_t need = (size_t(stride) * size_t(h > 0 ? h : 1) + 64) * sizeof(T);
    auto& p = pool();
    size_t best = p.size();
    for (size_t i = 0; i < p.size(); i++)
      if (p[i].size >= need && (best == p.size() || p[i].size < p[best].size)) best = i;
    if (best < p.size()) {
      block = std::move(p[best]);
      p.erase(p.begin() + ptrdiff_t(best));
    } else {
      block.mem.reset(new uint8_t[need]);
      block.size = need;
    }
    plane.p = reinterpret_cast<T*>(block.mem.get());
    plane.stride = stride;
    plane.w = w;
    plane.h = h;
  }
  ~Buffer() {
    auto& p = pool();
    if (p.size() < 8) p.push_back(std::move(block));
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

template <typename T>
void copyPlane(const CPlane<T>& s, const Plane<T>& d) {
  for (int y = 0; y < d.h; y++) std::memcpy(d.row(y), s.row(y), size_t(d.w) * sizeof(T));
}

template <typename T>
void fillPlane(const Plane<T>& d, T v) {
  for (int y = 0; y < d.h; y++) std::fill(d.row(y), d.row(y) + d.w, v);
}

// Integer samples: value on the 8-bit scale shifted to the bit depth; float: divided by 255.
template <typename T>
Num<T> scale8(int v, int bits);
template <>
int scale8<uint8_t>(int v, int) { return v; }
template <>
int scale8<uint16_t>(int v, int bits) { return v << (bits - 8); }
template <>
float scale8<float>(int v, int) { return float(v) / 255.0f; }

template <typename T>
Num<T> pixelMax(int bits) {
  return Num<T>((1 << bits) - 1);
}
template <>
float pixelMax<float>(int) { return 1.0f; }

template <typename T>
T neutralChroma(int bits) {
  return T(128 << (bits - 8));
}
template <>
float neutralChroma<float>(int) { return 0.0f; }


// One pass of MarcFD's HQ blur, in place and row by row: each row reads the rows above as already blurred, taps 1 4 10 1
// on rows y-2, y-1, y, y+2 (the centre row stands in for y+1), then 1 4 6 4 1 across; rows 0, 1, h-2, h-1 stay.
template <typename T>
void blurHQPass(const Plane<T>& m) {
  using N = Num<T>;
  const int w = m.w, h = m.h;
  std::vector<N> buf(size_t(w) + 4);
  N* row = buf.data() + 2;
  for (int y = 2; y < h - 2; y++) {
    const T *a = m.row(y - 2), *b = m.row(y - 1), *c = m.row(y), *e = m.row(y + 2);
    for (int x = 0; x < w; x++) row[x] = load(a, x) + 4 * load(b, x) + 10 * load(c, x) + load(e, x);
    row[-2] = row[-1] = row[0];
    row[w] = row[w + 1] = row[w - 1];
    T* o = m.row(y);
    for (int x = 0; x < w; x++) {
      const N v = row[x - 2] + 4 * row[x - 1] + 6 * row[x] + 4 * row[x + 1] + row[x + 2];
      if constexpr (std::is_same<T, float>::value)
        o[x] = v / 256.0f;
      else
        o[x] = T((v + 64) / 256);
    }
  }
}

template <typename T>
struct Ops {
  const KernelSet<T>& k;
  int bits;

  // Edge mask. Integer samples keep two quirks of the SSE2/AVX code of the original, which works in 16-byte blocks:
  // in rows 0 and 1 the first block takes the sample itself for the left neighbour in the row above and in the row,
  // in the last two rows the last block takes the sample itself for the right neighbour in the row below and in the
  // row. Columns 0 and w-1 repeat columns 1 and w-2, rows -1 and h repeat rows 0 and h-1.
  void sobel(const CPlane<T>& s, const Plane<T>& d, int thresh8) const {
    const int w = s.w, h = s.h;
    const Num<T> thr = scale8<T>(thresh8, bits), pmax = pixelMax<T>(bits);
    const bool quirks = !std::is_same<T, float>::value;
    const int block = 16 / int(sizeof(T));
    const int last_block = block * ((w + block - 1) / block - 1);
    for (int y = 0; y < h; y++) {
      const T* u = s.row(y > 0 ? y - 1 : 0);
      const T* m = s.row(y);
      const T* dn = s.row(y < h - 1 ? y + 1 : h - 1);
      T* o = d.row(y);
      if (w < 3) {
        for (int x = 0; x < w; x++) o[x] = m[x];
        continue;
      }
      k.sobel(u, m, dn, o, 1, w - 1, thr, pmax);
      if (quirks && y >= h - 2) {
        for (int x = std::max(last_block, 1); x < w - 1; x++)
          o[x] = T(sobelPx(load(u, x - 1), load(u, x), load(u, x + 1), load(m, x - 1), load(m, x), load(dn, x - 1),
                           load(dn, x), load(dn, x), thr, pmax));
      } else if (quirks && y < 2) {
        for (int x = 1; x < std::min(block, w - 1); x++)
          o[x] = T(sobelPx(load(u, x), load(u, x), load(u, x + 1), load(m, x), load(m, x + 1), load(dn, x - 1),
                           load(dn, x), load(dn, x + 1), thr, pmax));
      }
      o[0] = o[1];
      o[w - 1] = o[w - 2];
    }
  }

  // Horizontal blur step of one row.
  void blurRowH(const T* s, T* o, int w, int type) const {
    if (type == 0) {
      if (w >= 12) {
        k.blurH6(s, o, w);
      } else {
        for (int x = 0; x < w; x++) o[x] = T(blurH6Edge(s, x, w));
      }
    } else if (w >= 5) {
      for (int x : {0, 1}) o[x] = T(blurH2Clamped(s, x, w));
      k.blurH2(s, o, 2, w - 2);
      for (int x : {w - 2, w - 1}) o[x] = T(blurH2Clamped(s, x, w));
    } else {
      for (int x = 0; x < w; x++) o[x] = T(blurH2Clamped(s, x, w));
    }
  }

  // Vertical blur step of row y; at(yy) gives row yy (0 <= yy < h) of the horizontal step.
  template <typename At>
  void blurRowV(const At& at, T* o, int y, int h, int w, int type) const {
    const T* rows[13];
    auto r = [&](int yy) { return at(std::max(0, std::min(yy, h - 1))); };
    if (type == 0) {
      // Rows 0..5 take their six neighbours from below, the last six rows (from row 6 on) from above.
      for (int k6 = 1; k6 <= 6; k6++) {
        int above, below;
        if (y < 6) {
          above = below = y + k6;
        } else if (y >= h - 6) {
          above = below = y - k6;
        } else {
          above = y - k6;
          below = y + k6;
        }
        rows[6 - k6] = r(above);
        rows[6 + k6] = r(below);
      }
      rows[6] = at(y);
      k.blurV6(rows, o, w);
    } else {
      rows[0] = r(y > 1 ? y - 2 : (y > 0 ? y - 1 : y));
      rows[1] = r(y > 0 ? y - 1 : y);
      rows[2] = at(y);
      rows[3] = r(y < h - 1 ? y + 1 : y);
      rows[4] = r(y < h - 2 ? y + 2 : (y < h - 1 ? y + 1 : y));
      k.blurV2(rows, o, w);
    }
  }

  // n blur passes in place, row by row: pass i+1 trails pass i by the vertical radius and keeps its horizontal step
  // in a ring of rows, so the rows stay in the cache between the passes.
  void blurPasses(const Plane<T>& m, int type, bool hz, bool vt, int n) const {
    const int w = m.w, h = m.h;
    const int lag = vt ? (type == 0 ? 6 : 2) : 0;
    const int ringRows = vt ? (type == 0 ? 16 : 8) : 1;
    Buffer<T> ring(w, n * ringRows);
    auto ringRow = [&](int p, int y) { return ring.plane.row(p * ringRows + (y & (ringRows - 1))); };
    const size_t bytes = size_t(w) * sizeof(T);
    for (int s = 0; s < h + lag * n; s++) {
      for (int p = 0; p < n; p++) {
        const int hy = s - p * lag, vy = hy - lag;
        if (hy >= 0 && hy < h) {
          if (hz)
            blurRowH(m.row(hy), ringRow(p, hy), w, type);
          else
            std::memcpy(ringRow(p, hy), m.row(hy), bytes);
        }
        if (vy >= 0 && vy < h) {
          if (vt)
            blurRowV([&](int yy) { return static_cast<const T*>(ringRow(p, yy)); }, m.row(vy), vy, h, w, type);
          else
            std::memcpy(m.row(vy), ringRow(p, vy), bytes);
        }
      }
    }
  }

  // One pass in two sweeps over the plane: horizontal step mask -> t, vertical step back into mask.
  void blurSweeps(const Plane<T>& m, const Plane<T>& t, int type, bool hz, bool vt) const {
    const int w = m.w, h = m.h;
    const size_t bytes = size_t(w) * sizeof(T);
    for (int y = 0; y < h; y++) {
      if (hz)
        blurRowH(m.row(y), t.row(y), w, type);
      else
        std::memcpy(t.row(y), m.row(y), bytes);
    }
    for (int y = 0; y < h; y++) {
      if (vt)
        blurRowV([&](int yy) { return static_cast<const T*>(t.row(yy)); }, m.row(y), y, h, w, type);
      else
        std::memcpy(m.row(y), t.row(y), bytes);
    }
  }

  // Passes per call of blurPasses, from the plane size (measured on a 16-core Zen 4): below 1 MiB the plane stays in
  // the cache (0: two sweeps per pass), below 3 MiB one pass per call, above that ring and trailing rows in 512 KiB.
  static int passGroup(int w, int h, int type) {
    const size_t plane = size_t(w) * size_t(h) * sizeof(T);
    if (plane < (size_t(1) << 20)) return 0;
    if (plane < (size_t(3) << 20)) return 1;
    const size_t perPass = size_t(type == 0 ? 16 + 6 : 8 + 2) * size_t(w) * sizeof(T);
    return int(std::max<size_t>(1, (size_t(512) << 10) / perPass));
  }

  // Radius 6 edge columns: the first six take all neighbours from the right, the last six from the left.
  static Num<T> blurH6Edge(const T* s, int x, int w) {
    Num<T> p[6], q[6];
    for (int k6 = 1; k6 <= 6; k6++) {
      int i = x >= w - 6 ? x - k6 : (x < 6 ? x + k6 : -1);
      int l = i, r = i;
      if (i < 0) {
        l = x - k6;
        r = x + k6;
      }
      l = std::max(0, std::min(l, w - 1));
      r = std::max(0, std::min(r, w - 1));
      p[k6 - 1] = load(s, l);
      q[k6 - 1] = load(s, r);
    }
    return blur6(load(s, x), p, q);
  }

  static Num<T> blurH2Clamped(const T* s, int x, int w) {
    auto c = [&](int i) { return load(s, std::max(0, std::min(i, w - 1))); };
    return blur2(c(x), c(x - 1), c(x + 1), c(x - 2), c(x + 2));
  }

  // blur passes in both directions, then the passes of the larger of blurH/blurV in its direction only.
  // t: scratch plane of the size of m.
  void blur(const Plane<T>& m, const Plane<T>& t, int type, int blurH, int blurV) const {
    if (type == 2) {
      // blurH == blurV (checked): each HQ pass blurs both directions
      for (int i = 0; i < blurH; i++) blurHQPass(m);
      return;
    }
    const int both = std::min(blurH, blurV), rest = std::max(blurH, blurV) - both;
    const bool hz = blurH > both, vt = blurV > both;
    const int group = passGroup(m.w, m.h, type);
    if (group == 0) {
      for (int i = 0; i < both; i++) blurSweeps(m, t, type, true, true);
      for (int i = 0; i < rest; i++) blurSweeps(m, t, type, hz, vt);
      return;
    }
    for (int i = 0; i < both; i += group) blurPasses(m, type, true, true, std::min(group, both - i));
    for (int i = 0; i < rest; i += group) blurPasses(m, type, hz, vt, std::min(group, rest - i));
  }

  // Luma mask to chroma size. 4:2:0 and 4:2:2 as aWarpSharpMT (MPEG-1: rounded pair averages, horizontal first;
  // MPEG-2: 1-2-1 horizontal filter, vertical sum for 4:2:0); other subsamplings average pairs the same way.
  void guide(const CPlane<T>& y, const Plane<T>& c, int ssw, int ssh, bool mpeg2) const {
    if (mpeg2 && ssw == 1 && ssh <= 1) {
      for (int yy = 0; yy < c.h; yy++) {
        const T* a = y.row(yy << ssh);
        const T* b = y.row((yy << ssh) + ssh);
        T* o = c.row(yy);
        auto col = [&](int x) { return ssh ? load(a, x) + load(b, x) : load(a, x); };
        auto c2 = col(0);
        for (int x = 0; x < c.w; x++) {
          const auto c0 = c2, c1 = col(2 * x);
          c2 = col(2 * x + 1);
          o[x] = T(mpeg2Round(c0 + c1 + c1 + c2, ssh ? 3 : 2));
        }
      }
      return;
    }
    for (int yy = 0; yy < c.h; yy++) {
      T* o = c.row(yy);
      const T* a = y.row(yy << ssh);
      if (ssw == 1 && ssh == 1) {
        const T* b = y.row((yy << 1) + 1);
        for (int x = 0; x < c.w; x++)
          o[x] = T(avg(avg(load(a, 2 * x), load(a, 2 * x + 1)), avg(load(b, 2 * x), load(b, 2 * x + 1))));
      } else if (ssw == 1 && ssh == 0) {
        for (int x = 0; x < c.w; x++) o[x] = T(avg(load(a, 2 * x), load(a, 2 * x + 1)));
      } else {
        for (int x = 0; x < c.w; x++) o[x] = T(reduceV(y, x << ssw, yy << ssh, ssw, ssh));
      }
    }
  }
  static Num<T> mpeg2Round(Num<T> v, int shift) {
    if constexpr (std::is_same<T, float>::value)
      return v / float(1 << shift);
    else
      return (v + (1 << (shift - 1))) >> shift;
  }
  static Num<T> reduceH(const T* r, int x, int s) {
    return s ? avg(reduceH(r, x, s - 1), reduceH(r, x + (1 << (s - 1)), s - 1)) : load(r, x);
  }
  static Num<T> reduceV(const CPlane<T>& p, int x, int y, int ssw, int s) {
    return s ? avg(reduceV(p, x, y, ssw, s - 1), reduceV(p, x, y + (1 << (s - 1)), ssw, s - 1))
             : reduceH(p.row(y), x, ssw);
  }

  // Warp of a plane along a mask of its size; src is the plane itself (smagl 0) or four times its size (smagl 2).
  void warp(const CPlane<T>& src, const CPlane<T>& e, const Plane<T>& d, int depthH, int depthV, int smagl) const {
    WarpRow<T> r;
    r.src = src.p;
    r.src_stride = src.stride;
    r.w = d.w;
    r.h = d.h;
    r.pixel_max = pixelMax<T>(bits);
    if constexpr (std::is_same<T, float>::value) {
      r.depthHf = float(depthH);
      r.depthVf = float(depthV);
    } else {
      r.depthH = depthH * (1 << (21 - bits));
      r.depthV = depthV * (1 << (21 - bits));
    }
    for (int y = 0; y < d.h; y++) {
      r.y = y;
      r.eu = e.row(y > 0 ? y - 1 : 0);
      r.ec = e.row(y);
      r.ed = e.row(y < d.h - 1 ? y + 1 : y);
      r.dst = d.row(y);
      (smagl ? k.warp2 : k.warp0)(r, 0, d.w);
    }
  }

  // aWarp4 planes that are not cared for: every fourth sample of the large clip.
  static void pointSample(const CPlane<T>& s, const Plane<T>& d) {
    for (int y = 0; y < d.h; y++) {
      const T* a = s.row(y * 4);
      T* o = d.row(y);
      for (int x = 0; x < d.w; x++) o[x] = a[x * 4];
    }
  }
};

}  // namespace

Engine::Engine(const Params& p, const Format& f) : p_(p), fmt_(f) {
  const std::string name = kindName(p.kind);
  auto fail = [&](const std::string& m) { throw std::runtime_error(name + ": " + m); };
  if (!(f.bits >= 8 && f.bits <= 16) && f.bits != 32) fail("unsupported bit depth");
  if (f.planes == 1) p_.chroma = kChromaIgnore;
  if (!p.original) {
    if (p.thresh < 0 || p.thresh > 255) fail("'thresh' must be 0..255");
    if (p.threshC < 0 || p.threshC > 255) fail("'threshC' must be 0..255");
    if (p.blur < 0) fail("'blur' must be >=0");
    if (p.blurV < 0) fail("'blurV' must be >=0");
    if (p.blurC < 0) fail("'blurC' must be >=0");
    if (p.blurVC < 0) fail("'blurVC' must be >=0");
    if (p.type < 0 || p.type > 2) fail("'type' must be 0..2");
    if (p.type == 2 && (p.blurV != p.blur || p.blurVC != p.blurC))
      fail("type=2 blurs both directions in one pass: 'blurV' must equal 'blur' and 'blurVC' 'blurC'");
    if (p.depth < -128 || p.depth > 127) fail("'depth' must be -128..127");
    if (p.depthC < -128 || p.depthC > 127) fail("'depthC' must be -128..127");
    if (p.depthV < -128 || p.depthV > 127) fail("'depthV' must be -128..127");
    if (p.depthVC < -128 || p.depthVC > 127) fail("'depthVC' must be -128..127");
    if (p.chroma < 0 || p.chroma > 6) fail("'chroma' must be 0..6");
  }
  if (f.width < 1 || f.height < 1) fail("empty clip");
  if (!selectIsa(p.opt, &isa_)) fail("'opt' " + std::to_string(p.opt) + " is not available on this CPU");
  k_ = &kernelsFor(isa_);
}

void Engine::process(const FrameView& src, const FrameView& edge, const FrameOut& out) const {
  if (fmt_.bits == 32) {
    p_.original ? runOriginal<float>(src, out) : run<float>(src, edge, out);
  } else if (fmt_.bits > 8) {
    p_.original ? runOriginal<uint16_t>(src, out) : run<uint16_t>(src, edge, out);
  } else {
    p_.original ? runOriginal<uint8_t>(src, out) : run<uint8_t>(src, edge, out);
  }
}

namespace {

template <typename T>
CPlane<T> view(const FrameView& f, int i, int w, int h) {
  CPlane<T> p;
  p.p = reinterpret_cast<const T*>(f.data[i]);
  p.stride = f.stride[i] / ptrdiff_t(sizeof(T));
  p.w = w;
  p.h = h;
  return p;
}

template <typename T>
Plane<T> view(const FrameOut& f, int i, int w, int h) {
  Plane<T> p;
  p.p = reinterpret_cast<T*>(f.data[i]);
  p.stride = f.stride[i] / ptrdiff_t(sizeof(T));
  p.w = w;
  p.h = h;
  return p;
}

}  // namespace

template <typename T>
void Engine::run(const FrameView& srcf, const FrameView& edgef, const FrameOut& outf) const {
  const Params& p = p_;
  const Ops<T> ops{k_->get<T>(), fmt_.bits};
  const int np = fmt_.planes;
  const int ch = p.chroma;
  const int scale = p.kind == Kind::Warp4 ? 4 : 1;
  CPlane<T> src[3], edg[3];
  Plane<T> dst[3];
  for (int i = 0; i < np; i++) {
    const int w = fmt_.planeWidth(i), h = fmt_.planeHeight(i);
    src[i] = view<T>(srcf, i, w * scale, h * scale);
    dst[i] = view<T>(outf, i, w, h);
    if (p.kind == Kind::Warp || p.kind == Kind::Warp4) edg[i] = view<T>(edgef, i, w, h);
  }
  const T neutral = neutralChroma<T>(fmt_.bits);
  auto chromaFill = [&]() {
    for (int i = 1; i < np; i++) fillPlane(dst[i], neutral);
  };
  auto chromaCopy = [&]() {
    for (int i = 1; i < np; i++) {
      if (p.kind == Kind::Warp4)
        Ops<T>::pointSample(src[i], dst[i]);
      else
        copyPlane(src[i], dst[i]);
    }
  };

  switch (p.kind) {
    case Kind::Sobel: {
      if (ch < 5)
        ops.sobel(src[0], dst[0], p.thresh);
      else
        copyPlane(src[0], dst[0]);
      if (np == 1) break;
      if (ch == kChromaZero)
        chromaFill();
      else if (ch == kChromaIgnore || ch == kChromaCopy)
        chromaCopy();
      else
        for (int i = 1; i < np; i++) ops.sobel(src[i], dst[i], p.threshC);
      break;
    }
    case Kind::Blur: {
      Buffer<T> tmp(fmt_.width, fmt_.height);
      for (int i = 0; i < np; i++) copyPlane(src[i], dst[i]);
      auto blurPlane = [&](int i, int bh, int bv) {
        Plane<T> t = tmp.plane;
        t.w = dst[i].w;
        t.h = dst[i].h;
        ops.blur(dst[i], t, p.type, bh, bv);
      };
      if (ch < 5) blurPlane(0, p.blur, p.blurV);
      if (np == 1) break;
      if (ch == kChromaZero)
        chromaFill();
      else if (ch >= kChromaProcess)
        for (int i = 1; i < np; i++) blurPlane(i, p.blurC, p.blurVC);
      break;
    }
    case Kind::Warp:
    case Kind::Warp4: {
      const int smagl = p.kind == Kind::Warp4 ? 2 : 0;
      const bool w4 = p.kind == Kind::Warp4;
      if (ch < 5 && (w4 || p.depth != 0 || p.depthV != 0))
        ops.warp(src[0], edg[0], dst[0], p.depth, p.depthV, smagl);
      else if (w4)
        Ops<T>::pointSample(src[0], dst[0]);
      else
        copyPlane(src[0], dst[0]);
      if (np == 1) break;
      const bool cdepth = w4 || p.depthC != 0 || p.depthVC != 0;
      if (ch == kChromaZero) {
        chromaFill();
      } else if (ch == kChromaIgnore || ch == kChromaCopy || !cdepth) {
        chromaCopy();
      } else if (ch == kChromaProcess || ch == kChromaOnly) {
        for (int i = 1; i < np; i++) ops.warp(src[i], edg[i], dst[i], p.depthC, p.depthVC, smagl);
      } else {
        CPlane<T> guide = edg[0];
        Buffer<T> g(dst[1].w, dst[1].h);
        if (fmt_.ssw || fmt_.ssh) {
          ops.guide(edg[0], g.plane, fmt_.ssw, fmt_.ssh, p.mpeg2);
          guide = g.plane;
        }
        for (int i = 1; i < np; i++) ops.warp(src[i], guide, dst[i], p.depthC, p.depthVC, smagl);
      }
      break;
    }
    case Kind::WarpSharp2:
    case Kind::WarpSharp: {
      // The output planes serve as blur scratch (the warp writes them afterwards), as in the original.
      Buffer<T> maskY(fmt_.width, fmt_.height);
      if (ch != kChromaOnly) {
        ops.sobel(src[0], maskY.plane, p.thresh);
        ops.blur(maskY.plane, dst[0], p.type, p.blur, p.blurV);
        if (ch != kChromaGuideOnly && (p.depth != 0 || p.depthV != 0))
          ops.warp(src[0], maskY.plane, dst[0], p.depth, p.depthV, 0);
        else
          copyPlane(src[0], dst[0]);
      } else {
        copyPlane(src[0], dst[0]);
      }
      if (np == 1) break;
      const bool cdepth = p.depthC != 0 || p.depthVC != 0;
      if (ch == kChromaZero) {
        chromaFill();
      } else if (ch == kChromaIgnore || ch == kChromaCopy || !cdepth) {
        chromaCopy();
      } else if (ch == kChromaProcess || ch == kChromaOnly) {
        Buffer<T> mc(dst[1].w, dst[1].h);
        for (int i = 1; i < np; i++) {
          ops.sobel(src[i], mc.plane, p.threshC);
          ops.blur(mc.plane, dst[i], p.type, p.blurC, p.blurVC);
          ops.warp(src[i], mc.plane, dst[i], p.depthC, p.depthVC, 0);
        }
      } else {
        CPlane<T> guide = maskY.plane;
        Buffer<T> g(dst[1].w, dst[1].h);
        if (fmt_.ssw || fmt_.ssh) {
          ops.guide(maskY.plane, g.plane, fmt_.ssw, fmt_.ssh, p.mpeg2);
          guide = g.plane;
        }
        for (int i = 1; i < np; i++) ops.warp(src[i], guide, dst[i], p.depthC, p.depthVC, 0);
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// aWarpSharp(original=true): MarcFD's aWarpSharp as written out in aWarpSharpMT 2.2.0 (8-bit YV12 there), on the
// sample scale of any bit depth. Kept in plain C: the in-place HQ blur must run in row order.

namespace {

template <typename T>
struct Orig {
  using N = Num<T>;
  int bits;
  N pmax;

  // PADDB of the original: the sum of both gradients wraps at 256 (2^bits; float: 256/255).
  N wrap(N s) const {
    if constexpr (std::is_same<T, float>::value)
      return s >= 256.0f / 255.0f ? s - 256.0f / 255.0f : s;
    else
      return s & ((1 << bits) - 1);
  }

  // Row copy with `pad` repeated edge samples on both sides: out[pad + x] = r[clamp(x)].
  static void padRow(const T* r, int w, int pad, N* out) {
    for (int i = 0; i < pad; i++) {
      out[i] = load(r, 0);
      out[pad + w + i] = load(r, w - 1);
    }
    for (int x = 0; x < w; x++) out[pad + x] = load(r, x);
  }

  void sobel(const CPlane<T>& s, const Plane<T>& d, N thr) const {
    const int w = s.w, h = s.h;
    std::vector<N> buf(3 * size_t(w + 2));
    N* const rows[3] = {buf.data(), buf.data() + (w + 2), buf.data() + 2 * (w + 2)};
    for (int y = 0; y < h; y++) {
      padRow(s.row(std::max(0, y - 1)), w, 1, rows[0]);
      padRow(s.row(y), w, 1, rows[1]);
      padRow(s.row(std::min(h - 1, y + 1)), w, 1, rows[2]);
      const N *u = rows[0] + 1, *m = rows[1] + 1, *dn = rows[2] + 1;
      T* o = d.row(y);
      for (int x = 0; x < w; x++) {
        const N tl = u[x - 1], tr = u[x + 1], bl = dn[x - 1], br = dn[x + 1];
        const N top = avg(tl, tr), bottom = avg(bl, br);
        const N v = absd(avg(u[x], top), avg(dn[x], bottom));
        const N hz = absd(avg(m[x - 1], avg(tl, bl)), avg(m[x + 1], avg(top, bottom)));
        const N sum = wrap(hz + v);
        N e = std::min(pmax, std::min(pmax, sum + sum) + sum);
        e = std::min(pmax, e + absd(hz, v));
        e = std::min(pmax, std::min(pmax, e + e) + e);
        o[x] = T(std::min(e, thr));
      }
    }
  }

  static N blur7(const N* q) {
    const N outer = avg(avg(q[6], q[5]), avg(q[4], q[3]));
    const N inner = avg(avg(q[2], q[1]), q[0]);
    return avg(avg(outer, inner), inner);
  }

  void blurH(const CPlane<T>& s, const Plane<T>& d) const {
    const int w = s.w;
    std::vector<N> buf(size_t(w) + 12);
    const N* r = buf.data() + 6;
    for (int y = 0; y < s.h; y++) {
      padRow(s.row(y), w, 6, buf.data());
      T* o = d.row(y);
      for (int x = 0; x < w; x++) {
        N q[7];
        q[0] = r[x];
        for (int k = 1; k <= 6; k++) q[k] = avg(r[x - k], r[x + k]);
        o[x] = T(blur7(q));
      }
    }
  }

  // Rows 0..5 read six rows below, the last six six rows above (these win in planes of 7..11 rows), the rest both.
  void blurV(const CPlane<T>& s, const Plane<T>& d) const {
    const int w = s.w, h = s.h;
    for (int y = 0; y < h; y++) {
      const int mode = (h > 6 && y >= h - 6) ? 2 : (y < 6 ? 0 : 1);
      const T* up[7];
      const T* dn[7];
      for (int k = 0; k <= 6; k++) {
        up[k] = s.row(std::max(0, std::min(h - 1, y - k)));
        dn[k] = s.row(std::max(0, std::min(h - 1, y + k)));
      }
      T* o = d.row(y);
      for (int x = 0; x < w; x++) {
        N q[7];
        q[0] = load(dn[0], x);
        for (int k = 1; k <= 6; k++)
          q[k] = mode == 0 ? load(dn[k], x) : (mode == 2 ? load(up[k], x) : avg(load(up[k], x), load(dn[k], x)));
        o[x] = T(blur7(q));
      }
    }
  }

  void blurH2(const CPlane<T>& s, const Plane<T>& d) const {
    const int w = s.w;
    std::vector<N> buf(size_t(w) + 4);
    const N* c = buf.data() + 2;
    for (int y = 0; y < s.h; y++) {
      padRow(s.row(y), w, 2, buf.data());
      T* o = d.row(y);
      for (int x = 0; x < w; x++) o[x] = T(avg(avg(c[x], avg(c[x - 1], c[x + 1])), avg(c[x - 2], c[x + 2])));
    }
  }

  // Rows 2..h-3; the centre row is read twice where the next row belongs (the original's MM3/MM4 load).
  void blurV2(const CPlane<T>& s, const Plane<T>& d) const {
    const int w = s.w, h = s.h;
    if (h < 4) return;
    for (int y = 2; y < h - 2; y++) {
      const T *a = s.row(y - 2), *b = s.row(y - 1), *c = s.row(y), *e = s.row(y + 2);
      T* o = d.row(y);
      for (int x = 0; x < w; x++)
        o[x] = T(avg(avg(load(c, x), avg(load(b, x), load(c, x))), avg(load(a, x), load(e, x))));
    }
  }

  void blur(const Plane<T>& m, const Plane<T>& t, int bm, int passes) const {
    for (int i = 0; i < passes; i++) {
      if (bm == 0) {
        blurHQPass(m);
      } else if (bm == 2) {
        blurH(m, t);
        blurV(t, m);
      } else {
        blurH2(m, t);
        blurV2(t, m);
      }
    }
  }

  static int64_t floorDiv(int64_t x, int64_t d) { return x >= 0 ? x / d : -((-x + d - 1) / d); }

  // Rows 0 and h-1 are copied. dx/dy in 1/256 pixel from the gradient of the mask on the 8-bit scale (PMULHW and
  // PADDSW of the original), source rows clamped to the plane.
  void warp(const CPlane<T>& s, const CPlane<T>& e, const Plane<T>& d, int depthH, int depthV) const {
    const int w = s.w, h = s.h, wm1 = w - 1;
    for (int y = 0; y < h; y++) {
      T* o = d.row(y);
      if (y == 0 || y == h - 1) {
        std::memcpy(o, s.row(y), size_t(w) * sizeof(T));
        continue;
      }
      const T* above = e.row(y - 1);
      const T* below = e.row(y + 1);
      const int ym1 = y - 1;
      for (int x = 0; x < w; x++) {
        const N gx = load(above, std::max(0, x - 1)) - load(above, std::min(wm1, x + 1));
        const N gy = load(above, x) - load(below, x);
        if constexpr (std::is_same<T, float>::value) {
          // The integer steps with exact fractions: 2 * gx * 255 * 128 * depth / 65536 in 1/256 pixel.
          float dx = std::max(-32768.0f, std::min(32767.0f, gx * 255.0f * float(depthH) / 256.0f));
          float dy = std::max(-32768.0f, std::min(32767.0f, gy * 255.0f * float(depthV) / 256.0f));
          dy = std::max(std::max(-float(ym1) * 256.0f, -16384.0f), std::min(dy, std::min(float(h - ym1) * 256.0f - 1.0f, 16384.0f)));
          const float sxf = float(x) + dx / 256.0f, syf = float(y) + dy / 256.0f;
          int sx = int(std::floor(sxf)), sy = int(std::floor(syf));
          float fx = sxf - float(sx);
          const float fy = syf - float(sy);
          if (sx < 0 || sx >= wm1) {
            sx = std::max(0, std::min(sx, wm1));
            fx = 0.0f;
          }
          const int sx1 = std::min(sx + 1, wm1);
          const T* a = s.row(std::min(sy, h - 1));
          const T* b = s.row(std::min(sy + 1, h - 1));
          const float top = a[sx] + (a[sx1] - a[sx]) * fx;
          const float bot = b[sx] + (b[sx1] - b[sx]) * fx;
          o[x] = top + (bot - top) * fy;
        } else {
          const int64_t den = int64_t(65536) << (bits - 8);
          const int dx = int(std::max<int64_t>(-32768, std::min<int64_t>(32767, 2 * floorDiv(int64_t(gx) * 128 * depthH, den))));
          int dy = int(std::max<int64_t>(-32768, std::min<int64_t>(32767, 2 * floorDiv(int64_t(gy) * 128 * depthV, den))));
          dy = std::max(std::max(-ym1 * 256, -16384), std::min(dy, std::min((h - ym1) * 256 - 1, 16384)));
          const int ix = int(floorDiv(dx, 256)), iy = int(floorDiv(dy, 256));
          int sx = x + ix;
          const int sy = y + iy;
          int fx = dx - ix * 256;
          const int fy = dy - iy * 256;
          if (sx < 0 || sx >= wm1) {
            sx = std::max(0, std::min(sx, wm1));
            fx = 0;
          }
          const int sx1 = std::min(sx + 1, wm1);
          const T* a = s.row(std::min(sy, h - 1));
          const T* b = s.row(std::min(sy + 1, h - 1));
          const int top = (a[sx] * (256 - fx) + a[sx1] * fx + 128) / 256;
          const int bot = (b[sx] * (256 - fx) + b[sx1] * fx + 128) / 256;
          o[x] = T((top * (256 - fy) + bot * fy + 128) / 256);
        }
      }
    }
  }
};

}  // namespace

template <typename T>
void Engine::runOriginal(const FrameView& srcf, const FrameOut& outf) const {
  const Params& p = p_;
  const int np = fmt_.planes;
  const Ops<T> ops{k_->get<T>(), fmt_.bits};
  const Orig<T> og{fmt_.bits, pixelMax<T>(fmt_.bits)};
  CPlane<T> src[3];
  Plane<T> dst[3];
  for (int i = 0; i < np; i++) {
    src[i] = view<T>(srcf, i, fmt_.planeWidth(i), fmt_.planeHeight(i));
    dst[i] = view<T>(outf, i, fmt_.planeWidth(i), fmt_.planeHeight(i));
    copyPlane(src[i], dst[i]);
  }
  const Num<T> thr = scale8<T>(p.thresh, fmt_.bits);
  const int passes = p.origBlur * (p.origBm == 2 ? 1 : 3);
  const int cpasses = (p.origBlur / 2) * (p.origBm == 2 ? 1 : 3);
  const int depth = std::min(p.origDepth, 32767);

  Buffer<T> mask(fmt_.width, fmt_.height), tmp(fmt_.width, fmt_.height);
  og.sobel(src[0], mask.plane, thr);
  og.blur(mask.plane, tmp.plane, p.origBm, passes);
  if (p.origShow) {
    copyPlane(CPlane<T>(mask.plane), dst[0]);
    return;
  }
  og.warp(src[0], mask.plane, dst[0], depth, depth);
  if (np == 1 || p.chroma == kChromaCopy) return;

  // Chroma displacement in chroma samples: the luma depth divided by the subsampling (depth / 2 in YV12).
  const int depthH = std::min(p.origDepth >> fmt_.ssw, 32767), depthV = std::min(p.origDepth >> fmt_.ssh, 32767);
  Buffer<T> mc(dst[1].w, dst[1].h);
  Plane<T> t = tmp.plane;
  t.w = dst[1].w;
  t.h = dst[1].h;
  if (p.chroma == kChromaGuide) {
    ops.guide(mask.plane, mc.plane, fmt_.ssw, fmt_.ssh, false);
    for (int i = 1; i < np; i++) og.warp(src[i], mc.plane, dst[i], depthH, depthV);
  } else {
    for (int i = 1; i < np; i++) {
      og.sobel(src[i], mc.plane, thr);
      og.blur(mc.plane, t, p.origBm, cpasses);
      og.warp(src[i], mc.plane, dst[i], depthH, depthV);
    }
  }
}

}  // namespace awarp
