# aWarpSharpMT

**Warp sharpening for AviSynth+ and VapourSynth**: aWarpSharp2, aWarpSharp (including MarcFD's original
algorithm), aSobel, aBlur, aWarp and aWarp4 in one plugin binary. Built for Windows x64, Linux x86_64 and macOS
arm64, with SSE4.1, AVX2 and NEON code paths that give exactly the same output as the plain C code.

This is a port of aWarpSharpMT 2.2.0 (JPSDR), which was a Windows-only AviSynth plugin with MASM assembler and its own
thread pool. The port gives the same output (8-16 bit, see [Differences](#differences-to-awarpsharpmt-220)) and adds:

- a VapourSynth (API 4) interface next to the AviSynth+ one,
- Linux and macOS builds; SIMD as intrinsics (SSE4.1, AVX2, NEON) instead of x86 assembler,
- 32 bit float for every function,
- `aWarpSharp(original=true)` for every supported format (Y, 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0) and bit depth
  (8-16 bit, float), not only 8-bit YV12,
- every chroma subsampling for the luma-guided chroma modes (4:1:1 and 4:4:0 too).

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions and parameters](#functions-and-parameters)
- [aWarpSharp and original=true](#awarpsharp-and-originaltrue)
- [Formats](#formats)
- [Differences to aWarpSharpMT 2.2.0](#differences-to-awarpsharpmt-220)
- [Performance](#performance)
- [Build from source](#build-from-source)
- [Credits and license](#credits-and-license)

## Installation

Release files: `aWarpSharpMT-<version>-<platform>.zip` (plugin and docs) and a VapourSynth wheel.

| | AviSynth+ | VapourSynth |
|---|---|---|
| Plugin | `aWarpSharpMT.dll` into the `plugins64` folder, or `LoadPlugin("aWarpSharpMT.dll")` | `pip install vapoursynth_awarpsharpmt-<version>-py3-none-<platform>.whl` (installs into `site-packages/vapoursynth/plugins`), or the plugin file into a plugin folder / `core.std.LoadPlugin()` |
| File | `aWarpSharpMT.dll` (Windows) | `aWarpSharpMT.dll`, `libaWarpSharpMT.so`, `libaWarpSharpMT.dylib` |
| Needs | AviSynth+ (frame properties are kept from interface version 8) | API 4 (R55 or newer) |

The best instruction set is chosen at runtime; any x86-64 CPU works (C code without SSE4.1).

## Usage

AviSynth+, with the argument lists of aWarpSharpMT 2.2.0, so existing scripts work unchanged:

```
aWarpSharp2()                                   # thresh 128, blur 2, type 0, depth 16, chroma 4
aWarpSharp2(depth=8, chroma=3)
aWarpSharp(depth=16.0, blurlevel=2, thresh=0.5, cm=1, original=true)   # MarcFD's aWarpSharp
aWarp(aSobel().aBlur(), depth=3)
aWarp4(Spline36Resize(width*4, height*4, 0.375, 0.375), aSobel().aBlur(), depth=3)
```

VapourSynth (namespace `awarpsharpmt`, the same parameter names; the mask clip of aWarp/aWarp4 is `edges`):

```python
import vapoursynth as vs
core = vs.core
m = core.awarpsharpmt
clip = m.aWarpSharp2(clip, depth=8, chroma=3)
clip = m.aWarpSharp(clip, original=True)
edges = m.aBlur(m.aSobel(clip))
clip = m.aWarp(clip, edges, depth=3)
clip = m.aWarp4(clip.resize.Spline36(clip.width * 4, clip.height * 4, src_left=0.375, src_top=0.375), edges, depth=3)
```

## Functions and parameters

aWarpSharp2 runs the three steps in one go: an edge mask (aSobel), a blur of the mask (aBlur), and a warp of the
picture along the gradient of the mask (aWarp). `aWarpSharp2()` gives the same output as `aWarp(aSobel().aBlur())`
with the same settings. aWarp4 warps a source four times the size of the mask (better subpixel interpolation; the
upscale has to be top-left aligned, see the examples above). Blurring the mask, or filtering it temporally, before the
warp gives more stable results.

| Function | Arguments (AviSynth+ order) |
|---|---|
| aWarpSharp2 | clip, `thresh`, `blur`, `type`, `depth`, `chroma`, `depthC`, `cplace`, `blurV`, `depthV`, `depthVC`, `blurC`, `blurVC`, `threshC` |
| aWarpSharp | clip, `depth` (float), `blurlevel`, `thresh` (float), `cm`, `bm`, `show`, `original` |
| aSobel | clip, `thresh`, `chroma`, `threshC` |
| aBlur | clip, `blur`, `type`, `chroma`, `blurV`, `blurC`, `blurVC` |
| aWarp, aWarp4 | clip, edge mask clip, `depth`, `chroma`, `depthC`, `cplace`, `depthV`, `depthVC` |

Every function also takes `opt` (last, new) and, in AviSynth+ only, the thread pool arguments of aWarpSharpMT 2.2.0
`threads`, `logicalCores`, `MaxPhysCore`, `SetAffinity`, `sleep`, `prefetch`, `ThreadLevel`, which are accepted and
ignored (the hosts run the filters in parallel, see [Performance](#performance)).

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `thresh` | 128 | 0..255 | edge mask saturation limit; lower = less aggressive sharpening |
| `threshC` | `thresh` | 0..255 | the same for the chroma planes |
| `blur` | 2 (type 0), 3 (type 1) | >= 0 | blur passes over the mask; fewer passes sharpen more but can produce artefacts at high depth and thresh |
| `blurV` | `blur` | >= 0 | vertical passes; if it differs from `blur`, `blur` counts the horizontal passes only |
| `blurC`, `blurVC` | `(blur+1)/2`, `blurC` | >= 0 | the same for the chroma planes |
| `type` | 0 (aWarpSharp2), 1 (aBlur) | 0, 1 | 0: radius 6 blur; 1: radius 2 blur, about 8x as many passes for the same effect (2.5x as slow), better quality |
| `depth` | 16 (aWarpSharp2), 3 (aWarp, aWarp4) | -128..127 | warp strength; negative warps the other way |
| `depthV` | `depth` | -128..127 | vertical strength; `depth` is then the horizontal one |
| `depthC`, `depthVC` | `depth` (4:4:4, Y), else `depth/2`; `depthC` | -128..127 | the same for the chroma planes |
| `chroma` | 4 (warp functions), 1 (aSobel, aBlur) | 0..6 | chroma processing, see below |
| `cplace` | `"MPEG1"` | `"MPEG1"`, `"MPEG2"` | chroma sample position for the luma-guided modes 4 and 6 in 4:2:0 and 4:2:2 |
| `opt` | -1 | -1..3 | instruction set: -1 best, 0 C, 1 SSE4.1, 2 AVX2, 3 NEON; error if not available |

| `chroma` | U and V |
|---|---|
| 0 | filled with the neutral value (128 at 8 bit, 0.0 in float) |
| 1 | "don't care": copied from the source (aWarp4: every fourth sample of the large clip) |
| 2 | copied (aWarp4: as 1) |
| 3 | processed with their own masks |
| 4 | warped along the luma mask, downscaled to the chroma size (warp functions; aSobel/aBlur: as 3) |
| 5 | as 3, luma is copied (aWarp4: luma as in 1) |
| 6 | as 4, luma is copied (aWarp4: luma as in 1) |

At 9-16 bit the thresholds stay on the 8-bit scale (shifted by `bits - 8`); in float they are divided by 255, and the
mask, blur and warp run with exact fractions (the warp moves a sample by `mask difference * depth` pixels, which is
what the integer code does on the sample scale).

## aWarpSharp and original=true

`aWarpSharp(depth, blurlevel, thresh, cm, bm, show, original)` takes the parameters of MarcFD's aWarpSharp (2003).

- `original=false` (default) maps them to aWarpSharp2 as aWarpSharpMT 2.2.0 does: `thresh = thresh*256`,
  `blur = blurlevel` (`bm=2`) or `blurlevel*3` with `type=1` (`bm` 0 and 1), `depth = depth*blurlevel/2`, chroma
  `cm` 0 -> 1, 1 -> 4, 2 -> 3, 3 -> 2. `show` is ignored.
- `original=true` runs MarcFD's algorithm as aWarpSharpMT 2.2.0 writes it out (in plain C there): a Sobel mask with
  the byte wraparound of the original (PADDB), `bm` 0 = an in-place "high quality" blur (row by row, each row reads the
  rows above already blurred), 1 = a radius 2 blur with 3 passes per level, 2 = a radius 6 blur with 1 pass per level,
  and a warp with 1/256-pixel steps. `cm` 0 copies the chroma, 1 warps it along the luma mask, 2 with its own masks;
  `show=true` returns the mask in the luma plane. Ranges: `depth` 0..64, `blurlevel` 1..4, `thresh` 0.0..1.0 (passed
  as a byte: 1.0 becomes 0, as in the original), `bm` 0..2, `cm` 0..2.

aWarpSharpMT 2.2.0 limits `original=true` to 8-bit YV12. This port runs it on every format: at 9-16 bit the mask is
on the sample scale (wraparound at 2^bits, threshold shifted by `bits - 8`, the gradient divided by 2^(bits-8) for
the displacement), in float with exact fractions (wraparound at 256/255). The luma mask is reduced to the chroma size
by rounded pair averages (2x2 in 4:2:0 as in the original), and the chroma displacement is the luma depth divided by
the subsampling in each direction (depth/2 in YV12, as in the original).

## Formats

| | AviSynth+ | VapourSynth |
|---|---|---|
| Supported | Y, planar YUV and YUVA (alpha copied), 8-16 bit and 32 bit float, any subsampling | Gray, YUV, 8-16 bit integer and 32 bit float |
| Not supported | RGB, YUY2 | RGB, half float |

aWarp needs a mask clip of the same format and size, aWarp4 a source of four times the mask's width and height; the
output has the mask's size. Thread-safe and independent of the request order (AviSynth+ `MT_NICE_FILTER`,
VapourSynth `fmParallel`). Frame properties are kept.

## Differences to aWarpSharpMT 2.2.0

For 8-16 bit the output is identical to aWarpSharpMT 2.2.0 called with `threads=1` on its SSE2 code path
(`SetMaxCPU("sse4.2")`), for every function, parameter and format the original accepts (`test/compare_original.py`:
567 cases on three clips), except where the original's result is not defined by the picture:

- **Last column of the warp:** the original reads one mask sample past the end of the row there (row padding). The
  port uses the last sample of the row.
- **8-bit radius 6 blur on AVX CPUs:** the original has a separate AVX version of the horizontal blur that repeats
  the edge pixels (the SSE2 version takes all six neighbours from the inner side, like every aWarpSharp2 before it),
  and reads one byte of row padding when the width is not a multiple of 16. So the original's output depends on the
  CPU; the port gives the SSE2 result everywhere.
- **16-bit luma-guided chroma with `cplace="MPEG2"`:** the original's multithreaded code reads the row below the next
  one and truncates the result to 8 bit; the port gives the result of `threads=1`. In 4:2:2 MPEG1 the original's
  non-AVX code averages each sample with itself; the port gives the AVX result.
- **`chroma=1` ("don't care"):** the original leaves these planes undefined; the port copies them.
- **`original=true`:** jpsdr's code reads up to two rows below the plane at the bottom edge (the next plane in
  memory). The port repeats the last row.
- **YV411:** the original's results change from run to run (and its guided modes reject 4:1:1); the port's are
  defined.
- The thread pool arguments are ignored; use `Prefetch()` (AviSynth+) or VapourSynth's threads.

## Performance

AviSynth+ 3.7.6, 1920x1080 4:2:0, frames per second, AMD Ryzen 9 7950X (32 threads), AVX2. aWarpSharpMT 2.2.0 with
`threads=1`, with its internal thread pool (default `threads=0`), and with `threads=1` under `Prefetch(32)`:

| 1080p | 2.2.0, 1 thread | port, 1 thread | 2.2.0, own threads | 2.2.0, Prefetch(32) | port, Prefetch(32) |
|---|---:|---:|---:|---:|---:|
| `aWarpSharp2()` 8 bit | 290 | 214 | 1072 | 1655 | **1765** |
| `aWarpSharp2(chroma=3)` 8 bit | 264 | 223 | 871 | 1549 | **1768** |
| `aWarpSharp2(type=1, blur=7)` 8 bit | 269 | 221 | 572 | 917 | **1187** |
| `aWarpSharp2()` 16 bit | 165 | 141 | 781 | **691** | 689 |
| `aWarpSharp(original=true)` 8 bit | 13.3 | **16.1** | 163 | 218 | **268** |
| `aWarpSharp(original=true, bm=0)` 8 bit | 16.6 | **23.3** | 34.7 | 277 | **404** |
| `aWarpSharp2()` float | - | 93 | - | - | 322 |
| `aWarpSharp(original=true)` float | - | 22.4 | - | - | 271 |

With all threads the port is as fast as the original or faster; single-threaded it reaches 75-85% of the original's
hand-written SSE2/AVX assembler in the default mode (both gather the warp's source samples one by one) and is faster
in `original=true` mode. `bm=0` is serial inside a frame (each row reads the rows above already blurred), so only
frame threading helps there.

## Build from source

Meson and a C++17 compiler (MSVC, GCC or Clang):

```
meson setup build            # Windows: meson setup build --vsenv -Db_vscrt=mt   (or run build.bat)
meson compile -C build
meson test -C build          # every SIMD tier against the C kernels
```

Wheel: `pip wheel .` (Windows: add `--config-settings=setup-args=--vsenv --config-settings=setup-args=-Db_vscrt=mt`).

Tests in `test/`:

| Script | Checks |
|---|---|
| `kernel_test.cpp` | `meson test`: every SIMD tier against the C kernels on random data, 8/9-16 bit and float, many widths |
| `test_awarpsharpmt.py [plugin]` | VapourSynth: every function and format, every instruction set against C, the C output against MD5 digests (so every platform computes the same), argument errors |
| `compare_original.py` | AviSynth+ (Windows, `avs2yuv64`, BestSource): against the aWarpSharpMT 2.2.0 DLL, plane by plane |
| `compare_hosts.py` | the AviSynth+ and the VapourSynth interface give identical frames |

## Credits and license

- aWarpSharp © 2003 MarcFD; aWarpSharp2 © 2012 Skakov Pavel (SEt); modifications © 2015 Firesledge (cretindesalpes),
  2016 pinterf; aWarpSharpMT © JPSDR (thread pool, high bit depth, `original` mode written out after ChaosKing's
  description of the original algorithm).
- aWarpSharp2 for VapourSynth © dubhater: the C formulation of SEt's kernels.
- This port © 2026 Selur.

Released under the GNU General Public License version 2 or later, see [COPYING.txt](COPYING.txt). The vendored
VapourSynth headers in `include/vapoursynth` are LGPL-2.1, see the license file there.
