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
- every chroma subsampling for the luma-guided chroma modes (4:1:1 and 4:4:0 too),
- `type=2..4` in aBlur and aWarpSharp2: the HQ, fast 3-pass and fast 1-pass blurs of MarcFD's original,
- `mask=1` in aSobel and aWarpSharp2: the edge mask formula of MarcFD's original,
- `wmode=1` in aWarp and aWarpSharp2: the warp of MarcFD's original. With `mask=1` and `type=2..4`, aWarpSharp2 gives
  exactly `aWarpSharp(original=true)`, see [Original warp (wmode=1)](#original-warp-wmode1).

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions and parameters](#functions-and-parameters)
- [aWarpSharp and original=true](#awarpsharp-and-originaltrue)
- [Original blurs (type=2..4)](#original-blurs-type24)
- [Original mask (mask=1)](#original-mask-mask1)
- [Original warp (wmode=1)](#original-warp-wmode1)
- [Formats](#formats)
- [Differences to aWarpSharpMT 2.2.0](#differences-to-awarpsharpmt-220)
- [Differences to vapoursynth-awarpsharp2](#differences-to-vapoursynth-awarpsharp2)
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

AviSynth+, with the argument lists of aWarpSharpMT 2.2.0, so existing scripts work unchanged (except `aWarpSharp`,
whose `original` now defaults to true):

```
aWarpSharp2()                                   # thresh 128, blur 2, type 0, depth 16, chroma 4
aWarpSharp2(depth=8, chroma=3)
aWarpSharp(depth=16.0, blurlevel=2, thresh=0.5, cm=1)   # MarcFD's aWarpSharp (original=true)
aWarp(aSobel().aBlur(), depth=3)
aWarp4(Spline36Resize(width*4, height*4, 0.375, 0.375), aSobel().aBlur(), depth=3)
```

VapourSynth (namespace `awarpsharpmt`, the same parameter names; the mask clip of aWarp/aWarp4 is `edges`):

```python
import vapoursynth as vs
core = vs.core
m = core.awarpsharpmt
clip = m.aWarpSharp2(clip, depth=8, chroma=3)
clip = m.aWarpSharp(clip)                        # MarcFD's aWarpSharp
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
| aWarpSharp2 | clip, `thresh`, `blur`, `type`, `depth`, `chroma`, `depthC`, `cplace`, `blurV`, `depthV`, `depthVC`, `blurC`, `blurVC`, `threshC`, `mask`, `wmode` |
| aWarpSharp | clip, `depth` (float), `blurlevel`, `thresh` (float), `cm`, `bm`, `show`, `original` |
| aSobel | clip, `thresh`, `chroma`, `threshC`, `mask` |
| aBlur | clip, `blur`, `type`, `chroma`, `blurV`, `blurC`, `blurVC` |
| aWarp | clip, edge mask clip, `depth`, `chroma`, `depthC`, `cplace`, `depthV`, `depthVC`, `wmode` |
| aWarp4 | clip, edge mask clip, `depth`, `chroma`, `depthC`, `cplace`, `depthV`, `depthVC` |

Every function also takes `opt` (last, new) and, in AviSynth+ only, the thread pool arguments of aWarpSharpMT 2.2.0
`threads`, `logicalCores`, `MaxPhysCore`, `SetAffinity`, `sleep`, `prefetch`, `ThreadLevel`, which are accepted and
ignored (the hosts run the filters in parallel, see [Performance](#performance)).

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `thresh` | 128 | 0..255 | edge mask saturation limit; lower = less aggressive sharpening |
| `threshC` | `thresh` | 0..255 | the same for the chroma planes |
| `mask` | 0 | 0, 1 | edge mask formula: 0 aWarpSharp2, 1 MarcFD's original, see [Original mask (mask=1)](#original-mask-mask1) |
| `blur` | 2 (type 0, 4), 3 (type 1), 6 (type 2, 3) | >= 0 | blur passes over the mask; fewer passes sharpen more but can produce artefacts at high depth and thresh |
| `blurV` | `blur` | >= 0 | vertical passes; if it differs from `blur`, `blur` counts the horizontal passes only (not with type 2..4) |
| `blurC`, `blurVC` | `(blur+1)/2`, `blurC` | >= 0 | the same for the chroma planes (type 2..4: `blurVC` must equal `blurC`) |
| `type` | 0 (aWarpSharp2), 1 (aBlur) | 0..4 | 0: radius 6 blur; 1: radius 2 blur, about 8x as many passes for the same effect (2.5x as slow), better quality; 2, 3, 4: the HQ, fast 3-pass and fast 1-pass blur of MarcFD's original, see [Original blurs (type=2..4)](#original-blurs-type24) |
| `wmode` | 0 | 0, 1 | warp (aWarpSharp2, aWarp): 0 aWarpSharp2, 1 MarcFD's original, see [Original warp (wmode=1)](#original-warp-wmode1) |
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

- `original=false` maps them to aWarpSharp2 as aWarpSharpMT 2.2.0 does: `thresh = thresh*256`,
  `blur = blurlevel` (`bm=2`) or `blurlevel*3` with `type=1` (`bm` 0 and 1), `depth = depth*blurlevel/2`, chroma
  `cm` 0 -> 1, 1 -> 4, 2 -> 3, 3 -> 2. `show` is ignored.
- `original=true` (default) runs MarcFD's algorithm as aWarpSharpMT 2.2.0 writes it out (in plain C there), as the
  aWarpSharp2 steps `mask=1`, `type=2..4` and `wmode=1` (see [Original warp (wmode=1)](#original-warp-wmode1)): a Sobel
  mask with the byte wraparound of the original (PADDB), `bm` 0 = an in-place "high quality" blur (row by row, each row
  reads the rows above already blurred), 1 = a radius 2 blur with 3 passes per level, 2 = a radius 6 blur with 1 pass
  per level, and a warp with 1/256-pixel steps. `cm` 0 copies the chroma, 1 warps it along the luma mask, 2 with its
  own masks; `show=true` returns the mask in the luma plane. Ranges: `depth` 0..64, `blurlevel` 1..4, `thresh`
  0.0..1.0 (passed as a byte: 1.0 becomes 0, as in the original), `bm` 0..2, `cm` 0..2.

aWarpSharpMT 2.2.0 limits `original=true` to 8-bit YV12. This port runs it on every format: at 9-16 bit the mask is
on the sample scale (wraparound at 2^bits, threshold shifted by `bits - 8`, the gradient divided by 2^(bits-8) for
the displacement), in float with exact fractions (wraparound at 256/255). The luma mask is reduced to the chroma size
by rounded pair averages (2x2 in 4:2:0 as in the original), and the chroma displacement is the luma depth divided by
the subsampling in each direction (depth/2 in YV12, as in the original).

## Original blurs (type=2..4)

`type=2`, `3` and `4` in aBlur and aWarpSharp2 are the three blurs of MarcFD's aWarpSharp, `bm=0`, `1` and `2` there
(`aWarpSharp(original=true)` runs the same code). Each pass blurs both directions, so `blurV` must equal `blur` and
`blurVC` must equal `blurC` (left out, they are).

| `type` | Original | Default `blur` | One pass |
|---|---|---|---|
| 2 | `bm=0` "high quality", 3 passes per blur level | 6 | in place, see below |
| 3 | `bm=1` "fast 3-pass", 3 passes per blur level | 6 | radius 2 across (rounded averages of 1-1, 2-2 neighbours), then down rows 2..h-3 with the centre row read twice instead of the row below; the top and bottom two rows keep their values from before the pass |
| 4 | `bm=2` "fast 1-pass", 1 pass per blur level | 2 | radius 6 across and down as a cascade of rounded averages; rows 0..5 take their neighbours from below only, the last six rows from above only |

The defaults are the strength of the original's default `blurlevel=2`. These blurs run in plain C. VapourSynth, 1080p
8 bit `aWarpSharp2`: `type=3` 132 fps with one thread and 896 fps with 32, `type=4` 48 and 715 fps, against 231 and
1787 fps for `type=0`.

`type=2` in detail: one pass works in place, top to bottom, and blurs both directions at once:

- vertical taps 1, 4, 10, 1 on the rows y-2, y-1, y, y+2: the centre row stands in for the row below, as in the
  original's assembler; rows y-2 and y-1 are already blurred in this pass, so mask values are carried downwards,
- then horizontal taps 1, 4, 6, 4, 1 (columns clamped at the edges), divided by 256 with `+64` (rounds down, as in
  the original; float exact),
- the top and bottom two rows are not changed.

Inside a frame the passes run in row order on one thread; frames still run in parallel. VapourSynth, 1080p 8 bit
`aWarpSharp2`: `type=2` (6 passes) 103 fps with one thread and 1410 fps with 32, against 240 and 2876 fps for
`type=1, blur=6`.

`aWarpSharp(original=false, bm=0)` keeps the 2.2.0 mapping to `type=1`.

## Original mask (mask=1)

`mask=1` in aSobel and aWarpSharp2 builds the edge mask with the formula of MarcFD's aWarpSharp
(`aWarpSharp(original=true)` runs the same code). Against the default `mask=0`:

| | `mask=0` (aWarpSharp2) | `mask=1` (original) |
|---|---|---|
| sum of the two gradients | saturates at the maximum value | wraps around like the MMX `PADDB` (at 8 bit 250 + 10 = 4; 9-16 bit at 2^bits, float at 256/255): very strong edges can get a small mask value |
| amplification | + max(h, v), x3, x2 | x3, + \|h - v\|, x3 |
| horizontal gradient | right side averages the two right corners | right side averages all four corners |
| borders | columns 0 and w-1 copy columns 1 and w-2, block quirks of 2.2.0's SSE2 code in the first and last two rows | every row and column computed with clamped neighbours |

`thresh` and `threshC` keep their meaning (0..255 on the 8-bit scale). Together with the HQ blur the mask of the
original is reproduced exactly: the luma of `aBlur(aSobel(thresh=T, mask=1), type=2, blur=3*L)` equals the luma of
`aWarpSharp(original=true, bm=0, show=true, thresh=T/256, blurlevel=L)` (tested for every format). With the default
`wmode=0` the warp is aWarpSharp2's, so `aWarpSharp2(mask=1, type=2)` alone is not the original algorithm; together
with `wmode=1` it is, see [Original warp (wmode=1)](#original-warp-wmode1).

`mask=1` has SSE4.1, AVX2 and NEON code like `mask=0` and is as fast: VapourSynth, 1080p 8 bit, one thread,
`aWarpSharp2` 247 fps with `mask=1` against 249 fps with `mask=0` (32 threads: 2837 against 2848 fps).

## Original warp (wmode=1)

`wmode=1` in aWarp and aWarpSharp2 warps with the code of MarcFD's aWarpSharp (`aWarpSharp(original=true)` runs the
same code): the displacement comes from the mask, across from the left and right neighbours in the row above, down
from the rows above and below, in 1/256-pixel steps, then a bilinear interpolation with 1/256 weights; the first and
the last row are copied. `depth` keeps its scale: `depth` 1 is 256 units of the original, whose depth is `depth * blurlevel * 128`, so
`aWarpSharp(depth=16, blurlevel=2)` is `depth=16` here, as in aWarpSharpMT 2.2.0's mapping. aWarp4 has no `wmode`.

`aWarpSharp(original=true, depth=D, blurlevel=L, thresh=T, cm, bm)` gives the same frames as this aWarpSharp2 call:

| aWarpSharp2 | Value |
|---|---|
| `mask`, `wmode` | 1, 1 |
| `thresh` | `int(T * 256) & 255` (1.0 becomes 0, as in the original) |
| `type` | `bm` 0, 1, 2 -> 2, 3, 4 |
| `blur` | `L` with `bm=2`, else `3 * L` |
| `blurC` | `L / 2` (rounded down) with `bm=2`, else `3 * (L / 2)` |
| `depth`, `depthV` | `D * L / 2` |
| `depthC`, `depthVC` | `D * L / 2` divided by the horizontal and the vertical chroma subsampling |
| `chroma` | `cm` 0, 1, 2 -> 2, 4, 3 |

This holds whenever the depths come out as whole numbers (tested for every format, bit depth, `bm` and `cm`). With the
defaults it is simply `aWarpSharp2(mask=1, type=4, wmode=1)` in 4:2:0, 4:4:4 and Y; 4:2:2, 4:1:1 and 4:4:0 need
`depthC` and `depthVC`, because the original divides the chroma depth per direction.

`aWarpSharp(original=true)` runs exactly this path internally. It hands its depth to the warp in the original's units,
so depths that are no whole numbers on the aWarpSharp2 scale (`depth=15, blurlevel=1`) work there, and `show=true`
returns the blurred mask after the mask and blur steps.

The warp runs in plain C. VapourSynth, 1080p 8 bit: `aWarpSharp2(mask=1, type=4, wmode=1)` 23 fps with one thread and
372 fps with 32 threads, as fast as `aWarpSharp(original=true)` (23 and 349 fps).

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
- **`aWarpSharp` default:** `original` defaults to true (2.2.0: false); `original=false` gives the 2.2.0 default.
- The thread pool arguments are ignored; use `Prefetch()` (AviSynth+) or VapourSynth's threads.

## Differences to vapoursynth-awarpsharp2

VapourSynth also has the `warp` plugin ([vapoursynth-awarpsharp2](https://github.com/Selur/vapoursynth-awarpsharp2),
dubhater's port of aWarpSharp2) with `AWarpSharp2`, `ASobel`, `ABlur` and `AWarp`. Compared with this plugin on the
same input (4:2:0 8, 10 and 16 bit, the same edge mask for the warps):

| This plugin | `warp` | Output |
|---|---|---|
| `aSobel` | `ASobel` | Differs only in rows 0 and 1 (chroma also in the last two rows): this plugin keeps the original's edge handling there; every other sample is identical at every bit depth. `mask=1` exists only here. |
| `aBlur` | `ABlur` | Identical (type 0 and 1, 8-16 bit). `ABlur` always blurs chroma (blur+1)/2 times, which is this plugin's default `blurC`. `type=2..4` exist only here. |
| `aWarp` | `AWarp` | 8 bit identical (own chroma masks and luma-guided chroma). 10-16 bit slightly different: the two plugins round the high bit depth warp differently; this plugin follows aWarpSharpMT 2.2.0 (10 bit, `depth=16`: 16% of the luma samples differ, by at most 15 of 1023). `wmode=1` exists only here. |
| `aWarp4` | `AWarp` with a 4x clip | Same as `aWarp`: identical at 8 bit, slightly different at 10-16 bit. |
| `aWarpSharp2` | `AWarpSharp2` | 8 bit: a few pixels next to the top edge, carried there from the first two mask rows (0.008% of the luma samples, by at most 5). 10-16 bit: additionally the warp rounding above. |
| `aWarpSharp` | – | No counterpart. With `original=false` it maps to an `aWarpSharp2` call, so it inherits the `aWarpSharp2` differences; `original=true` (MarcFD's algorithm) exists only here. |

Parameters differ as well: `warp` takes `depth` as a list per plane, `chroma` 0/1 and `planes`; this plugin takes the
AviSynth parameter set (`chroma` 0-6, separate `depthC`, `depthV`, `blurV`, `blurC`, `threshC`, `cplace`). Only this
plugin accepts 32 bit float and subsamplings beyond 2 (for example 4:1:1). Speed: with one thread both are equally
fast (1080p 8 bit `aWarpSharp2`: 257 against 256 fps).

## Performance

AviSynth+ 3.7.6, 1920x1080 4:2:0, frames per second, AMD Ryzen 9 7950X (32 threads), AVX2, Windows release build
(clang-cl). aWarpSharpMT 2.2.0 with `threads=1`, with its internal thread pool (default `threads=0`), and with
`threads=1` under `Prefetch(32)`:

| 1080p | 2.2.0, 1 thread | port, 1 thread | 2.2.0, own threads | 2.2.0, Prefetch(32) | port, Prefetch(32) |
|---|---:|---:|---:|---:|---:|
| `aWarpSharp2()` 8 bit | **293** | 258 | 1320 | 1700 | **2954** |
| `aWarpSharp2(chroma=3)` 8 bit | 275 | **287** | 1113 | 1553 | **2789** |
| `aWarpSharp2(type=1, blur=7)` 8 bit | **268** | 225 | 767 | 958 | **2776** |
| `aWarpSharp2()` 16 bit | **167** | 157 | 894 | 693 | **1281** |
| `aWarpSharp(original=true)` 8 bit | 13.4 | **21.0** | 176 | 222 | **370** |
| `aWarpSharp(original=true, bm=0)` 8 bit | 17.2 | **27.8** | 35.5 | 276 | **508** |
| `aWarpSharp2()` float | - | 105 | - | - | 321 |
| `aWarpSharp(original=true)` float | - | 47.5 | - | - | 266 |

Single-threaded the port reaches 84-104% of the original's hand-written SSE2/AVX assembler in the default mode and
1.6x its speed in `original=true` mode. With all threads it is 1.7-2.9x as fast: the blur passes run row by row
behind each other on planes of 1 MiB and more, so the mask stays in the cache instead of going through memory once
per pass. This matters most for large frames and `blur` values: VapourSynth, 32 threads, 3840x2160 8 bit
`aWarpSharp2(blur=32)` 227 fps (two sweeps per pass: 37 fps), 1920x1080 16 bit 494 fps (84 fps). `bm=0` is serial
inside a frame (each row reads the rows above already blurred), so only frame threading helps there.

## Build from source

Meson and a C++17 compiler (GCC, Clang, clang-cl or MSVC):

```
meson setup build            # Windows: meson setup build --vsenv -Db_vscrt=mt   (or run build.bat)
meson compile -C build
meson test -C build          # every SIMD tier against the C kernels
```

The Windows release is built with clang-cl (`set CXX=clang-cl` before `meson setup`; `build.bat` picks it when it is
on the PATH), which gives about 5-10% faster code than MSVC here; both produce the same output.

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
