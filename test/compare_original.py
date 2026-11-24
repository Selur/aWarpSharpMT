#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compares the port with the original aWarpSharpMT 2.2.0 DLL (JPSDR) inside AviSynth+ (Windows).

Both scripts read the same y4m clips with BestSource and convert them the same way. Every case compares all output
planes sample by sample; planes the original leaves undefined ("don't care") are skipped. Cases the original rejects
are listed as skipped.

Usage: python -I compare_original.py --original aWarpsharpMT.dll --plugin build/aWarpSharpMT.dll --work DIR
       --avs2yuv avs2yuv64.exe --avs-bestsource BestSource.dll [--maxcpu sse4.2] [--only REGEX] CLIP.y4m...
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys

import numpy as np

# name -> (conversion, bits, planes, ssw, ssh)
FORMATS = {
    "YV12": ("", 8, 3, 1, 1),
    "YV16": ("ConvertToYV16()", 8, 3, 1, 0),
    "YV24": ("ConvertToYV24()", 8, 3, 0, 0),
    "Y8": ("ConvertToY8()", 8, 1, 0, 0),
    "YV411": ("Crop(0, 0, (width / 4) * 4, 0).ConvertToYV411()", 8, 3, 2, 0),
    "P10": ("ConvertBits(10)", 10, 3, 1, 1),
    "P16": ("ConvertBits(16)", 16, 3, 1, 1),
    "P12_444": ("ConvertToYUV444().ConvertBits(12)", 12, 3, 0, 0),
    "P16_422": ("ConvertToYUV422().ConvertBits(16)", 16, 3, 1, 0),
    "Y16": ("ConvertToY().ConvertBits(16)", 16, 1, 0, 0),
}
# YV411 is left out by default: the original gives different results from run to run there.
MAIN_FORMATS = ["YV12", "YV16", "YV24", "Y8", "P10", "P16", "P12_444", "P16_422", "Y16"]

ORIGINAL_BORDER = 96
EDGES = "edg = last.aSobel(chroma=3).aBlur(chroma=3)\n"
BIG = "big = last.Spline36Resize(width*4, height*4, 0.375, 0.375)\n"

# (function call template, chroma planes undefined in the original, all formats?)
CASES = [
    ("aWarpSharp2()", False, True),
    ("aWarpSharp2(depth=30)", False, False),
    ("aWarpSharp2(chroma=3)", False, True),
    ("aWarpSharp2(chroma=0)", False, False),
    ("aWarpSharp2(chroma=2)", False, False),
    ("aWarpSharp2(chroma=5)", False, False),
    ("aWarpSharp2(chroma=6)", False, False),
    ("aWarpSharp2(type=1)", False, True),
    ("aWarpSharp2(type=1, blur=7, blurV=3, chroma=3)", False, True),
    ("aWarpSharp2(blur=4, blurV=1)", False, False),
    ("aWarpSharp2(blur=0)", False, False),
    ("aWarpSharp2(thresh=60, threshC=200, chroma=3)", False, False),
    ("aWarpSharp2(depth=-20, depthV=5)", False, False),
    ("aWarpSharp2(cplace=\"MPEG2\")", False, True),
    ("aWarpSharp2(chroma=3, blurC=3, blurVC=0)", False, False),
    ("aWarpSharp2(depthC=20, depthVC=-10)", False, False),
    ("aWarpSharp2(depth=127, thresh=255, blur=1)", False, False),
    ("aSobel()", True, True),
    ("aSobel(thresh=255, chroma=3)", False, True),
    ("aSobel(chroma=5, threshC=40)", False, False),
    ("aSobel(chroma=0)", False, False),
    ("aBlur()", True, True),
    ("aBlur(type=0, chroma=3)", False, True),
    ("aBlur(blur=5, blurV=2, chroma=3)", False, False),
    ("aBlur(type=0, blur=1, blurV=4, chroma=2)", False, False),
    ("aWarp(edg)", False, True),
    ("aWarp(edg, depth=20, chroma=3)", False, True),
    ("aWarp(edg, chroma=6)", False, False),
    ("aWarp(edg, depth=-30, depthV=60, chroma=4, cplace=\"MPEG2\")", False, False),
    ("aWarp4(big, edg)", False, True),
    ("aWarp4(big, edg, depth=10, chroma=3)", False, True),
    ("aWarp4(big, edg, depth=-7, depthV=12, chroma=4, cplace=\"MPEG2\")", False, False),
    ("aWarpSharp()", False, True),
    ("aWarpSharp(cm=0)", True, False),
    ("aWarpSharp(cm=2)", False, False),
    ("aWarpSharp(bm=0)", False, False),
    ("aWarpSharp(depth=8, blurlevel=3, thresh=0.3)", False, False),
    ("aWarpSharp(original=true)", False, True),
    ("aWarpSharp(original=true, bm=0)", False, True),
    ("aWarpSharp(original=true, bm=1)", False, True),
    ("aWarpSharp(original=true, cm=0)", False, False),
    ("aWarpSharp(original=true, cm=2)", False, True),
    ("aWarpSharp(original=true, show=true)", False, False),
    ("aWarpSharp(original=true, depth=40, blurlevel=4, thresh=0.8)", False, False),
    ("aWarpSharp(original=true, depth=64, blurlevel=1, thresh=1.0, bm=0, cm=2)", False, False),
]


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def run(avs2yuv, path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    p = subprocess.run([avs2yuv, "-nstdr", "-raw", path, "-o", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None, p.stderr.decode(errors="replace").strip()[-400:]
    return p.stdout, ""


def planes(data, w, h, fmt):
    _, bits, np_, ssw, ssh = FORMATS[fmt]
    if ssw == 2:  # read back as YV24 (avs2yuv has no raw YV411)
        ssw = 0
    dt = np.uint8 if bits == 8 else np.uint16
    arr = np.frombuffer(data, dtype=dt)
    sizes = [(h, w)] + [(h >> ssh, w >> ssw)] * (np_ - 1)
    fsize = sum(a * b for a, b in sizes)
    n = arr.size // fsize
    out = []
    for i in range(n):
        f = arr[i * fsize:(i + 1) * fsize]
        off, pl = 0, []
        for (ph, pw) in sizes:
            pl.append(f[off:off + ph * pw].reshape(ph, pw).astype(np.int32))
            off += ph * pw
        out.append(pl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--avs2yuv", required=True)
    ap.add_argument("--avs-bestsource", required=True)
    ap.add_argument("--maxcpu", default="", help="SetMaxCPU() argument for the original (e.g. sse4.2 forces SSE2)")
    ap.add_argument("--opt", default="", help="opt value for the port")
    ap.add_argument("--orig-args", default="", help="extra arguments for the original (e.g. threads=1)")
    ap.add_argument("--only", default="", help="regex on 'format call'")
    ap.add_argument("--formats", default="", help="comma separated subset of formats")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    fmts = a.formats.split(",") if a.formats else MAIN_FORMATS
    cases = []
    for ci, clip in enumerate(a.clips):
        for fmt in fmts:
            for call, ignore_chroma, all_fmts in CASES:
                if fmt != "YV12" and not all_fmts:
                    continue
                if a.only and not re.search(a.only, "%s %s" % (fmt, call)):
                    continue
                cases.append((ci, clip, fmt, call, ignore_chroma))

    def one(idx_case):
        idx, (ci, clip, fmt, call, ignore_chroma) = idx_case
        conv = FORMATS[fmt][0]
        post = ".ConvertToYV24(chromaresample=\"point\")" if FORMATS[fmt][3] == 2 else ""
        # original=true reads up to two rows below the plane at the bottom (see README); a flat bottom border keeps
        # the displacement there at zero, so both read inside the plane.
        border = ORIGINAL_BORDER if "original=true" in call else 0
        if border:
            conv += "\nAddBorders(0, 0, 0, %d)" % border
        # The warp of the original reads one mask sample past the row end for the last column: counted apart.
        warps = call.startswith("aWarp") and "original=true" not in call
        port_call = call
        if a.opt:
            port_call = call[:-1] + (", " if not call.endswith("()") else "") + "opt=%s)" % a.opt
        # SetMaxCPU must run before the original's LoadPlugin, which reads the CPU flags.
        body = '%%sLoadPlugin("%s")\nLoadPlugin("%%s")\nBSVideoSource("%s")\n%s\n%s%s' % (
            fwd(a.avs_bestsource), fwd(clip), conv, EDGES if "edg" in call else "", BIG if "big" in call else "")
        maxcpu = 'SetMaxCPU("%s")\n' % a.maxcpu if a.maxcpu else ""
        orig_call = call
        if a.orig_args:
            orig_call = call[:-1] + (", " if not call.endswith("()") else "") + a.orig_args + ")"
        o, oe = run(a.avs2yuv, os.path.join(a.work, "o%05d.avs" % idx), body % (maxcpu, fwd(a.original)) + orig_call + post + "\n")
        p, pe = run(a.avs2yuv, os.path.join(a.work, "p%05d.avs" % idx), body % ("", fwd(a.plugin)) + port_call + post + "\n")
        if o is None:
            return ci, fmt, call, "skip", "original: " + oe.splitlines()[-1] if oe else "original failed"
        if p is None:
            return ci, fmt, call, "FAIL", "port: " + (pe.splitlines()[-1] if pe else "failed")
        hdr = open(clip, "rb").read(200).split(b"\n")[0].decode()
        w = int(re.search(r" W(\d+)", hdr).group(1))
        h = int(re.search(r" H(\d+)", hdr).group(1)) + border
        po, pp = planes(o, w, h, fmt), planes(p, w, h, fmt)
        if len(po) != len(pp):
            return ci, fmt, call, "FAIL", "frame count %d vs %d" % (len(po), len(pp))
        msgs = []
        lastcol = 0
        for pi in range(len(po[0])):
            if pi > 0 and ignore_chroma:
                continue
            nd, mx, first = 0, 0, None
            for fo, fp in zip(po, pp):
                d = np.abs(fo[pi] - fp[pi])
                if warps:
                    nlast = 4 if (FORMATS[fmt][3] == 2 and pi > 0) else 1  # YV411 chroma, point-upsampled
                    lastcol += int(np.count_nonzero(d[:, -nlast:]))
                    d = d[:, :-nlast]
                k = int(np.count_nonzero(d))
                if k:
                    nd += k
                    mx = max(mx, int(d.max()))
                    if first is None:
                        yy, xx = np.argwhere(d)[0]
                        first = (int(xx), int(yy))
            if nd:
                msgs.append("plane %d: %d samples differ (max %d, first at x=%d y=%d)" % (pi, nd, mx, first[0], first[1]))
        if lastcol:
            msgs.append("(last column: %d samples differ)" % lastcol)
        failed = any(not m.startswith("(") for m in msgs)
        return ci, fmt, call, "FAIL" if failed else "ok", "; ".join(msgs)

    fails = skips = 0
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for ci, fmt, call, st, msg in ex.map(one, enumerate(cases)):
            fails += st == "FAIL"
            skips += st == "skip"
            print("%-4s clip%d %-8s %s %s" % (st, ci, fmt, call, msg))
    print("%d of %d cases differ, %d skipped" % (fails, len(cases), skips))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
