#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compares the AviSynth+ and the VapourSynth interface of the same plugin binary (Windows).

Both read the same y4m clips with BestSource; every output frame must be identical. Formats come from the clips
themselves (make them with ffmpeg: yuv420p, yuv422p10le, yuv444p16le, gray, yuv411p, ...), so no host conversion is
involved. Run with a VapourSynth Python:

  python -I compare_hosts.py --plugin build/aWarpSharpMT.dll --avs2yuv avs2yuv64.exe --avs-bestsource BestSource.dll
         --vs-bestsource libbestsource.dll --work DIR CLIP.y4m...
"""
import argparse
import hashlib
import os
import subprocess
import sys

import vapoursynth as vs

core = vs.core

# (AviSynth call, VapourSynth call on c / e / big); e = aBlur(aSobel(c, chroma=3), chroma=3), big = CLIP.x4.y4m next
# to CLIP.y4m (four times the size, e.g. ffmpeg -vf scale=iw*4:ih*4:flags=neighbor), aWarp4 is skipped without it
CASES = [
    ("aWarpSharp2()", "m.aWarpSharp2(c)"),
    ("aWarpSharp2(chroma=3, type=1, blur=5, blurV=2, depth=40)", "m.aWarpSharp2(c, chroma=3, type=1, blur=5, blurV=2, depth=40)"),
    ("aWarpSharp2(chroma=6, cplace=\"MPEG2\", depth=-30, depthV=50)", "m.aWarpSharp2(c, chroma=6, cplace='MPEG2', depth=-30, depthV=50)"),
    ("aSobel(chroma=3, thresh=200, threshC=90)", "m.aSobel(c, chroma=3, thresh=200, threshC=90)"),
    ("aBlur(type=0, blur=3, blurV=1, chroma=3)", "m.aBlur(c, type=0, blur=3, blurV=1, chroma=3)"),
    ("aBlur(type=2, chroma=3)", "m.aBlur(c, type=2, chroma=3)"),
    ("aWarpSharp2(type=2, chroma=3, depth=20)", "m.aWarpSharp2(c, type=2, chroma=3, depth=20)"),
    ("aWarp(e, depth=20)", "m.aWarp(c, e, depth=20)"),
    ("aWarp4(big, e, depth=9, chroma=3)", "m.aWarp4(big, e, depth=9, chroma=3)"),
    ("aWarpSharp(original=false, depth=8, blurlevel=3, thresh=0.3)", "m.aWarpSharp(c, original=False, depth=8, blurlevel=3, thresh=0.3)"),
    ("aWarpSharp(original=true)", "m.aWarpSharp(c, original=True)"),
    ("aWarpSharp(original=true, bm=0, cm=2)", "m.aWarpSharp(c, original=True, bm=0, cm=2)"),
    ("aWarpSharp(original=true, bm=1, show=true)", "m.aWarpSharp(c, original=True, bm=1, show=True)"),
]


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def frame_sizes(clip):
    f = clip.format
    sizes = []
    for p in range(f.num_planes):
        w = clip.width >> (f.subsampling_w if p else 0)
        h = clip.height >> (f.subsampling_h if p else 0)
        sizes.append(w * h * f.bytes_per_sample)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--avs2yuv", required=True)
    ap.add_argument("--avs-bestsource", required=True)
    ap.add_argument("--vs-bestsource", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    core.std.LoadPlugin(a.plugin)
    if not hasattr(core, "bs"):
        core.std.LoadPlugin(a.vs_bestsource)
    m = core.awarpsharpmt
    failures = total = 0
    for ci, clip_path in enumerate(a.clips):
        c = core.bs.VideoSource(clip_path)
        e = m.aBlur(m.aSobel(c, chroma=3), chroma=3)
        big_path = clip_path[:-len(".y4m")] + ".x4.y4m"
        big = core.bs.VideoSource(big_path) if os.path.exists(big_path) else None
        for k, (avs_call, vs_call) in enumerate(CASES):
            if "big" in vs_call and big is None:
                continue
            total += 1
            out = eval(vs_call)
            sizes = frame_sizes(out)
            vs_md5 = []
            for f in out.frames():
                h = hashlib.md5()
                for p in range(out.format.num_planes):
                    h.update(memoryview(f[p]).tobytes())
                vs_md5.append(h.hexdigest())
            script = os.path.join(a.work, "h%02d_%02d.avs" % (ci, k))
            with open(script, "w", encoding="utf-8") as fh:
                fh.write('LoadPlugin("%s")\nLoadPlugin("%s")\nBSVideoSource("%s")\n' % (
                    fwd(a.avs_bestsource), fwd(a.plugin), fwd(clip_path)))
                fh.write("e = last.aSobel(chroma=3).aBlur(chroma=3)\n")
                if big is not None:
                    fh.write('big = BSVideoSource("%s")\n' % fwd(big_path))
                fh.write(avs_call + "\n")
            p = subprocess.run([a.avs2yuv, "-nstdr", "-raw", script, "-o", "-"], capture_output=True)
            fs = sum(sizes)
            data = p.stdout
            if p.returncode or not data or len(data) % fs:
                print("FAIL clip%d %s: AviSynth+ %s" % (ci, avs_call, p.stderr.decode(errors="replace")[-300:]))
                failures += 1
                continue
            avs_md5 = [hashlib.md5(data[i:i + fs]).hexdigest() for i in range(0, len(data), fs)]
            diff = [i for i, (x, y) in enumerate(zip(avs_md5, vs_md5)) if x != y]
            ok = not diff and len(avs_md5) == len(vs_md5)
            failures += not ok
            print("%s clip%d %-12s %s%s" % ("ok  " if ok else "FAIL", ci, out.format.name, avs_call,
                                            "" if ok else " frames differ: %s" % diff[:8]))
    print("%d of %d cases differ" % (failures, total))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
