#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Functional test of the VapourSynth interface (also run by CI on Windows, Linux and macOS).

- every instruction set of the CPU (opt 1..3) gives the output of the C code (opt 0), for all functions and formats
- the C output matches the MD5 digests below (computed on Windows x64), so every platform computes the same
- invalid arguments raise errors

Usage: python test_awarpsharpmt.py [path to the plugin]   (without a path: the installed wheel)
       python test_awarpsharpmt.py [path] --update          prints the digests for DIGESTS
"""
import hashlib
import math
import sys

import vapoursynth as vs

core = vs.core

FORMATS = ["GRAY8", "YUV420P8", "YUV422P10", "YUV444P16", "YUV411P8", "YUV440P8", "YUV420PS", "GRAYS"]


def pattern(w, h, fn):
    """A GRAYS clip of 3 frames with the values fn(x, y, n) (deterministic, written in Python)."""
    base = core.std.BlankClip(width=w, height=h, format=vs.GRAYS, length=3)

    def fill(n, f):
        out = f.copy()
        mv = memoryview(out[0])
        for y in range(h):
            for x in range(w):
                mv[y, x] = fn(x, y, n)
        return out

    return core.std.ModifyFrame(base, base, fill)


def source(fmt, w=204, h=116):
    # Edges in every direction (rings, stripes, a ramp), moving from frame to frame; values in 0..1.
    def luma(x, y, n):
        r = math.hypot(x - 31 - 3 * n, y - 47)
        return (0.5 + 0.5 * math.sin(r / 9.0) + 0.3 * ((x // 7) % 2) + (x + y) / 420.0) / 3.0 + 0.02 * n

    target = core.get_video_format(getattr(vs, fmt))
    lum = pattern(w, h, luma)
    if target.color_family == vs.GRAY:
        return lum.resize.Point(format=target.id)
    u = pattern(w, h, lambda x, y, n: 0.4 * math.cos((x - 13) * 0.04) * math.sin(y * 0.07 + n))
    v = pattern(w, h, lambda x, y, n: 0.4 * math.sin((x - y) * 0.05 - n))
    yuv = core.std.ShufflePlanes([lum, u, v], [0, 0, 0], vs.YUV)
    return yuv.resize.Point(format=target.id)


def cases(m, clip):
    edges = m.aBlur(m.aSobel(clip, chroma=3), chroma=3)
    big = clip.resize.Point(clip.width * 4, clip.height * 4)
    return [
        ("aWarpSharp2", lambda o: m.aWarpSharp2(clip, opt=o)),
        ("aWarpSharp2 c3 t1", lambda o: m.aWarpSharp2(clip, chroma=3, type=1, blur=5, blurV=2, depth=40, opt=o)),
        ("aWarpSharp2 c6 mpeg2", lambda o: m.aWarpSharp2(clip, chroma=6, cplace="MPEG2", depth=-30, depthV=50, opt=o)),
        ("aWarpSharp2 c0", lambda o: m.aWarpSharp2(clip, chroma=0, blur=0, opt=o)),
        ("aSobel", lambda o: m.aSobel(clip, chroma=3, thresh=200, threshC=90, opt=o)),
        ("aBlur t0", lambda o: m.aBlur(clip, type=0, blur=3, blurV=1, chroma=3, opt=o)),
        ("aBlur t1", lambda o: m.aBlur(clip, chroma=5, opt=o)),
        ("aWarp", lambda o: m.aWarp(clip, edges, depth=20, chroma=4, opt=o)),
        ("aWarp c3", lambda o: m.aWarp(clip, edges, depth=-12, chroma=3, opt=o)),
        ("aWarp4", lambda o: m.aWarp4(big, edges, depth=9, chroma=3, opt=o)),
        ("aWarp4 c4", lambda o: m.aWarp4(big, edges, chroma=4, opt=o)),
        ("aBlur t2", lambda o: m.aBlur(clip, type=2, chroma=3, opt=o)),
        ("aSobel m1", lambda o: m.aSobel(clip, mask=1, chroma=3, thresh=200, threshC=90, opt=o)),
        ("aWarpSharp2 m1", lambda o: m.aWarpSharp2(clip, mask=1, chroma=3, opt=o)),
        ("aWarpSharp2 m1 t2", lambda o: m.aWarpSharp2(clip, mask=1, type=2, depth=24, opt=o)),
        ("aWarpSharp2 t2", lambda o: m.aWarpSharp2(clip, type=2, chroma=3, depth=20, opt=o)),
        ("aWarpSharp2 t2 c4", lambda o: m.aWarpSharp2(clip, type=2, blur=4, opt=o)),
        ("aWarpSharp", lambda o: m.aWarpSharp(clip, original=False, opt=o)),
        ("original", lambda o: m.aWarpSharp(clip, original=True, opt=o)),
        ("original bm0 cm2", lambda o: m.aWarpSharp(clip, original=True, bm=0, cm=2, depth=40, blurlevel=3, opt=o)),
        ("original bm1 cm0", lambda o: m.aWarpSharp(clip, original=True, bm=1, cm=0, thresh=0.9, opt=o)),
        ("original show", lambda o: m.aWarpSharp(clip, original=True, show=True, opt=o)),
    ]


# Planes of 1 to 3 MiB and above take the other blur schedules of the engine (one pass per sweep, groups of passes).
LARGE_FORMATS = ["GRAY8", "YUV420P16", "GRAYS"]


def large_cases(m, clip):
    big = clip.resize.Point(clip.width * 12, clip.height * 12)
    return [
        ("aWarpSharp2", lambda o: m.aWarpSharp2(big, chroma=3, opt=o)),
        ("aWarpSharp2 t1", lambda o: m.aWarpSharp2(big, type=1, blur=12, depth=30, opt=o)),
        ("aBlur t0 h", lambda o: m.aBlur(big, type=0, blur=9, blurV=3, chroma=3, opt=o)),
        ("aBlur t1 v", lambda o: m.aBlur(big, type=1, blur=2, blurV=11, chroma=3, opt=o)),
        ("aBlur t2", lambda o: m.aBlur(big, type=2, blur=3, chroma=3, opt=o)),
    ]


def digest(clip):
    h = hashlib.md5()
    for f in clip.frames():
        for p in range(f.format.num_planes):
            h.update(memoryview(f[p]).tobytes())  # the visible samples in row order
    return h.hexdigest()


# (format, case) -> MD5 of the C output
DIGESTS = {
    ("GRAY8", "aWarpSharp2"): "e7a33b3ed764299e8c6d607bc4dd832d",
    ("GRAY8", "aWarpSharp2 c3 t1"): "7346655a5a01e4aaffd1a106dda4fc76",
    ("GRAY8", "aWarpSharp2 c6 mpeg2"): "b8531035695832c05e76095abb69e609",
    ("GRAY8", "aWarpSharp2 c0"): "64aa67d18c3b3ce2c019ec9d2e5194f9",
    ("GRAY8", "aSobel"): "2b75f25fc22f5b08cfd6fe2699058cf3",
    ("GRAY8", "aBlur t0"): "9ff6d404124933e3e70cb2d0b6197037",
    ("GRAY8", "aBlur t1"): "1e7f63df785cf4bdc444164b9c8eecf7",
    ("GRAY8", "aWarp"): "bd24caa3c2c58617a0f140a84e2daeaf",
    ("GRAY8", "aWarp c3"): "e8c921cca5a6d96fd61d990751f77de6",
    ("GRAY8", "aWarp4"): "1b793f4cc79d26b70338d33079d75f3e",
    ("GRAY8", "aWarp4 c4"): "e9d9f4ada727931086c77710cc2b81bd",
    ("GRAY8", "aWarpSharp"): "e7a33b3ed764299e8c6d607bc4dd832d",
    ("GRAY8", "original"): "4a011f3ba9a32dda4992281d15631216",
    ("GRAY8", "original bm0 cm2"): "ad528888ada62b1d4b210e024c42d24e",
    ("GRAY8", "original bm1 cm0"): "cfaf6e66162f046290a2b20d105a0463",
    ("GRAY8", "original show"): "da47a31366da0613aa6a486c4dab3eae",
    ("YUV420P8", "aWarpSharp2"): "c99e02ed498db7608d4888a5cc126570",
    ("YUV420P8", "aWarpSharp2 c3 t1"): "11955d9aab271253cc7d61c4e63f7664",
    ("YUV420P8", "aWarpSharp2 c6 mpeg2"): "93b3e725dfc807c923951560036bea70",
    ("YUV420P8", "aWarpSharp2 c0"): "398d78c12880df2f1ae6deac8a6d5181",
    ("YUV420P8", "aSobel"): "95cd07dfbf53f652a6c717e1e866b5d0",
    ("YUV420P8", "aBlur t0"): "67e01d3a6bb5ccffd8b4b548b3f5e28a",
    ("YUV420P8", "aBlur t1"): "ae877e2c7610afde8d9f6cd6b681916a",
    ("YUV420P8", "aWarp"): "8fd10cfcd7ead48103d34fc7b284e8bd",
    ("YUV420P8", "aWarp c3"): "0e7de34e6258d0e01627416a9931e692",
    ("YUV420P8", "aWarp4"): "08503bd90620aacdfb98fa4c01cebe69",
    ("YUV420P8", "aWarp4 c4"): "0a42133fb57f2ccaf2b3215b65f0f24d",
    ("YUV420P8", "aWarpSharp"): "c99e02ed498db7608d4888a5cc126570",
    ("YUV420P8", "original"): "c41e04a44dade708fa3054a3af71c011",
    ("YUV420P8", "original bm0 cm2"): "cc1b6cb017338b4d365c35849492354a",
    ("YUV420P8", "original bm1 cm0"): "93dcaf8ace3382c3e080847580a348aa",
    ("YUV420P8", "original show"): "4d9f3ee41e9a0670e737b98a580472c1",
    ("YUV422P10", "aWarpSharp2"): "16032d49ac16d6675820fcf238a7a58e",
    ("YUV422P10", "aWarpSharp2 c3 t1"): "193ceb20a18aa64923638f822489d88c",
    ("YUV422P10", "aWarpSharp2 c6 mpeg2"): "eed19d355b0ddbeb6cecf4c6bf8164e1",
    ("YUV422P10", "aWarpSharp2 c0"): "a5e50c6e1b5efc8a0104f02ddffd156d",
    ("YUV422P10", "aSobel"): "a7c17280f9dbfcdcb66ce94ed02f2e0a",
    ("YUV422P10", "aBlur t0"): "6db61ee5d59c721d9bd1d7298231aaf1",
    ("YUV422P10", "aBlur t1"): "fee5f31043edf546c4de4880dee467f1",
    ("YUV422P10", "aWarp"): "16fe367206b821b37dd95ffa7b4fe408",
    ("YUV422P10", "aWarp c3"): "dfed7e571362861e5e502a9514af2ac7",
    ("YUV422P10", "aWarp4"): "312235581756b93e15834435775ce7ff",
    ("YUV422P10", "aWarp4 c4"): "773d8c79439a71b98f80339da34b8cf5",
    ("YUV422P10", "aWarpSharp"): "16032d49ac16d6675820fcf238a7a58e",
    ("YUV422P10", "original"): "7d994814f9382a6c3d62f1f98a6c39bb",
    ("YUV422P10", "original bm0 cm2"): "2e004626c4d2a86b38d601a332b54779",
    ("YUV422P10", "original bm1 cm0"): "b693383fe49a48f4c5fda64bdb5ccbe7",
    ("YUV422P10", "original show"): "9074b18631e6f166edef2703827407da",
    ("YUV444P16", "aWarpSharp2"): "f0ca7c2af684cfabc909a78d89f3e2ed",
    ("YUV444P16", "aWarpSharp2 c3 t1"): "b83674b3368dd295c2a426b8331147b5",
    ("YUV444P16", "aWarpSharp2 c6 mpeg2"): "9b65f34d20f69adcdd01924ba018abc8",
    ("YUV444P16", "aWarpSharp2 c0"): "5f65fd05ee60577e99950562ad2cd907",
    ("YUV444P16", "aSobel"): "44394f443412cf3614c7839fd6ad1a2e",
    ("YUV444P16", "aBlur t0"): "5c0df483f023e7542e9d2520ca075ed0",
    ("YUV444P16", "aBlur t1"): "00c5fb17d39775bbb30af36957a7984e",
    ("YUV444P16", "aWarp"): "aac2ddef02979e60c75189b035f6fde7",
    ("YUV444P16", "aWarp c3"): "320694aca0bd341282c649f3fd284a68",
    ("YUV444P16", "aWarp4"): "cf833ba17c593ffab38087ee06564cfb",
    ("YUV444P16", "aWarp4 c4"): "905f924642c7f861846bcfabdd57b5df",
    ("YUV444P16", "aWarpSharp"): "f0ca7c2af684cfabc909a78d89f3e2ed",
    ("YUV444P16", "original"): "250f7c5b056ff61582361a37782f78a8",
    ("YUV444P16", "original bm0 cm2"): "f700e30eca856e8baf7e5ad2b90676b7",
    ("YUV444P16", "original bm1 cm0"): "6a92e6902d3d426c2180788168e37649",
    ("YUV444P16", "original show"): "8c5da9d26ccb754d0f25111c7b7aea23",
    ("YUV411P8", "aWarpSharp2"): "128c8fc2f07c727970030375790ec9fd",
    ("YUV411P8", "aWarpSharp2 c3 t1"): "75b9ea91b87717fda4dda10eebf9c4ec",
    ("YUV411P8", "aWarpSharp2 c6 mpeg2"): "89197f33441afbe65ec5bc69b0a66d44",
    ("YUV411P8", "aWarpSharp2 c0"): "398d78c12880df2f1ae6deac8a6d5181",
    ("YUV411P8", "aSobel"): "0291b35c5a4241c7bdfce9c0edb042ad",
    ("YUV411P8", "aBlur t0"): "af0e03ca33ae6ae5ff402f4e102c9136",
    ("YUV411P8", "aBlur t1"): "45fa53bb1596d310bdd8860ff47e1a1b",
    ("YUV411P8", "aWarp"): "042b08acce614f354df9f3520ff99ea1",
    ("YUV411P8", "aWarp c3"): "4f66621529f64c399a3e8694e9a78cab",
    ("YUV411P8", "aWarp4"): "b09e19e63d6a9a5f98debc4c05c9a402",
    ("YUV411P8", "aWarp4 c4"): "bc7affc626cae826488d1a8344c746cc",
    ("YUV411P8", "aWarpSharp"): "128c8fc2f07c727970030375790ec9fd",
    ("YUV411P8", "original"): "f9703c1c76fca22e42ef31f1c9c0b31a",
    ("YUV411P8", "original bm0 cm2"): "61c383566143489b7de20abebcef9cf6",
    ("YUV411P8", "original bm1 cm0"): "3a633119947208a8588a4332f0bbb54f",
    ("YUV411P8", "original show"): "e8ed8360476b7d1478a9259646c8d8ff",
    ("YUV440P8", "aWarpSharp2"): "4db104a879e9a3c18f865f85d3b92057",
    ("YUV440P8", "aWarpSharp2 c3 t1"): "2416b1bd545293fd68c254871438354b",
    ("YUV440P8", "aWarpSharp2 c6 mpeg2"): "3875184483623375e6bb4e940905d307",
    ("YUV440P8", "aWarpSharp2 c0"): "9c22d8669507486ab784c473dfad66d2",
    ("YUV440P8", "aSobel"): "892519e900374c98d465aed47aab8dd8",
    ("YUV440P8", "aBlur t0"): "f06513a28f40eb5db50e4287fd6a65d1",
    ("YUV440P8", "aBlur t1"): "6fec80d7b82e19cebe74d9693275b49e",
    ("YUV440P8", "aWarp"): "9c7d6119b3d327e3756de67b72616e33",
    ("YUV440P8", "aWarp c3"): "c8d70a2b28dbdd3ac2a4b00adb9db16a",
    ("YUV440P8", "aWarp4"): "7f767ebb31dc517a5ee68f0335fc843d",
    ("YUV440P8", "aWarp4 c4"): "71fe441aa56ac42080ac344373b660a0",
    ("YUV440P8", "aWarpSharp"): "4db104a879e9a3c18f865f85d3b92057",
    ("YUV440P8", "original"): "cf8c05943b3c01ec5a1d1d67244bb44e",
    ("YUV440P8", "original bm0 cm2"): "68cd85149e743bd60e2a3be13cd339ea",
    ("YUV440P8", "original bm1 cm0"): "aa2c7a852eb93dbc301b997339f5c57c",
    ("YUV440P8", "original show"): "f5ce466c72bae05e509e9807e8f1428b",
    ("YUV420PS", "aWarpSharp2"): "e7f7ec153d1c89e7008636a3739ff1b1",
    ("YUV420PS", "aWarpSharp2 c3 t1"): "76a1bbe51a3f32633501514e228be3f0",
    ("YUV420PS", "aWarpSharp2 c6 mpeg2"): "c7731d2b725563843fbf446ea1edc6de",
    ("YUV420PS", "aWarpSharp2 c0"): "27fec856be722aa015b39d135037565b",
    ("YUV420PS", "aSobel"): "c75ffac74b08a3585bcf17ce0c7e9b3f",
    ("YUV420PS", "aBlur t0"): "c2270af58344c9f1a9b2862ea557e536",
    ("YUV420PS", "aBlur t1"): "511acbee7b9fe2dcbbf27c8a2158a94e",
    ("YUV420PS", "aWarp"): "891a348ebb251e6bce7003b58f41fe7c",
    ("YUV420PS", "aWarp c3"): "b6b270c853984e1b4111c2f415b21e5c",
    ("YUV420PS", "aWarp4"): "487020924ea598a29c54567ee07a6e86",
    ("YUV420PS", "aWarp4 c4"): "3215b706b3da4d67456798d27e3ec9d9",
    ("YUV420PS", "aWarpSharp"): "e7f7ec153d1c89e7008636a3739ff1b1",
    ("YUV420PS", "original"): "834c6bb97955e4a151101339638661e2",
    ("YUV420PS", "original bm0 cm2"): "af85d4a84665f900b1cac34dda708fdc",
    ("YUV420PS", "original bm1 cm0"): "7602c30bccc40e0bc34311c04e4504f9",
    ("YUV420PS", "original show"): "dee5cd75fbfa09afbf7292e9a9c51e4d",
    ("GRAYS", "aWarpSharp2"): "826958578ba7c080639360b62f4d6e12",
    ("GRAYS", "aWarpSharp2 c3 t1"): "81a2d9ff345a5e8584fe6b64e5ec343b",
    ("GRAYS", "aWarpSharp2 c6 mpeg2"): "7d4b0696a9ac43fb81714f31236877c1",
    ("GRAYS", "aWarpSharp2 c0"): "59bbf930d7762f266f9197c060f97a4b",
    ("GRAYS", "aSobel"): "56664d9d0cc052a9d38195d0c78c5e94",
    ("GRAYS", "aBlur t0"): "057276d2f5e3202014109caca8a171db",
    ("GRAYS", "aBlur t1"): "9d4fa9629972df7f9654d7283851da85",
    ("GRAYS", "aWarp"): "0c0d5bf2ce72be769eaa2c394c88b1bf",
    ("GRAYS", "aWarp c3"): "10b5cb9a7e08946f446812d8e1e55acc",
    ("GRAYS", "aWarp4"): "1a44cc443462cb704de9dd6ca6ffeb89",
    ("GRAYS", "aWarp4 c4"): "b31130eb72ef3323c676320ea5dca6d2",
    ("GRAYS", "aWarpSharp"): "826958578ba7c080639360b62f4d6e12",
    ("GRAYS", "original"): "0efb9939fbd3be016f2190f7da2323ac",
    ("GRAYS", "original bm0 cm2"): "34688871fff758b5acf33db5c2daeeaa",
    ("GRAYS", "original bm1 cm0"): "0434424dcd7b8e2bc15c3ef0d11297b5",
    ("GRAYS", "original show"): "22859116e63ff5df9ac997d7a76fff80",
    ("GRAY8 x12", "aWarpSharp2"): "daa279f152995933661a89129239427e",
    ("GRAY8 x12", "aWarpSharp2 t1"): "8e7f47aadce83fa2c9853e7e5e9da7dd",
    ("GRAY8 x12", "aBlur t0 h"): "ba6fe0977abfc2d71e384939435e2047",
    ("GRAY8 x12", "aBlur t1 v"): "279c44b2d120c71129a57937ffa58f26",
    ("YUV420P16 x12", "aWarpSharp2"): "12b6ef2bdd402efd63b104cc94ed1d05",
    ("YUV420P16 x12", "aWarpSharp2 t1"): "7fdd2ba44e85e052b06d3ba7e9b2a6bc",
    ("YUV420P16 x12", "aBlur t0 h"): "ac10ba44a6c557d248b31283fc3216e9",
    ("YUV420P16 x12", "aBlur t1 v"): "822349c19c3b5f3df336286fb04ca0f4",
    ("GRAYS x12", "aWarpSharp2"): "bcf5cfaaaf1310326407af99387ce5b1",
    ("GRAYS x12", "aWarpSharp2 t1"): "7ee9315e230fce1978c1ae3faea4f427",
    ("GRAYS x12", "aBlur t0 h"): "1d9e6ccdff04ce5aa9de703673d3f116",
    ("GRAYS x12", "aBlur t1 v"): "05984039ea84f99d30ba98588f271f47",
    ("GRAY8", "aBlur t2"): "6f25263215954db25c6c11723c878681",
    ("GRAY8", "aWarpSharp2 t2"): "0ca37c02738a31690586d5b4e2dd799e",
    ("GRAY8", "aWarpSharp2 t2 c4"): "fe4d13a006a3e0e2ecd3f298f9eea68d",
    ("YUV420P8", "aBlur t2"): "0dd1633628427f79a46faa4f22c63713",
    ("YUV420P8", "aWarpSharp2 t2"): "b2540cf0703e8618cf5fc8cae9dff572",
    ("YUV420P8", "aWarpSharp2 t2 c4"): "26cfa765ba45cb87609e3a1957a39292",
    ("YUV422P10", "aBlur t2"): "b6b7f09b559323511d96559614231dee",
    ("YUV422P10", "aWarpSharp2 t2"): "cd63fa9177c716903202fc509dea5306",
    ("YUV422P10", "aWarpSharp2 t2 c4"): "6759256f318776e56f77687e0afbdacc",
    ("YUV444P16", "aBlur t2"): "e50cd51ed82e92fafd1d22e89db06f93",
    ("YUV444P16", "aWarpSharp2 t2"): "cfb431425197d60886832fb277e91426",
    ("YUV444P16", "aWarpSharp2 t2 c4"): "f37aa629776ce77abcb7ad11e3be7d6e",
    ("YUV411P8", "aBlur t2"): "2cac1a633e9ad20b745be49e5532cde6",
    ("YUV411P8", "aWarpSharp2 t2"): "c6a3276cd0cfd5e87a53a5970ead98dd",
    ("YUV411P8", "aWarpSharp2 t2 c4"): "10e92b09390227cef665d912a4964b12",
    ("YUV440P8", "aBlur t2"): "0f5e8dc1db34eeb673e2c7331fbb794f",
    ("YUV440P8", "aWarpSharp2 t2"): "0077528f07b00bf68b2065a9cdb0f812",
    ("YUV440P8", "aWarpSharp2 t2 c4"): "ee9004002b5855de5957597bf5c16df5",
    ("YUV420PS", "aBlur t2"): "c04ac3a46cb1c84b2db5054309eb986e",
    ("YUV420PS", "aWarpSharp2 t2"): "cb262b6f614385b76ca09ffd74cc268c",
    ("YUV420PS", "aWarpSharp2 t2 c4"): "f20c954c5ac738855fe6fa4c4f49e388",
    ("GRAYS", "aBlur t2"): "59d48d4059520518fddc6867153835a2",
    ("GRAYS", "aWarpSharp2 t2"): "b05d5755e04efc863e328577e359362e",
    ("GRAYS", "aWarpSharp2 t2 c4"): "33f24cd4b9462b4eb8e97d6b3cbfc3dd",
    ("GRAY8 x12", "aBlur t2"): "8af3d8a8742f4e88200ead656438a6ba",
    ("YUV420P16 x12", "aBlur t2"): "c26383aa695f69b59e07713527596c0a",
    ("GRAYS x12", "aBlur t2"): "bc9b376b81fca9f1810b17f07db7fbfc",
    ("GRAY8", "aSobel m1"): "57327a0e2f9a6bd83741e08f5c33417e",
    ("GRAY8", "aWarpSharp2 m1"): "dc361910bf18e514efe740758a5d3e85",
    ("GRAY8", "aWarpSharp2 m1 t2"): "6eeac1275972cdc42150ba1c0c717586",
    ("YUV420P8", "aSobel m1"): "8a1096c17f65a3aeee5b718bcdcb4923",
    ("YUV420P8", "aWarpSharp2 m1"): "8a142c834e06a1f84aa2895ca07a8055",
    ("YUV420P8", "aWarpSharp2 m1 t2"): "9e274a062876af418856fadc60882423",
    ("YUV422P10", "aSobel m1"): "b5022c49d12d3ac8ebaa1db0994a0303",
    ("YUV422P10", "aWarpSharp2 m1"): "fbf2bd03bc68801113a289720ebe6c4d",
    ("YUV422P10", "aWarpSharp2 m1 t2"): "cc6cb652ec6366e973b5555ee4836aed",
    ("YUV444P16", "aSobel m1"): "633820815aa3dec622b13d127a7471b9",
    ("YUV444P16", "aWarpSharp2 m1"): "07362a4462d79a672b71025b7e4b886c",
    ("YUV444P16", "aWarpSharp2 m1 t2"): "fc912f90e2eb86edda2902208c91e826",
    ("YUV411P8", "aSobel m1"): "38093c0f4114d60b6e5d9bec29470639",
    ("YUV411P8", "aWarpSharp2 m1"): "c1c9d2381d6ed050f6b460d420bcc082",
    ("YUV411P8", "aWarpSharp2 m1 t2"): "b1ab9efa60f36d924c56d7d1f2273fd3",
    ("YUV440P8", "aSobel m1"): "bac787c50e499918da7475f52b88c180",
    ("YUV440P8", "aWarpSharp2 m1"): "fbd6729c1944ff107fa3151176c25a01",
    ("YUV440P8", "aWarpSharp2 m1 t2"): "84e91e607b05000379ab8b09bfd70048",
    ("YUV420PS", "aSobel m1"): "2e0001eb99f36fc4d3a2e83321a6bc0c",
    ("YUV420PS", "aWarpSharp2 m1"): "45a7115549d90258adef4f7ffbd04389",
    ("YUV420PS", "aWarpSharp2 m1 t2"): "d6c70b99a5843ebae74023c904139f7b",
    ("GRAYS", "aSobel m1"): "1e9622415b378a053f098f877e976095",
    ("GRAYS", "aWarpSharp2 m1"): "09802d81133ca63791450b59170f413e",
    ("GRAYS", "aWarpSharp2 m1 t2"): "d2e02ae4305728da71b0dc55abec3b92",
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    update = "--update" in sys.argv
    if args:
        core.std.LoadPlugin(args[0])
    m = core.awarpsharpmt
    failures = 0
    available = [0]
    for o in (1, 2, 3):
        try:
            m.aSobel(core.std.BlankClip(format=vs.GRAY8), opt=o)
            available.append(o)
        except vs.Error:
            pass
    print("instruction sets:", available)
    runs = [(fmt, fmt, cases) for fmt in FORMATS] + [(fmt, fmt + " x12", large_cases) for fmt in LARGE_FORMATS]
    for fmt, key, make in runs:
        clip = source(fmt)
        for name, fn in make(m, clip):
            ref = digest(fn(0))
            if update:
                print('    ("%s", "%s"): "%s",' % (key, name, ref))
                continue
            msg = []
            want = DIGESTS.get((key, name))
            if want is None:
                msg.append("no digest")
            elif want != ref:
                msg.append("C output differs from the digest")
            for o in available[1:]:
                if digest(fn(o)) != ref:
                    msg.append("opt=%d differs from C" % o)
            failures += bool(msg and msg != ["no digest"])
            print("%-4s %-14s %-22s %s" % ("FAIL" if msg and msg != ["no digest"] else "ok", key, name, "; ".join(msg)))
    if update:
        return 0

    # mask=1 + type=2 is the edge mask of aWarpSharp(original=true, bm=0): luma of show=true must match
    for fmt in FORMATS:
        clip = source(fmt)
        for t, level in ((0.5, 2), (0.3, 3), (0.75, 4)):
            ours = m.aBlur(m.aSobel(clip, thresh=int(t * 256) & 255, mask=1), type=2, blur=3 * level)
            orig = m.aWarpSharp(clip, original=True, bm=0, show=True, thresh=t, blurlevel=level)
            same = digest(ours.std.ShufflePlanes(0, vs.GRAY)) == digest(orig.std.ShufflePlanes(0, vs.GRAY))
            failures += not same
            print("%-4s %-14s %s" % ("ok" if same else "FAIL", fmt, "original mask, thresh %.2f blurlevel %d" % (t, level)))

    # argument checks
    c8 = core.std.BlankClip(format=vs.YUV420P8)
    bad = [
        lambda: m.aWarpSharp2(c8, thresh=256),
        lambda: m.aWarpSharp2(c8, depth=128),
        lambda: m.aWarpSharp2(c8, chroma=7),
        lambda: m.aWarpSharp2(c8, type=3),
        lambda: m.aSobel(c8, mask=2),
        lambda: m.aWarpSharp2(c8, mask=-1),
        lambda: m.aBlur(c8, type=2, blurV=2),
        lambda: m.aWarpSharp2(c8, type=2, blurC=2, blurVC=1),
        lambda: m.aBlur(c8, blur=-1),
        lambda: m.aWarp(c8, c8.resize.Point(320, 240)),
        lambda: m.aWarp4(c8, c8),
        lambda: m.aWarpSharp(c8, original=True, bm=3),
        lambda: m.aWarpSharp(c8, original=True, depth=65.0),
        lambda: m.aWarpSharp(c8, original=True, blurlevel=5),
        lambda: m.aWarpSharp2(core.std.BlankClip(format=vs.RGB24)),
        lambda: m.aWarpSharp2(core.std.BlankClip(format=vs.YUV420PH)),
        lambda: m.aSobel(c8, opt=9),
    ]
    for i, f in enumerate(bad):
        try:
            f().get_frame(0)
            print("FAIL argument check %d: no error" % i)
            failures += 1
        except vs.Error:
            pass
    print("%d failures" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
