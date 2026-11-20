#!/usr/bin/env python3
"""
hdr10plus_inject.py — Copy HDR10+ dynamic metadata from one HEVC video file into another
version of the same content, with automatic frame alignment.

Intended for use with video files you legally own.

Pipeline
  1. Extract HDR10+ metadata JSON from the source with hdr10plus_tool
  2. Extract the target HEVC video stream
  3. Align the metadata to the target's frame count
       --direct       1:1 map, pad/trim the tail   (endpoints known to match)
       (default)      constant offset, detected from the head via perceptual hash
       --force-align  per-frame alignment with re-acquisition after drift
  4. Inject metadata losslessly (hdr10plus_tool inject)
  5. Re-mux with mkvmerge, keeping all audio/subtitle/chapter tracks

Letterboxed vs cropped sources are normalised with ffmpeg cropdetect before
hashing. On macOS, VideoToolbox decode is used when available, with fallback.

Requirements: python3 + numpy, ffmpeg, ffprobe, mkvmerge, hdr10plus_tool
  macOS:  brew install ffmpeg mkvtoolnix hdr10plus_tool && pip3 install numpy

Usage
  python3 hdr10plus_inject.py                       # prompts for both files
  python3 hdr10plus_inject.py SOURCE.mkv TARGET.mkv [--direct | --force-align]

Original files are never modified. Output: <target>.HDR10plus.mkv
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip3 install numpy")

HASH_W, HASH_H = 17, 16          # dHash grid -> 16*16 = 256 bits
HASH_BITS = 256
MATCH_BITS = 40                  # max Hamming distance counted as a match
HEAD_FRAMES = 3000               # frames hashed for offset detection
MAX_OFFSET = 1500                # search window for constant offset (frames)
ALIGN_K = 48                     # per-frame search radius
REACQUIRE_AFTER = 24             # misses before widening the search
REACQUIRE_K = 2000               # widened search radius
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ───────────────────────────── helpers ─────────────────────────────

class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0 = max(1, total), label, time.time()

    def update(self, n):
        el = time.time() - self.t0
        fps = n / el if el > 0 else 0
        eta = (self.total - n) / fps if fps > 0 else 0
        pct = min(1.0, n / self.total)
        bar = "#" * int(pct * 30)
        sys.stderr.write(f"\r  {self.label:<10} [{bar:<30}] {pct*100:5.1f}% "
                         f"{fps:7.1f} fps  ETA {int(eta)//60:02d}:{int(eta)%60:02d}")
        sys.stderr.flush()

    def done(self):
        self.update(self.total)
        sys.stderr.write("\n")


def die(msg):
    sys.exit(f"\n✗ {msg}")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        die(f"Command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def check_tools():
    missing = [t for t in ("ffmpeg", "ffprobe", "mkvmerge", "hdr10plus_tool")
               if not shutil.which(t)]
    if missing:
        die("Missing tools: " + ", ".join(missing) +
            "\n  macOS: brew install ffmpeg mkvtoolnix hdr10plus_tool")


def ask_path(prompt):
    while True:
        p = input(prompt).strip().strip("'\"").replace("\\ ", " ")
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            return p
        print("  File not found, try again (you can drag the file into the terminal).")


def probe_video(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate,width,height",
               "-of", "json", path])
    s = json.loads(out)["streams"][0]
    return s["r_frame_rate"], int(s["width"]), int(s["height"])


def count_frames(path):
    print("  counting target frames (packet count)...")
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_packets", "-show_entries", "stream=nb_read_packets",
               "-of", "csv=p=0", path])
    return int(out.strip().split(",")[0])


def detect_crop(path):
    """Return an ffmpeg crop filter string, sampling a few points in the film."""
    crops = []
    for ss in (300, 1200, 2400):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(ss), "-i", path,
                            "-t", "8", "-vf", "cropdetect=24:2:0", "-an", "-sn",
                            "-f", "null", "-"], capture_output=True, text=True)
        crops += re.findall(r"crop=(\d+:\d+:\d+:\d+)", r.stderr)
    if not crops:
        return None
    best = max(set(crops), key=crops.count)
    return f"crop={best}"


def has_videotoolbox():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                       capture_output=True, text=True)
    return "videotoolbox" in r.stdout


# ───────────────────────────── stages ─────────────────────────────

def extract_metadata(src, out_json):
    print("[1/5] Extracting HDR10+ metadata from source...")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0",
                           "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb",
                           "-f", "hevc", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    r = subprocess.run(["hdr10plus_tool", "extract", "-o", out_json, "-"],
                       stdin=ff.stdout, capture_output=True, text=True)
    ff.stdout.close()
    ff.wait()
    if r.returncode != 0 or not os.path.isfile(out_json):
        die("hdr10plus_tool extract failed — does the source actually contain HDR10+?\n"
            + r.stderr[-1500:])
    with open(out_json) as f:
        meta = json.load(f)
    n = len(meta.get("SceneInfo", []))
    if n == 0:
        die("No HDR10+ frames found in source.")
    print(f"  {n} frames of HDR10+ metadata")
    return meta


def extract_hevc(target, out_hevc):
    print("[2/5] Extracting target HEVC stream...")
    run(["ffmpeg", "-v", "error", "-y", "-i", target, "-map", "0:v:0", "-c:v", "copy",
         "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", out_hevc])


def hash_frames(path, crop, limit=None, total_hint=None, label="hashing"):
    """Decode frames to tiny grayscale and return an (N, 32) uint8 dHash array."""
    vf = ",".join(f for f in (crop, f"scale={HASH_W}:{HASH_H}:flags=area",
                               "format=gray") if f)
    fb = HASH_W * HASH_H

    def attempt(hw):
        cmd = ["ffmpeg", "-v", "quiet"]
        if hw:
            cmd += ["-hwaccel", "videotoolbox"]
        cmd += ["-i", path, "-map", "0:v:0", "-an", "-sn", "-vf", vf]
        if limit:
            cmd += ["-frames:v", str(limit)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=fb * 4096)
        prog = Progress(limit or total_hint or 1, label)
        hashes, n = [], 0
        while True:
            buf = p.stdout.read(fb * 1024)
            if not buf:
                break
            k = len(buf) // fb
            a = np.frombuffer(buf[:k * fb], np.uint8).reshape(k, HASH_H, HASH_W)
            bits = (a[:, :, 1:] > a[:, :, :-1]).reshape(k, -1)
            hashes.append(np.packbits(bits, axis=1))
            n += k
            prog.update(n)
        p.wait()
        prog.done()
        return np.concatenate(hashes) if hashes else np.zeros((0, 32), np.uint8)

    hw = has_videotoolbox()
    h = attempt(hw)
    if len(h) == 0 and hw:
        print("  VideoToolbox decode failed, falling back to software...")
        h = attempt(False)
    if len(h) == 0:
        die(f"Could not decode frames from {path}")
    return h


def hamming(a, b):
    return _POPCOUNT[np.bitwise_xor(a, b)].sum(axis=-1)


def find_offset(src_h, rem_h):
    """Return s such that target frame i ≈ source frame i - s, and mean distance."""
    best = (None, 1e9)
    n = min(len(src_h), len(rem_h)) - MAX_OFFSET
    if n < 200:
        n = min(len(src_h), len(rem_h)) // 2
    for s in range(-MAX_OFFSET, MAX_OFFSET + 1):
        r0, s0 = max(0, s), max(0, -s)
        m = min(n, len(rem_h) - r0, len(src_h) - s0)
        if m < 200:
            continue
        d = hamming(src_h[s0:s0 + m], rem_h[r0:r0 + m]).mean()
        if d < best[1]:
            best = (s, d)
    return best


def align_full(src_h, rem_h, seed_s):
    """Per-frame greedy alignment with re-acquisition. Returns (map, matched)."""
    R, W = len(rem_h), len(src_h)
    fmap = np.full(R, -1, dtype=np.int64)
    e, misses, matched = -seed_s, 0, 0
    prog = Progress(R, "aligning")
    for i in range(R):
        k = REACQUIRE_K if misses >= REACQUIRE_AFTER else ALIGN_K
        lo, hi = max(0, e - k), min(W, e + k + 1)
        hit = False
        if lo < hi:
            d = hamming(src_h[lo:hi], rem_h[i])
            j = int(np.argmin(d))
            if int(d[j]) <= MATCH_BITS:
                fmap[i] = lo + j
                e, misses, hit = lo + j + 1, 0, True
                matched += 1
        if not hit:
            e += 1
            misses += 1
        if (i & 1023) == 0:
            prog.update(i)
    prog.done()
    return fmap, matched


def fill_map(fmap, n_src):
    """Fill unmatched (-1) entries by interpolating from neighbours, clamp to range."""
    fmap = fmap.copy()
    idx = np.where(fmap >= 0)[0]
    if len(idx) == 0:
        return np.clip(np.arange(len(fmap)), 0, n_src - 1)
    for i in np.where(fmap < 0)[0]:
        p = idx[np.searchsorted(idx, i) - 1] if np.searchsorted(idx, i) > 0 else None
        q = idx[np.searchsorted(idx, i)] if np.searchsorted(idx, i) < len(idx) else None
        if p is not None:
            fmap[i] = fmap[p] + (i - p)
        else:
            fmap[i] = fmap[q] - (q - i)
    return np.clip(fmap, 0, n_src - 1)


def rebuild_json(src, fmap):
    """Build new metadata JSON keeping all top-level keys (ToolInfo, JSONInfo...)."""
    si = src["SceneInfo"]
    new_si, first_frames, scene_counts = [], [], []
    prev_scene, scene_id, in_scene = None, -1, 0
    for i, j in enumerate(fmap):
        fr = dict(si[int(j)])
        sid = fr.get("SceneId")
        if sid != prev_scene or i == 0:
            scene_id += 1
            first_frames.append(i)
            scene_counts.append(0)
            in_scene, prev_scene = 0, sid
        fr["SceneId"] = scene_id
        fr["SceneFrameIndex"] = in_scene
        fr["SequenceFrameIndex"] = i
        new_si.append(fr)
        scene_counts[-1] += 1
        in_scene += 1
    out = {k: v for k, v in src.items() if k != "SceneInfo"}
    out["SceneInfo"] = new_si
    out["SceneInfoSummary"] = {"SceneFirstFrameIndex": first_frames,
                               "SceneFrameNumbers": scene_counts}
    return out


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Copy HDR10+ metadata into another HEVC file.")
    ap.add_argument("source", nargs="?", help="file that contains HDR10+ metadata")
    ap.add_argument("target", nargs="?", help="target file to add HDR10+ metadata to")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--direct", action="store_true",
                   help="1:1 frame map (use when start/end are known to line up)")
    g.add_argument("--force-align", action="store_true",
                   help="full per-frame alignment (slow, decodes both files)")
    ap.add_argument("-o", "--output", help="output MKV path")
    ap.add_argument("--keep-temp", action="store_true", help="keep temp files")
    a = ap.parse_args()

    check_tools()
    src = a.source or ask_path("HDR10+ source file: ")
    rem = a.target or ask_path("Target file to add HDR10+ to: ")
    out = a.output or os.path.splitext(rem)[0] + ".HDR10plus.mkv"
    if os.path.abspath(out) in (os.path.abspath(src), os.path.abspath(rem)):
        die("Output path would overwrite an input.")

    tmp = tempfile.mkdtemp(prefix="hdr10plus_",
                           dir=os.path.dirname(os.path.abspath(out)))
    try:
        meta = extract_metadata(src, os.path.join(tmp, "source.json"))
        n_src = len(meta["SceneInfo"])
        rem_hevc = os.path.join(tmp, "target.hevc")
        extract_hevc(rem, rem_hevc)
        n_rem = count_frames(rem)
        fps, _, _ = probe_video(rem)
        print(f"  source {n_src} frames, target {n_rem} frames "
              f"(diff {n_rem - n_src:+d}), {fps} fps")

        print("[3/5] Aligning metadata...")
        if a.direct or (n_src == n_rem and not a.force_align):
            print("  mode: direct 1:1")
            fmap = np.clip(np.arange(n_rem), 0, n_src - 1)
        else:
            c_src, c_rem = detect_crop(src), detect_crop(rem)
            print(f"  crop source={c_src}  target={c_rem}")
            if a.force_align:
                print("  mode: full per-frame alignment")
                sh = hash_frames(src, c_src, total_hint=n_src, label="src hash")
                rh = hash_frames(rem, c_rem, total_hint=n_rem, label="tgt hash")
                s, d = find_offset(sh[:HEAD_FRAMES], rh[:HEAD_FRAMES])
                print(f"  seed offset {s}, head distance {d:.1f}/{HASH_BITS}")
                fmap, matched = align_full(sh, rh, s)
                rate = matched / n_rem * 100
                print(f"  matched {matched}/{n_rem} frames ({rate:.1f}%)")
                if rate < 60:
                    die("Alignment too poor — likely a different cut or master. "
                        "Not injecting. Try --direct if you know the endpoints match.")
                fmap = fill_map(fmap, n_src)
            else:
                print("  mode: constant offset (head detection)")
                sh = hash_frames(src, c_src, limit=HEAD_FRAMES, label="src head")
                rh = hash_frames(rem, c_rem, limit=HEAD_FRAMES, label="tgt head")
                s, d = find_offset(sh, rh)
                print(f"  offset {s} frames, head distance {d:.1f}/{HASH_BITS}")
                if s is None or d > MATCH_BITS:
                    die("Could not find a reliable offset. Not injecting.\n"
                        "  Use --direct if start/end align, or --force-align.")
                fmap = np.clip(np.arange(n_rem) - s, 0, n_src - 1)

        new = rebuild_json(meta, fmap)
        new_json = os.path.join(tmp, "aligned.json")
        with open(new_json, "w") as f:
            json.dump(new, f)

        print("[4/5] Injecting HDR10+ into target HEVC...")
        inj = os.path.join(tmp, "injected.hevc")
        run(["hdr10plus_tool", "inject", "-i", rem_hevc, "-j", new_json, "-o", inj])

        print("[5/5] Re-muxing with mkvmerge...")
        r = subprocess.run(["mkvmerge", "-o", out,
                            "--default-duration", f"0:{fps}p", inj,
                            "-D", rem, "--track-order", "0:0"],
                           capture_output=True, text=True)
        if r.returncode >= 2:
            die("mkvmerge failed:\n" + r.stdout[-2000:] + r.stderr[-1000:])
        print(f"\n✓ Done: {out}")
        print("  Verify with: mediainfo \"%s\" | grep -Ei 'HDR|Dolby|2094'" % out)
        print("  Expect 'SMPTE ST 2094 App 4' (HDR10+). If the target had "
              "Dolby Vision, check it is still listed.")
    finally:
        if a.keep_temp:
            print(f"  temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
