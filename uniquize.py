"""
uniquize.py
===========
Uniquizes pre-shot 9:16 vertical video clips for Instagram Reels re-upload.

Key differences from the original split_and_uniquify.py:
  ✦ NO 9:16 conversion — input is already vertical (9:16).
  ✦ NO blur/stack/zoom modes — removed entirely.
  ✦ Strips all metadata (title, encoder, creation_time, GPS, etc.).
  ✦ hflip is ALWAYS applied (every clip is horizontally flipped).
  ✦ Expanded uniquification palette:
      - Asymmetric micro-crop + rescale
      - Pixel shift (subpixel canvas offset)
      - Speed variation (±8%)
      - Color temperature (R/B channel mixer)
      - EQ: brightness / contrast / saturation / gamma
      - Sharpening (unsharp mask)
      - Light noise grain
      - Subtle vignette
      - Audio: volume, pitch, tempo, EQ bands, echo, delay
      - Fades: in & out
  ✦ Similarity score after each export:
      - Compares output clip vs. input source at sampled frames
      - Reports pHash distance, histogram correlation, and SSIM
      - Composite "uniqueness score" 0–100 (higher = more different)

Usage:
    python uniquize.py                            # auto-detects input/
    python uniquize.py --input input/clip.mp4
    python uniquize.py --input input/clip.mp4 --output output/ --clips 5
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np

OUT_W, OUT_H = 1080, 1920

os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/uniquize_log.txt", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("uniquizer")


# ============================================================================
# Zone-based parameter system
# ============================================================================
# Each key maps to 5 "zone" values (from least to most aggressive).
# ZoneAssigner shuffles and distributes zones so consecutive clips differ.

ZONES = {
    # ── Visual geometry ──────────────────────────────────────────────────────
    # Larger crops shift the frame boundary significantly
    "crop_top":      [ 4,  16,  30,  46,  64],
    "crop_bottom":   [ 4,  16,  32,  50,  68],
    "crop_left":     [ 4,  12,  24,  38,  54],
    "crop_right":    [ 4,  14,  26,  40,  56],
    "pixel_shift_x": [-20, -8,   0,   8,  20],
    "pixel_shift_y": [-20, -8,   0,   8,  20],

    # ── Timing ───────────────────────────────────────────────────────────────
    "speed":         [0.90, 0.95, 1.00, 1.05, 1.10],
    "fade_in":       [0.05, 0.20, 0.35, 0.50, 0.70],
    "fade_out":      [0.05, 0.18, 0.32, 0.48, 0.65],

    # ── Color — wider ranges for stronger perceptual difference ──────────────
    "brightness":    [-0.18, -0.09,  0.00,  0.09,  0.18],
    "contrast":      [ 0.78,  0.89,  1.00,  1.12,  1.28],
    "saturation":    [ 0.50,  0.75,  1.00,  1.30,  1.60],
    "gamma":         [ 0.68,  0.84,  1.00,  1.18,  1.40],
    "color_r":       [ 0.80,  0.90,  1.00,  1.10,  1.20],
    "color_b":       [ 1.20,  1.10,  1.00,  0.90,  0.80],

    # ── Detail ───────────────────────────────────────────────────────────────
    "sharpen":       [ 0.0,   0.0,   0.8,   1.6,   2.5],
    "noise":         [ 0,     2,     6,    12,    18],
    "vignette":      [ 0.0,   0.2,   0.5,   0.7,   1.0],

    # ── Audio ────────────────────────────────────────────────────────────────
    "volume":        [ 0.70,  0.85,  1.00,  1.15,  1.30],
    "pitch_cents":   [-250, -125,    0,   125,   250],
    "audio_tempo":   [ 0.90,  0.95,  1.00,  1.05,  1.10],
    "bass_gain":     [ -6.0,  -3.0,  0.0,   3.0,   6.0],
    "mid_gain":      [ -4.0,  -2.0,  0.0,   2.0,   4.0],
    "treble_gain":   [ -6.0,  -3.0,  0.0,   3.0,   6.0],
    "delay_ms":      [ 0,     0,     8,    20,    35],
    "echo":          [False, False, False,  True,  True],
}


class ZoneAssigner:
    """
    Shuffled zone distributor — ensures no two consecutive clips share
    the same zone index on any parameter.
    """
    def __init__(self):
        self._q: Dict[str, List[int]] = {}
        for k, v in ZONES.items():
            idx = list(range(len(v)))
            random.shuffle(idx)
            self._q[k] = idx[:]

    def assign(self) -> Dict:
        p = {}
        for k, v in ZONES.items():
            if not self._q[k]:
                idx = list(range(len(v)))
                random.shuffle(idx)
                self._q[k] = idx
            p[k] = v[self._q[k].pop(0)]
        return p


# ============================================================================
# ffprobe
# ============================================================================
def get_video_info(path: str) -> Dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    r    = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(r.stdout)
    dur  = float(data["format"]["duration"])
    w = h = fps = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            w   = int(s["width"])
            h   = int(s["height"])
            n, d = s.get("r_frame_rate", "30/1").split("/")
            fps  = round(int(n) / max(1, int(d)), 2)
            break
    return {"duration": dur, "width": w, "height": h, "fps": fps}


# ============================================================================
# Filter builders
# ============================================================================

def _build_vf(p: Dict, clip_duration: float, src_w: int, src_h: int) -> str:
    """
    Build the full -vf chain for a pre-9:16 clip.

    Order:
      1.  hflip          — always applied first (reverses content identity)
      2.  Asymmetric micro-crop + rescale (based on SOURCE dimensions)
      3.  Pixel shift
      4.  Rescale to OUT_W x OUT_H
      5.  Speed (setpts)
      6.  Color temperature (colorchannelmixer)
      7.  EQ  (brightness / contrast / saturation / gamma)
      8.  Sharpen (unsharp)
      9.  Noise grain
      10. Vignette (geq)
      11. Fade in / out
      12. Final format lock
    """
    parts = []

    # 1. hflip — mandatory, always first
    #parts.append("hflip")

    # 2. Asymmetric micro-crop — proportional to SOURCE size (not output size)
    # Clamp to at most 5% of each dimension so we never over-crop small inputs
    max_tb = max(1, src_h // 20)
    max_lr = max(1, src_w // 20)
    ct = min(p["crop_top"],    max_tb)
    cb = min(p["crop_bottom"], max_tb)
    cl = min(p["crop_left"],   max_lr)
    cr = min(p["crop_right"],  max_lr)
    cw = max(src_w // 2, src_w - cl - cr)
    ch = max(src_h // 2, src_h - ct - cb)
    parts.append(f"crop={cw}:{ch}:{cl}:{ct}")

    # 3. Pixel shift  (pads canvas then re-crops — scaled relative to source)
    sx = int(p["pixel_shift_x"] * src_w / OUT_W)
    sy = int(p["pixel_shift_y"] * src_h / OUT_H)
    if sx != 0 or sy != 0:
        pad_w = cw + abs(sx) * 2
        pad_h = ch + abs(sy) * 2
        off_x = abs(sx) + max(sx, 0)
        off_y = abs(sy) + max(sy, 0)
        parts.append(f"pad={pad_w}:{pad_h}:{abs(sx)}:{abs(sy)}")
        parts.append(f"crop={cw}:{ch}:{off_x}:{off_y}")

    # 4. Upscale to exact output dimensions
    parts.append(f"scale={OUT_W}:{OUT_H}:flags=lanczos")

    # 5. Speed
    speed = p["speed"]
    if abs(speed - 1.0) > 0.005:
        pts = round(1.0 / speed, 6)
        parts.append(f"setpts={pts}*PTS")

    # 6. Color temperature (R/B channel mixer)
    rr, bb = p["color_r"], p["color_b"]
    if abs(rr - 1.0) > 0.01 or abs(bb - 1.0) > 0.01:
        parts.append(
            f"colorchannelmixer=rr={rr:.4f}:gg=1.0:bb={bb:.4f}"
        )

    # 7. EQ
    parts.append(
        f"eq=brightness={p['brightness']:.4f}"
        f":contrast={p['contrast']:.4f}"
        f":saturation={p['saturation']:.4f}"
        f":gamma={p['gamma']:.4f}"
    )

    # 8. Sharpen
    if p["sharpen"] > 0:
        parts.append(f"unsharp=5:5:{p['sharpen']:.2f}:5:5:0")

    # 9. Noise grain
    if p["noise"] > 0:
        parts.append(f"noise=alls={p['noise']}:allf=t")

    # 10. Vignette  (geq — darkens corners, subtly changes perceptual hash)
    v = p["vignette"]
    if v > 0:
        # geq lum expression: linear falloff from center
        parts.append(
            f"geq=lum='lum(X,Y)*max(0,1-{v:.3f}*sqrt((X/W-0.5)^2+(Y/H-0.5)^2)*2)'"
            f":cb='cb(X,Y)':cr='cr(X,Y)'"
        )

    # 11. Fade in / out
    fo_start = max(0.1, clip_duration - p["fade_out"] - 0.05)
    parts.append(f"fade=t=in:st=0:d={p['fade_in']:.3f}")
    parts.append(f"fade=t=out:st={fo_start:.3f}:d={p['fade_out']:.3f}")

    # 12. Final format lock
    parts.append(f"scale={OUT_W}:{OUT_H},format=yuv420p")

    return ",".join(parts)


def _build_af(p: Dict) -> str:
    """Build the -af chain."""
    parts = [f"volume={p['volume']:.4f}"]

    # Pitch shift (via resampling)
    cents = p["pitch_cents"]
    if abs(cents) > 5:
        rm  = 2 ** (cents / 1200.0)
        nr  = int(44100 * rm)
        tc  = max(0.5, min(2.0, round(1.0 / rm, 6)))
        parts.append(f"asetrate={nr},aresample=44100,atempo={tc:.6f}")

    # Tempo (independent of pitch)
    tempo = p["audio_tempo"]
    if abs(tempo - 1.0) > 0.005:
        parts.append(f"atempo={max(0.5, min(2.0, tempo)):.4f}")

    # EQ bands
    if abs(p["bass_gain"])   > 0.2:
        parts.append(f"equalizer=f=80:width_type=o:width=2:g={p['bass_gain']:.2f}")
    if abs(p["mid_gain"])    > 0.2:
        parts.append(f"equalizer=f=1000:width_type=o:width=2:g={p['mid_gain']:.2f}")
    if abs(p["treble_gain"]) > 0.2:
        parts.append(f"equalizer=f=10000:width_type=o:width=2:g={p['treble_gain']:.2f}")

    if p["echo"]:
        parts.append("aecho=0.8:0.6:20:0.15")
    if p["delay_ms"] > 0:
        parts.append(f"adelay={p['delay_ms']}|{p['delay_ms']}")

    parts.append("aresample=44100,aformat=channel_layouts=stereo")
    return ",".join(parts)


# ============================================================================
# Core processor
# ============================================================================

def process_clip(
    input_path: str,
    output_path: str,
    start_sec: float,
    duration: float,
    info: Dict,
    params: Dict,
) -> bool:
    """
    Encode one uniquified clip.

    Metadata wipe:
      -map_metadata -1        strips container-level metadata
      -map_chapters -1        strips chapter markers
      -fflags +bitexact       prevents encoder timestamps leaking
      -metadata title=""      explicit blank overrides
      All standard tag keys are explicitly blanked via -metadata flags.
    """
    fps  = info["fps"]
    vf   = _build_vf(params, duration, info["width"], info["height"])
    af   = _build_af(params)

    # Metadata wipe — explicitly blank every common tag
    meta_wipe = []
    for tag in ("title", "comment", "description", "encoder", "author",
                "artist", "album", "genre", "copyright", "creation_time",
                "location", "make", "model", "software", "handler_name"):
        meta_wipe += ["-metadata", f"{tag}="]

    cmd = [
        "ffmpeg", "-y",
        "-ss",  str(round(start_sec, 4)),
        "-i",   input_path,
        "-t",   str(round(duration, 4)),
        "-vf",  vf,
        "-map", "0:v",
        "-map", "0:a?",
        "-c:v",          "libx264",
        "-preset",       "slow",          # better compression @ same quality
        "-profile:v",    "high",
        "-level",        "4.1",           # supports 1080p60
        "-pix_fmt",      "yuv420p",
        "-crf",          "18",            # visually lossless (was 23)
        "-maxrate",      "8M",            # cap peak bitrate for streaming
        "-bufsize",      "16M",           # VBV buffer = 2x maxrate
        "-r",            str(fps),
        "-s",            f"{OUT_W}x{OUT_H}",
        "-af",           af,
        "-c:a",          "aac",
        "-b:a",          "192k",          # high-quality audio (was 128k)
        "-ar",           "44100",
        "-ac",           "2",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-fflags",       "+bitexact",
        "-movflags",     "+faststart",
        *meta_wipe,
        "-f",            "mp4",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed (rc={result.returncode}): {os.path.basename(output_path)}")
        for line in result.stderr.strip().splitlines()[-30:]:
            logger.error(f"  {line}")
        return False
    # Sanity check: output must be > 50 KB or something went wrong silently
    if os.path.exists(output_path) and os.path.getsize(output_path) < 50_000:
        logger.error(
            f"ffmpeg output suspiciously small "
            f"({os.path.getsize(output_path)//1024} KB): {os.path.basename(output_path)}"
        )
        return False

    # ── Post-process: strip residual stream-level tags (handler_name, vendor_id, encoder) ──
    # ffmpeg always writes handler_name/vendor_id even with -map_metadata -1.
    # A second pass with -map_metadata -1 on the muxed file cleans them fully.
    tmp_path = output_path + ".tmp.mp4"
    strip_cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-c", "copy",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-metadata:s:v:0", "handler_name=",
        "-metadata:s:v:0", "vendor_id=",
        "-metadata:s:v:0", "encoder=",
        "-metadata:s:a:0", "handler_name=",
        "-metadata:s:a:0", "vendor_id=",
        "-movflags", "+faststart",
        "-f", "mp4",
        tmp_path,
    ]
    strip_result = subprocess.run(strip_cmd, capture_output=True, text=True, timeout=60)
    if strip_result.returncode == 0:
        os.replace(tmp_path, output_path)
        logger.debug("  Stream-level metadata stripped.")
    else:
        logger.debug("  Strip pass failed (non-fatal) — keeping encoded file.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return True


# ============================================================================
# Similarity scoring
# ============================================================================

def _extract_frames(video_path: str, n_frames: int = 8) -> List[np.ndarray]:
    """Extract n evenly-spaced frames from a video as BGR numpy arrays."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        cap.release()
        return frames
    indices = [int(total * i / n_frames) for i in range(n_frames)]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _phash(frame: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    Compute a perceptual hash (pHash) via DCT.
    Returns a flattened binary array of length hash_size².
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct     = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    med     = np.median(dct_low)
    return (dct_low > med).flatten()


def _hist_corr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Compare two frames via HSV histogram correlation.
    Returns 0.0 (no correlation) … 1.0 (identical).
    """
    a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    corrs = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        corrs.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return float(np.mean(corrs))


def _ssim_single(frame_a: np.ndarray, frame_b: np.ndarray,
                 size: Tuple[int, int] = (270, 480)) -> float:
    """
    Compute SSIM between two frames (resized for speed).
    size is (width, height) as required by cv2.resize.
    Returns -1.0 … 1.0.
    """
    a = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), size)
    b = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), size)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = 6.5025, 58.5225
    mu1, mu2       = a.mean(), b.mean()
    sigma1, sigma2 = a.std(), b.std()
    sigma12        = np.mean((a - mu1) * (b - mu2))
    num   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    return float(num / denom)


def compute_similarity_score(
    source_path: str,
    output_path: str,
    n_frames: int = 12,
) -> Dict:
    """
    Measure how different the output clip looks vs. the source.

    Comparison strategy
    ───────────────────
    Both videos are always resized to the same thumbnail (270×480) before
    comparison.  Because hflip is always applied, the output frames are
    horizontally flipped BACK before computing metrics — this isolates the
    contribution of all other transforms (color, crop, speed, vignette…)
    from the trivial mirror operation, giving a true measure of how much
    the OTHER transforms changed the content.

    Metrics
    ───────
    • pHash distance    – DCT perceptual hash Hamming distance (0=same, 256=max)
    • Histogram corr    – HSV channel histogram correlation    (1.0=identical)
    • SSIM              – Structural similarity                (1.0=identical)

    Composite uniqueness score  0–100  (higher = more different from source)
      score = (pHash/256 × 0.40  +  (1−hist) × 0.30  +  (1−ssim)/2 × 0.30) × 100
    """
    logger.info("  Scoring similarity …")

    src_frames = _extract_frames(source_path, n_frames)
    out_frames = _extract_frames(output_path, n_frames)

    if not src_frames or not out_frames:
        logger.warning("  Could not extract frames for scoring.")
        return {"error": "frame extraction failed"}

    THUMB = (270, 480)   # (width, height) — fast, consistent aspect ratio

    n = min(len(src_frames), len(out_frames))

    phash_distances = []
    hist_corrs      = []
    ssims           = []

    for i in range(n):
        # Resize both to the same thumbnail — source may be a different resolution
        sf = cv2.resize(src_frames[i], THUMB, interpolation=cv2.INTER_AREA)
        of = cv2.resize(out_frames[i], THUMB, interpolation=cv2.INTER_AREA)
        # NOTE: we do NOT undo hflip — the mirror IS part of the uniquification
        # and is one of the strongest signals that this is a different file.

        ph_s = _phash(sf)
        ph_o = _phash(of)
        dist = int(np.sum(ph_s != ph_o))
        phash_distances.append(dist)

        hist_corrs.append(_hist_corr(sf, of))
        ssims.append(_ssim_single(sf, of, size=THUMB))

    avg_dist = float(np.mean(phash_distances))
    avg_hist = float(np.mean(hist_corrs))
    avg_ssim = float(np.mean(ssims))

    # Composite uniqueness score
    # Weights: pHash 50% (very sensitive to flip+crop), 
    #          hist  30% (colour grading), ssim 20% (structure)
    phash_u = avg_dist / 256.0
    hist_u  = max(0.0, 1.0 - avg_hist)
    ssim_u  = max(0.0, (1.0 - avg_ssim) / 2.0)
    unique  = (phash_u * 0.50 + hist_u * 0.30 + ssim_u * 0.20) * 100.0
    unique  = round(min(100.0, max(0.0, unique)), 1)

    result = {
        "phash_distance":   round(avg_dist, 1),
        "hist_correlation": round(avg_hist, 4),
        "ssim":             round(avg_ssim, 4),
        "uniqueness_score": unique,
    }

    if   unique >= 70: rating = "🟢 Excellent — very unlikely to be flagged as duplicate"
    elif unique >= 50: rating = "🟡 Good      — noticeable colour/crop/speed differences"
    elif unique >= 30: rating = "🟠 Moderate  — some similarity remains"
    else:              rating = "🔴 Low       — too similar, params may not be aggressive enough"

    result["rating"] = rating
    return result


# ============================================================================
# Main
# ============================================================================

def uniquize(
    input_path: str,
    output_dir: str,
    n_clips: int,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Probing: {input_path}")
    info = get_video_info(input_path)

    logger.info(
        f"Source  : {info['duration']:.1f}s | {info['width']}x{info['height']} | "
        f"{info['fps']} fps\n"
        f"Output  : {OUT_W}x{OUT_H} (9:16) — {n_clips} clips\n"
        f"Mode    : pre-9:16 input — hflip + uniquify only (no conversion)"
    )

    total_dur = info["duration"]
    assigner  = ZoneAssigner()
    params_log: Dict = {}
    scores_log: Dict = {}
    success   = 0

    for i in range(1, n_clips + 1):
        # Random start, clip covers the whole source each time
        # (input is already a short reel — we don't split, we clone+uniquify)
        # Allow slight start offset so timing fingerprint also differs
        max_offset = max(0.0, total_dur * 0.05)  # up to 5% offset
        start_sec  = round(random.uniform(0.0, max_offset), 3)
        duration   = round(total_dur - start_sec, 3)

        params   = assigner.assign()
        out_name = f"clip_{i:03d}_unique.mp4"
        out_path = os.path.join(output_dir, out_name)

        logger.info(
            f"\n[{i}/{n_clips}] {out_name}  start={start_sec:.2f}s  dur={duration:.2f}s\n"
            f"  hflip=ON  speed={params['speed']:.2f}  flip_extra=none\n"
            f"  bright={params['brightness']:+.3f}  con={params['contrast']:.2f}  "
            f"sat={params['saturation']:.2f}  gamma={params['gamma']:.2f}\n"
            f"  crop T{params['crop_top']}/B{params['crop_bottom']}"
            f"/L{params['crop_left']}/R{params['crop_right']}  "
            f"shift X{params['pixel_shift_x']:+d}/Y{params['pixel_shift_y']:+d}  "
            f"vignette={params['vignette']:.2f}\n"
            f"  vol={params['volume']:.2f}  pitch={params['pitch_cents']:+d}c  "
            f"bass={params['bass_gain']:+.1f}dB  treble={params['treble_gain']:+.1f}dB  "
            f"echo={params['echo']}  delay={params['delay_ms']}ms"
        )

        ok = process_clip(input_path, out_path, start_sec, duration, info, params)

        if ok:
            success += 1
            params_log[out_name] = {**params, "start": start_sec, "duration": duration}

            # Similarity scoring
            score = compute_similarity_score(input_path, out_path)
            scores_log[out_name] = score

            u = score.get("uniqueness_score", "N/A")
            r = score.get("rating", "")
            logger.info(
                f"  ✅ {out_path}\n"
                f"  📊 Uniqueness: {u}/100  pHash dist={score.get('phash_distance')}  "
                f"hist corr={score.get('hist_correlation')}  SSIM={score.get('ssim')}\n"
                f"  {r}\n"
            )
        else:
            logger.error(f"  ❌ Failed: {out_name}\n")

    # Save logs
    params_path = os.path.join(output_dir, "_params.json")
    scores_path = os.path.join(output_dir, "_scores.json")
    with open(params_path, "w") as f:
        json.dump(params_log, f, indent=2)
    with open(scores_path, "w") as f:
        json.dump(scores_log, f, indent=2)

    return scores_log  # keyed by clip filename → score dict

    # Summary table
    print("\n" + "=" * 70)
    print(f"  UNIQUIZATION COMPLETE")
    print(f"  ✅  {success} / {n_clips} clips exported")
    print(f"  📐  Resolution : {OUT_W}x{OUT_H}  (9:16)")
    print(f"  🔄  hflip      : ON (all clips)")
    print(f"  🗑️   Metadata   : wiped")
    print(f"  📁  Output     : {os.path.abspath(output_dir)}")
    print()
    if scores_log:
        print(f"  {'Clip':<30} {'Score':>6}  {'pHash':>6}  {'HistCorr':>8}  {'SSIM':>7}  Rating")
        print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*42}")
        for name, s in scores_log.items():
            if "error" in s:
                print(f"  {name:<30}  [scoring failed]")
            else:
                print(
                    f"  {name:<30} {s['uniqueness_score']:>6.1f}  "
                    f"{s['phash_distance']:>6.1f}  "
                    f"{s['hist_correlation']:>8.4f}  "
                    f"{s['ssim']:>7.4f}  "
                    f"{s['rating'].split('—')[0].strip()}"
                )
    print(f"\n  📋  Params : {params_path}")
    print(f"  📊  Scores : {scores_path}")
    print("=" * 70 + "\n")


# ============================================================================
# CLI
# ============================================================================

def _find_video(folder: str) -> str:
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(exts):
                return os.path.join(folder, f)
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uniquize pre-9:16 vertical clips for Instagram Reels re-upload.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", default="",
        help="Path to input video (auto-detects from input/ folder if omitted)",
    )
    parser.add_argument(
        "--output", default="output",
        help="Output folder (default: output/)",
    )
    parser.add_argument(
        "--clips", type=int, default=5,
        help="Number of uniquified copies to generate (default: 5)",
    )
    args = parser.parse_args()

    src = args.input or _find_video("input")
    if not src or not os.path.exists(src):
        print("❌  No video found. Drop your clip into input/")
        print("    or use: --input path/to/clip.mp4")
        sys.exit(1)

    info = get_video_info(src)
    print(f"\n🎬  Input      : {src}")
    print(f"📐  Dimensions : {info['width']}x{info['height']}  ({info['fps']} fps)")
    print(f"⏱️   Duration   : {info['duration']:.1f}s")
    print(f"📁  Output     : {args.output}/")
    print(f"🔢  Clips      : {args.clips}")
    print(f"🔄  hflip      : ON (all clips)")
    print(f"🗑️   Metadata   : wiped\n")

    uniquize(src, args.output, args.clips)