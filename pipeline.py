#!/usr/bin/env python3
"""
pipeline.py
===========
Downloads Instagram Reel URLs from saved_urls.txt, runs each video through
the uniquizer, and saves the final processed clips to output_videos/.

Usage:
    python pipeline.py
    python pipeline.py --urls saved_urls.txt --output output_videos
    python pipeline.py --urls saved_urls.txt --output output_videos --keep-tmp

Authentication (Instagram requires login for most Reels):
    OPTION A — Cookie file (most reliable, recommended):
        1. Install the browser extension "Get cookies.txt LOCALLY"
        2. Go to instagram.com while logged in
        3. Click the extension → Export → save as  instagram_cookies.txt
        4. Place instagram_cookies.txt next to pipeline.py

    OPTION B — Live browser (auto-detected, may fail if browser is open):
        Just be logged into Instagram in Chrome/Firefox — pipeline auto-detects.

Pipeline per URL:
    1. read_links()      — parse saved_urls.txt
    2. download_video()  — yt-dlp → highest quality mp4 into tmp/
    3. process_video()   — uniquizer (hflip + color/audio/crop transforms)
    4. save_video()      — rename / move to output_videos/video_NNN.mp4

Requirements:
    pip install yt-dlp opencv-python-headless numpy
    ffmpeg must be on PATH
"""

import argparse
import importlib.util
import logging
import os
import re
import shutil

import sys
import time
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_URLS_FILE  = Path(r".\saved_urls.txt")
DEFAULT_OUTPUT_DIR = Path("output_videos")
TMP_DIR            = Path("tmp_downloads")
ERROR_LOG          = Path("download_errors.log")
COOKIES_FILE       = Path("instagram_cookies.txt")   # Netscape format
UNIQUIZER_SCRIPT   = Path(__file__).parent / "uniquize.py"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("pipeline")
    err_handler = logging.FileHandler(ERROR_LOG, encoding="utf-8")
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(err_handler)
    return log

logger = _setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# 1. read_links
# ─────────────────────────────────────────────────────────────────────────────
def read_links(urls_file: Path) -> List[str]:
    """
    Parse the URL file. Accepts lines in formats:
        https://www.instagram.com/reel/ABC123/
        https://www.instagram.com/reel/ABC123/   (likes: 12,345)   [niche: roblox]

    Returns a deduplicated list of URLs in file order.
    """
    if not urls_file.exists():
        logger.error(f"URL file not found: {urls_file}")
        sys.exit(1)

    seen: set  = set()
    urls: list = []

    for raw in urls_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        url = line.split()[0]
        if url in seen:
            logger.debug(f"Skipping duplicate: {url}")
            continue
        seen.add(url)
        urls.append(url)

    logger.info(f"Found {len(urls)} unique URL(s) in {urls_file}")
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# 2. download_video  — uses yt_dlp Python API (no subprocess)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp not installed. Run:  pip install yt-dlp")
    sys.exit(1)


def download_video(url: str, dest_dir: Path, index: int) -> Optional[Path]:
    """
    Download a single Instagram Reel using the yt_dlp Python API.

    Uses 'best' format — the simplest selector that reliably works for
    Instagram without requiring ffmpeg merging of separate streams.

    Skips silently if the output file already exists (resume-safe).
    Returns the Path to the downloaded file, or None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Stable output path — %(ext)s lets yt-dlp fill in the real extension
    out_template = str(dest_dir / f"raw_{index:03d}.%(ext)s")

    # ── Skip if already on disk ─────────────────────────────────────────
    for ext in ("mp4", "mkv", "webm", "mov"):
        existing = dest_dir / f"raw_{index:03d}.{ext}"
        if existing.exists() and existing.stat().st_size > 10_000:
            logger.info(f"  [#{index}] Already downloaded — skipping: {existing.name}")
            return existing

    ydl_opts = {
        "outtmpl":  out_template,
        "format":   "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "quiet":    False,       # let yt-dlp print progress to stdout
        "noprogress": False,
        "noplaylist": True,
        "retries":  5,
        "socket_timeout": 30,
        "merge_output_format": "mp4",
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    logger.info(f"  [#{index}] Downloading: {url}")
    t0 = time.time()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        elapsed = time.time() - t0

        # Find what yt-dlp actually wrote
        for ext in ("mp4", "mkv", "webm", "mov"):
            out = dest_dir / f"raw_{index:03d}.{ext}"
            if out.exists() and out.stat().st_size > 10_000:
                size_mb = out.stat().st_size / 1_048_576
                logger.info(
                    f"  [#{index}] ✅ Downloaded in {elapsed:.1f}s "
                    f"({size_mb:.1f} MB) → {out.name}"
                )
                return out

        logger.error(f"  [#{index}] ❌ File not found after download: {url}")
        return None

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"  [#{index}] ❌ Download error: {e}")
        return None
    except Exception as e:
        logger.error(f"  [#{index}] ❌ Unexpected error: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. process_video  — calls uniquizer as a library
# ─────────────────────────────────────────────────────────────────────────────
def _load_uniquizer():
    """Import uniquize.py as a module from several candidate locations."""
    search_paths = [
        UNIQUIZER_SCRIPT,
        Path(__file__).parent / "uniquize.py",
        Path("uniquize.py"),
        Path("uniquizer") / "uniquize.py",
    ]
    for p in search_paths:
        if p.exists():
            spec   = importlib.util.spec_from_file_location("uniquize", str(p))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


# Minimum uniqueness score to allow upload (0–100). Raise to be stricter.
UPLOAD_THRESHOLD = 45.0

def process_video(
    raw_path: Path,
    work_dir: Path,
    index: int,
) -> tuple:
    """
    Run the uniquizer on raw_path → one uniquified clip in work_dir.

    Returns (path, score_dict) on success, or (None, None) on failure.
    score_dict keys: uniqueness_score, phash_distance, hist_correlation,
                     ssim, rating, upload_ok
    """
    uniquize_mod = _load_uniquizer()
    if uniquize_mod is None:
        logger.error("  uniquize.py not found. Place it next to pipeline.py.")
        return None, None

    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  [#{index}] Processing with uniquizer → {work_dir}/")

    try:
        u_log = logging.getLogger("uniquizer")
        u_log.handlers.clear()
        u_log.propagate = True

        scores_log = uniquize_mod.uniquize(
            input_path=str(raw_path),
            output_dir=str(work_dir),
            n_clips=1,
        )

        candidates = sorted(work_dir.glob("clip_*_unique.mp4"))
        if not candidates:
            logger.error(f"  [#{index}] Uniquizer produced no output in {work_dir}")
            return None, None

        out_path  = candidates[0]
        size_mb   = out_path.stat().st_size / 1_048_576

        # Pull the score for this clip from the returned dict
        score = {}
        if scores_log:
            score = list(scores_log.values())[0]

        u_score   = score.get("uniqueness_score", 0.0)
        upload_ok = u_score >= UPLOAD_THRESHOLD
        score["upload_ok"] = upload_ok

        # ── Upload decision banner ──────────────────────────────────────
        if upload_ok:
            verdict = f"✅ UPLOAD  (score {u_score}/100 ≥ threshold {UPLOAD_THRESHOLD})"
        else:
            verdict = f"🚫 SKIP    (score {u_score}/100 < threshold {UPLOAD_THRESHOLD})"

        logger.info(
            f"  [#{index}] Uniquized → {out_path.name} ({size_mb:.1f} MB)\n"
            f"  ┌─ Uniqueness score : {u_score}/100\n"
            f"  ├─ pHash distance   : {score.get('phash_distance', 'N/A')}\n"
            f"  ├─ Hist correlation : {score.get('hist_correlation', 'N/A')}\n"
            f"  ├─ SSIM             : {score.get('ssim', 'N/A')}\n"
            f"  ├─ Rating           : {score.get('rating', 'N/A')}\n"
            f"  └─ Decision         : {verdict}"
        )

        return out_path, score

    except Exception as exc:
        logger.error(f"  [#{index}] Uniquizer error: {exc}", exc_info=True)
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. save_video
# ─────────────────────────────────────────────────────────────────────────────
def save_video(processed_path: Path, output_dir: Path, index: int) -> Path:
    """Copy processed clip to output_dir/video_NNN.mp4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"video_{index:03d}.mp4"
    shutil.copy2(processed_path, final)
    size_mb = final.stat().st_size / 1_048_576
    logger.info(f"  [#{index}] 💾 Saved → {final}  ({size_mb:.1f} MB)")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_dependencies() -> None:
    """Abort early if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found on PATH. Install from https://ffmpeg.org")
        sys.exit(1)
    logger.info("  ffmpeg found ✅")


def _already_saved(output_dir: Path, index: int) -> bool:
    p = output_dir / f"video_{index:03d}.mp4"
    return p.exists() and p.stat().st_size > 10_000


def _print_summary(
    total: int, success: int, skipped: int, skipped_low_score: int, failed: int,
    output_dir: Path, elapsed: float,
) -> None:
    mins, secs = divmod(int(elapsed), 60)
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  ✅  Saved     : {success}")
    print(f"  🚫  Low score : {skipped_low_score}  (below uniqueness threshold — not uploaded)")
    print(f"  ⏭️   Skipped   : {skipped}  (already existed)")
    print(f"  ❌  Failed    : {failed}")
    print(f"  📦  Total     : {total}")
    print(f"  ⏱️   Duration  : {mins}m {secs}s")
    print(f"  📁  Output    : {output_dir.resolve()}")
    if failed:
        print(f"  📋  Errors    : {ERROR_LOG.resolve()}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download → Uniquize → Save Instagram Reels pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--urls",   type=Path, default=DEFAULT_URLS_FILE,
                        help=f"URL list file (default: {DEFAULT_URLS_FILE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--tmp",    type=Path, default=TMP_DIR,
                        help=f"Temp download folder (default: {TMP_DIR})")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep raw downloads and work dirs after processing")
    args = parser.parse_args()

    session_start = time.time()

    _check_dependencies()
    urls = read_links(args.urls)
    if not urls:
        logger.error("No URLs to process.")
        sys.exit(0)

    total = len(urls)
    success = skipped = failed = skipped_low_score = 0

    print(f"\n{'=' * 65}")
    print(f"  Instagram Reel Pipeline")
    print(f"  URLs      : {total}")
    print(f"  Output    : {args.output}/")
    print(f"  Temp dir  : {args.tmp}/")
    print(f"  Keep tmp  : {args.keep_tmp}")
    print(f"{'=' * 65}\n")

    for idx, url in enumerate(urls, start=1):
        print(f"\n── [{idx}/{total}] {url[:80]}{'…' if len(url) > 80 else ''}")

        if _already_saved(args.output, idx):
            logger.info(f"  [#{idx}] Output already exists — skipping.")
            skipped += 1
            continue

        work_root     = args.tmp / f"job_{idx:03d}"
        uniquize_work = work_root / "uniquized"
        work_root.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1 — Download
            raw_path = download_video(url, work_root, idx)
            if raw_path is None:
                failed += 1
                continue

            # Step 2 — Uniquize
            processed, score = process_video(raw_path, uniquize_work, idx)
            if processed is None:
                logger.error(f"  [#{idx}] ❌ Processing failed — skipping.")
                failed += 1
                continue

            # Step 3 — Upload gate: only save if score passes threshold
            if score and not score.get("upload_ok", True):
                u = score.get("uniqueness_score", 0)
                logger.warning(
                    f"  [#{idx}] 🚫 Video NOT saved — uniqueness score "
                    f"{u}/100 is below threshold {UPLOAD_THRESHOLD}.\n"
                    f"           The video is too similar to the source to safely re-upload."
                )
                skipped_low_score += 1
                continue

            # Step 4 — Save
            save_video(processed, args.output, idx)
            success += 1

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break
        except Exception as exc:
            logger.error(f"  [#{idx}] ❌ Unexpected error: {exc}", exc_info=True)
            failed += 1

        finally:
            if not args.keep_tmp and work_root.exists():
                try:
                    shutil.rmtree(work_root)
                except OSError as e:
                    logger.warning(f"  Could not clean {work_root}: {e}")

    _print_summary(total, success, skipped, skipped_low_score, failed, args.output,
                   time.time() - session_start)


if __name__ == "__main__":
    main()