# Clips Uniquizer

**Clips Uniquizer** is a video processing pipeline designed to batch download, transform, and randomize video clips to bypass automated content detection mechanisms (such as social media duplicate filters) by making each output clip digitally unique.

---

## 💡 Suggested Project Titles

If you are looking for alternative names for this repository, consider:
* **ClipsUniquizer** (Original / Direct)
* **Video Metadata & Hash Permutator**
* **AutoClip Uniquizer Engine**
* **Media-Pipeline-Uniquizer**

---

## 📌 Important Integration Note

> **Note:** To populate `saved_urls.txt` with video links for processing, use the companion repository [**`insta_android_bot`**](../insta_android_bot). 
> 
> The `insta_android_bot` tool automates the process of scraping, filtering, and extracting post/reel URLs directly from the Instagram Android app. Once extracted, copy or output those URLs into `saved_urls.txt` in this repository to run the automated download and modification pipeline.

---

## 📁 Repository Structure

```text
clips_uniquizer/
├── pipeline.py            # Main automation entry point (orchestrates downloading and processing)
├── uniquize.py            # Core video processing script (applies subtle visual/audio variations)
├── saved_urls.txt         # Text file containing target video URLs to download
├── download_errors.log    # Log file tracking failed downloads
├── tmp_downloads/        # Temporary storage for downloaded raw clips
├── output_videos/        # Destination directory for uniquely modified videos
└── output/
    └── uniquize_log.txt  # Detailed execution log of applied transformations
```

---

## ⚙️ How It Works

1. **URL Collection:** Target URLs are supplied via `saved_urls.txt` (extracted via `insta_android_bot`).
2. **Batch Downloading:** `pipeline.py` reads the batch of URLs and downloads the raw video clips into `tmp_downloads/`.
3. **Uniquification Engine:** `uniquize.py` processes each downloaded clip by applying slight alterations, such as:
   * Metadata stripping/overwriting
   * Micro-adjustments to color, brightness, or contrast
   * Subtle scaling, cropping, or frame rate tweaks
   * Audio pitch/frequency adjustments
4. **Export:** Modified, unique clips are output to the `output_videos/` directory ready for reuse or reposting.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* FFmpeg installed and added to your system `PATH`
* `yt-dlp` (or relevant video downloader dependency)

### Usage

1. **Extract URLs:** Run the `insta_android_bot` workflow to gather target Instagram URLs into `saved_urls.txt`.
2. **Run the Pipeline:**
   ```bash
   python pipeline.py
   ```
3. **Check Results:** Retrieve transformed videos from the `output_videos/` directory.

---

## 🛠 Troubleshooting & Logs

* If a video fails to download, inspect `download_errors.log`.
* To check modification status and parameters used during transformations, view `output/uniquize_log.txt`.