import yt_dlp
import os
import shutil
import time

# File containing YouTube URLs
LINK_FILE = "links.txt"

# Output folder
OUTPUT_DIR = "Downloaded_Videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prefer H.264 (avc1) — OpenCV cannot reliably read AV1 ("Missing Sequence Header").
ydl_opts = {
    "format": (
        "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
        "bestvideo[height<=1080][vcodec^=avc]+bestaudio/"
        "bestvideo[height<=1080][vcodec!=av01][vcodec!=av1]+bestaudio/"
        "best[height<=1080]"
    ),
    "merge_output_format": "mp4",
    "outtmpl": os.path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),
    "ignoreerrors": True,
    # Bypass "Sign in to confirm you’re not a bot" (Chrome must be closed or unlocked)
    "cookiesfrombrowser": ("chrome",),
    # Needed so yt-dlp can solve YouTube JS challenges (video formats, not just images)
    "remote_components": ["ejs:github"],
    # Reduce 429 rate-limit hits
    "sleep_interval": 3,
    "max_sleep_interval": 8,
    "retries": 5,
    "fragment_retries": 5,
}

# Prefer Deno (default), fall back to Node if present
deno = shutil.which("deno") or os.path.expanduser("~/.deno/bin/deno")
node = shutil.which("node") or shutil.which("nodejs")
if deno and os.path.isfile(deno):
    ydl_opts["js_runtimes"] = {"deno": {"path": deno}}
elif node:
    ydl_opts["js_runtimes"] = {"node": {"path": node}}

with open(LINK_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, url in enumerate(urls):
        try:
            print(f"Downloading: {url}")
            ydl.download([url])
        except Exception as e:
            print(f"Failed: {url}")
            print(e)
        if i < len(urls) - 1:
            time.sleep(5)  # extra pause between videos to avoid 429

print("All downloads completed.")
