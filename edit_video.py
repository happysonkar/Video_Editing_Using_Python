import cv2
import json
import numpy as np
import subprocess
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import io

# ── Download emoji PNGs from OpenMoji (open source) ──────────────
def download_emoji(url, save_path):
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded: {save_path}")

def prepare_emojis():
    os.makedirs("emojis", exist_ok=True)
    emojis = {
        "laugh": "https://abs.twimg.com/emoji/v2/72x72/1f602.png",
    }
    # Use PNG versions from OpenMoji GitHub
    png_urls = {
        "laugh": "https://abs.twimg.com/emoji/v2/72x72/1f602.png",
        "fire":  "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/72x72/1F525.png",
    }
    for name, url in png_urls.items():
        download_emoji(url, f"emojis/{name}.png")

def load_emoji(path, size):
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    return np.array(img)

def overlay_emoji(frame, emoji_rgba, cx, cy):
    """Overlay an RGBA emoji image centered at (cx, cy) on a BGR frame."""
    h, w = frame.shape[:2]
    eh, ew = emoji_rgba.shape[:2]

    x1 = cx - ew // 2
    y1 = cy - eh // 2
    x2 = x1 + ew
    y2 = y1 + eh

    # Clamp to frame bounds
    fx1 = max(x1, 0); fy1 = max(y1, 0)
    fx2 = min(x2, w); fy2 = min(y2, h)
    ex1 = fx1 - x1;   ey1 = fy1 - y1
    ex2 = ex1 + (fx2 - fx1)
    ey2 = ey1 + (fy2 - fy1)

    if fx2 <= fx1 or fy2 <= fy1:
        return frame

    roi      = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    emoji_crop = emoji_rgba[ey1:ey2, ex1:ex2]
    rgb      = emoji_crop[:, :, :3].astype(np.float32)
    alpha    = emoji_crop[:, :, 3:4].astype(np.float32) / 255.0

    # Convert RGB → BGR
    bgr = rgb[:, :, ::-1]

    blended = bgr * alpha + roi * (1 - alpha)
    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)
    return frame

def add_text_with_outline(frame, text, position, font_scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, position, font, font_scale,
                (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, font_scale,
                color, thickness, cv2.LINE_AA)

def add_overlays(frame, frame_idx, total_frames,
                 emoji_laugh, emoji_fire, watermark_text="@YourChannel"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    # ── 1. Caption bar at bottom center with emoji ────────────────
    caption      = "Wait for the ending"
    font_scale   = w / 800.0
    thickness    = 2
    (tw, th), _  = cv2.getTextSize(caption, font, font_scale, thickness)

    emoji_size   = int(th * 2.2)
    laugh_resized = cv2.resize(
        emoji_laugh, (emoji_size, emoji_size),
        interpolation=cv2.INTER_LANCZOS4
    ) if emoji_laugh is not None else None

    gap          = 10
    total_width  = tw + (emoji_size + gap if laugh_resized is not None else 0)
    start_x      = (w - total_width) // 2
    caption_y    = h - int(h * 0.08)

    # Dark semi-transparent pill background
    pad     = 14
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (start_x - pad, caption_y - th - pad),
                  (start_x + total_width + pad, caption_y + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw caption text
    add_text_with_outline(frame, caption, (start_x, caption_y),
                          font_scale, (255, 255, 255), thickness)

    # Overlay laugh emoji right after text
    if laugh_resized is not None:
        ex = start_x + tw + gap + emoji_size // 2
        ey = caption_y - th // 2
        frame = overlay_emoji(frame, laugh_resized, ex, ey)

    # ── 2. Large laugh emoji in center of frame ───────────────────
    if emoji_laugh is not None:
        center_size  = int(min(w, h) * 0.15)
        center_emoji = cv2.resize(emoji_laugh, (center_size, center_size),
                                  interpolation=cv2.INTER_LANCZOS4)
        frame = overlay_emoji(frame, center_emoji, w // 2, h // 2)

    # ── 3. Watermark (semi-transparent, center) ───────────────────
    wm_scale     = w / 900.0
    (wmw, wmh), _ = cv2.getTextSize(watermark_text, font, wm_scale, 2)
    wm_overlay   = frame.copy()
    cv2.putText(wm_overlay, watermark_text,
                ((w - wmw) // 2, h // 2 - int(h * 0.12)),
                font, wm_scale, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.addWeighted(wm_overlay, 0.25, frame, 0.75, 0, frame)

    # ── 4. Cyan progress bar ──────────────────────────────────────
    progress = int((frame_idx / max(total_frames, 1)) * w)
    cv2.rectangle(frame, (0, h - 6), (progress, h), (0, 200, 255), -1)

    # ── 5. Blinking top hype banner (first 3 sec) ─────────────────
    alpha  = 0.6 if (frame_idx // 15) % 2 == 0 else 0.3
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, int(h * 0.10)), (0, 0, 0), -1)
    cv2.addWeighted(banner, alpha, frame, 1 - alpha, 0, frame)

    top_text     = "Follow for more videos like this"
    b_scale      = w / 900.0
    (btw, bth), _ = cv2.getTextSize(top_text, font, b_scale, 2)
    bx           = (w - btw) // 2
    by           = int(h * 0.075)

    # Fire emojis flanking the text
    if emoji_fire is not None:
        f_size       = int(bth * 2)
        fire_resized = cv2.resize(emoji_fire, (f_size, f_size),
                                    interpolation=cv2.INTER_LANCZOS4)
        frame = overlay_emoji(frame, fire_resized,
                                bx - f_size, by - bth // 2)
        frame = overlay_emoji(frame, fire_resized,
                                bx + btw + f_size, by - bth // 2)

    add_text_with_outline(frame, top_text, (bx, by),
                            b_scale, (0, 200, 255), 2)

    return frame


def probe_fps_ffprobe(path):
    """Return container-reported video FPS, or None if unavailable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    try:
        streams = json.loads(out).get("streams") or []
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    s = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = s.get(key)
        if not rate or rate == "0/0":
            continue
        num, _, den = rate.partition("/")
        try:
            v = float(num) / float(den) if den else float(num)
        except ValueError:
            continue
        if 0.5 < v < 240:
            return v
    return None


def _resolve_fps(cap, input_path):
    """Match source timing so muxed audio stays in sync (no artificial speed change)."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 0.5 or fps > 240:
        fps = probe_fps_ffprobe(input_path) or 0.0
    if fps < 0.5 or fps > 240:
        fps = 30.0
    return fps


def _transform_video_with_emojis(input_path, output_path, watermark, emoji_laugh, emoji_fire):
    cap = cv2.VideoCapture(input_path)
    fps = _resolve_fps(cap, input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = "temp_no_audio.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.02) : int(h * 0.98), int(w * 0.02) : int(w * 0.98)]
        frame = cv2.resize(crop, (w, h))
        noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 1, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))

        frame = add_overlays(frame, frame_idx, total, emoji_laugh, emoji_fire, watermark)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processing... {frame_idx}/{total} frames")

    cap.release()
    out.release()

    # Force CFR timing for the OpenCV AVI (metadata/timestamps are often wrong).
    # Must match the rate used in VideoWriter so duration matches source audio.
    fps_s = f"{fps:.6f}".rstrip("0").rstrip(".")
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-r",
        fps_s,
        "-i",
        temp_video,
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Some sources need re-encoded audio for MP4 (e.g. unsupported codec)
        cmd_fallback = list(cmd)
        i = cmd_fallback.index("-c:a") + 1
        cmd_fallback[i] = "aac"
        cmd_fallback[i + 1 : i + 1] = ["-b:a", "192k"]
        subprocess.run(cmd_fallback, check=True)
    os.remove(temp_video)
    print(f"\n✅ Done! Saved to: {output_path}")


def transform_video(input_path, output_path, watermark="@YourChannel"):
    prepare_emojis()
    emoji_laugh = load_emoji("emojis/laugh.png", 128) if os.path.exists("emojis/laugh.png") else None
    emoji_fire = load_emoji("emojis/fire.png", 128) if os.path.exists("emojis/fire.png") else None
    _transform_video_with_emojis(input_path, output_path, watermark, emoji_laugh, emoji_fire)


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def list_videos(folder):
    """Return sorted paths to video files in folder (non-recursive)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a directory: {folder}")
    names = []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in VIDEO_EXTENSIONS):
            names.append(name)
    return sorted(os.path.join(folder, n) for n in names)


def transform_folder(input_dir, output_dir, watermark="@YourChannel"):
    os.makedirs(output_dir, exist_ok=True)
    paths = list_videos(input_dir)
    if not paths:
        print(f"No videos found in {input_dir} ({', '.join(VIDEO_EXTENSIONS)})")
        return
    prepare_emojis()
    emoji_laugh = load_emoji("emojis/laugh.png", 128) if os.path.exists("emojis/laugh.png") else None
    emoji_fire = load_emoji("emojis/fire.png", 128) if os.path.exists("emojis/fire.png") else None

    for input_path in paths:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base}_edited.mp4")
        print(f"\n--- {input_path} -> {output_path} ---")
        _transform_video_with_emojis(
            input_path, output_path, watermark, emoji_laugh, emoji_fire
        )


# ── Run ───────────────────────────────────────────────────────────
# Process every video in input_dir; outputs go to output_dir as <name>_edited.mp4
transform_folder(
    input_dir="input",
    output_dir="output",
    watermark="ComedyCentral_TV",
)