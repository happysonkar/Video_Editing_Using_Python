import cv2
import json
import numpy as np
import subprocess
import os
import urllib.request
from PIL import Image

OUTPUT_SIZE = (720, 1280)

AUDIO_WORK_DIR = "audio_work"
AUDIO_SAMPLE_RATE = 44100
PITCH_SHIFT = 1.12
ORIGINAL_MIX = 0.30
EFFECTS_MIX = 0.70

# ── Download emoji PNGs from OpenMoji (open source) ──────────────
def download_emoji(url, save_path):
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded: {save_path}")

def prepare_emojis():
    os.makedirs("emojis", exist_ok=True)
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

    fx1 = max(x1, 0)
    fy1 = max(y1, 0)
    fx2 = min(x2, w)
    fy2 = min(y2, h)
    ex1 = fx1 - x1
    ey1 = fy1 - y1
    ex2 = ex1 + (fx2 - fx1)
    ey2 = ey1 + (fy2 - fy1)

    if fx2 <= fx1 or fy2 <= fy1:
        return frame

    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    emoji_crop = emoji_rgba[ey1:ey2, ex1:ex2]
    rgb = emoji_crop[:, :, :3].astype(np.float32)
    alpha = emoji_crop[:, :, 3:4].astype(np.float32) / 255.0
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


def resize_to_cover(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = max(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    rh, rw = resized.shape[:2]
    x1 = max(0, (rw - target_w) // 2)
    y1 = max(0, (rh - target_h) // 2)
    return resized[y1 : y1 + target_h, x1 : x1 + target_w]


def resize_to_fit(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_dynamic_reframe(frame, frame_idx, total_frames):
    h, w = frame.shape[:2]
    total = max(total_frames - 1, 1)
    t = frame_idx / total
    zoom = 1.05 + 0.035 * (0.5 + 0.5 * np.sin(2 * np.pi * t * 1.3))
    crop_w = max(2, int(w / zoom))
    crop_h = max(2, int(h / zoom))

    max_x = max(0, w - crop_w)
    max_y = max(0, h - crop_h)
    drift_x = int(max_x * 0.30 * np.sin(2 * np.pi * t * 1.1))
    drift_y = int(max_y * 0.24 * np.cos(2 * np.pi * t * 1.7))

    x1 = (w - crop_w) // 2 + drift_x
    y1 = (h - crop_h) // 2 + drift_y
    x1 = int(np.clip(x1, 0, max_x))
    y1 = int(np.clip(y1, 0, max_y))

    crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_color_grade(frame, frame_idx):
    base = frame.astype(np.float32)
    phase = (frame_idx // 90) % 3

    if phase == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 12 * np.sin(frame_idx / 12.0)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45 + 22, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.18 + 14, 0, 255)
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    elif phase == 1:
        sepia_matrix = np.array(
            [
                [0.131, 0.534, 0.272],
                [0.168, 0.686, 0.349],
                [0.189, 0.769, 0.393],
            ],
            dtype=np.float32,
        )
        sepia = cv2.transform(base, sepia_matrix)
        graded = 0.72 * np.clip(sepia, 0, 255) + 0.28 * base
    else:
        negative = 255.0 - base
        cool_tint = np.zeros_like(base)
        cool_tint[:, :, 0] = 20
        cool_tint[:, :, 2] = 8
        graded = 0.62 * base + 0.38 * negative + cool_tint

    rows = np.linspace(0.0, 1.0, graded.shape[0], dtype=np.float32)[:, None]
    graded[:, :, 0] = np.clip(graded[:, :, 0] + (1.0 - rows) * 14, 0, 255)
    graded[:, :, 2] = np.clip(graded[:, :, 2] + rows * 18, 0, 255)
    return np.clip(graded, 0, 255).astype(np.uint8)


def apply_wave_distortion(frame, frame_idx):
    h, w = frame.shape[:2]
    y_map, x_map = np.indices((h, w), dtype=np.float32)
    x_map = x_map + 6.0 * np.sin((y_map / 48.0) + frame_idx / 12.0)
    y_map = y_map + 3.0 * np.cos((x_map / 72.0) + frame_idx / 18.0)
    return cv2.remap(
        frame,
        x_map,
        y_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def apply_blur_and_sharpen(frame, frame_idx):
    sigma = 1.1 + 0.6 * (0.5 + 0.5 * np.sin(frame_idx / 15.0))
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharpened = cv2.addWeighted(frame, 1.45, blurred, -0.45, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_vignette(frame):
    h, w = frame.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    mask = 1.0 - 0.22 * (x * x + y * y)
    mask = np.clip(mask, 0.72, 1.0)
    return np.clip(frame.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def apply_scanlines(frame, frame_idx):
    scanlined = frame.astype(np.float32)
    scanlined[(frame_idx % 4) :: 4, :, :] *= 0.90
    return np.clip(scanlined, 0, 255).astype(np.uint8)


def apply_pixelate(frame, scale_divisor=12):
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame,
        (max(1, w // scale_divisor), max(1, h // scale_divisor)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_edge_glow(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.2)
    edge_overlay = np.zeros_like(frame)
    edge_overlay[:, :, 0] = edges // 3
    edge_overlay[:, :, 1] = edges // 2
    edge_overlay[:, :, 2] = edges
    return cv2.addWeighted(frame, 1.0, edge_overlay, 0.18, 0)


def add_styled_border(frame, frame_idx):
    h, w = frame.shape[:2]
    pulse = 0.5 + 0.5 * np.sin(frame_idx / 10.0)
    border_color = (
        int(120 + 60 * pulse),
        int(70 + 40 * pulse),
        int(220 - 50 * pulse),
    )
    thickness = max(4, min(h, w) // 85)
    cv2.rectangle(
        frame,
        (thickness // 2, thickness // 2),
        (w - thickness // 2 - 1, h - thickness // 2 - 1),
        border_color,
        thickness,
    )
    return frame


def build_vertical_canvas(frame, frame_idx):
    canvas_w, canvas_h = OUTPUT_SIZE
    background = resize_to_cover(frame, canvas_w, canvas_h)
    if (frame_idx // 150) % 2 == 1:
        background = cv2.flip(background, 1)
    background = cv2.GaussianBlur(background, (0, 0), 18)
    background = cv2.convertScaleAbs(background, alpha=0.72, beta=-12)
    background = apply_wave_distortion(background, frame_idx)
    background = apply_vignette(background)

    foreground = resize_to_fit(frame, int(canvas_w * 0.92), int(canvas_h * 0.72))
    fg_h, fg_w = foreground.shape[:2]
    x1 = (canvas_w - fg_w) // 2
    y1 = (canvas_h - fg_h) // 2 + int(24 * np.sin(frame_idx / 22.0))
    y1 = int(np.clip(y1, 44, canvas_h - fg_h - 44))

    canvas = background.copy()
    shadow = canvas.copy()
    pad = 16
    cv2.rectangle(
        shadow,
        (x1 - pad, y1 - pad),
        (x1 + fg_w + pad, y1 + fg_h + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(shadow, 0.28, canvas, 0.72, 0, canvas)
    canvas[y1 : y1 + fg_h, x1 : x1 + fg_w] = foreground
    cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x1 + fg_w + 4, y1 + fg_h + 4), (255, 255, 255), 2)
    return canvas


def add_rotated_preview(canvas, source_frame, frame_idx):
    preview = resize_to_fit(source_frame, int(canvas.shape[1] * 0.26), int(canvas.shape[0] * 0.18))
    if (frame_idx // 120) % 2 == 0:
        preview = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
    else:
        preview = cv2.rotate(preview, cv2.ROTATE_180)
    preview = apply_pixelate(preview, scale_divisor=10)
    preview = apply_edge_glow(preview)

    ph, pw = preview.shape[:2]
    x1 = canvas.shape[1] - pw - 28
    y1 = int(canvas.shape[0] * 0.16)

    panel = canvas.copy()
    cv2.rectangle(panel, (x1 - 12, y1 - 12), (x1 + pw + 12, y1 + ph + 12), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.34, canvas, 0.66, 0, canvas)
    canvas[y1 : y1 + ph, x1 : x1 + pw] = preview
    cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x1 + pw + 3, y1 + ph + 3), (0, 200, 255), 2)
    return canvas


def add_overlays(frame, frame_idx, total_frames,
                 emoji_laugh, emoji_fire, watermark_text="@YourChannel"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    caption = "Comment the best part of the video"
    font_scale = w / 800.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(caption, font, font_scale, thickness)

    emoji_size = int(th * 2.2)
    laugh_resized = cv2.resize(
        emoji_laugh, (emoji_size, emoji_size),
        interpolation=cv2.INTER_LANCZOS4
    ) if emoji_laugh is not None else None

    gap = 10
    total_width = tw + (emoji_size + gap if laugh_resized is not None else 0)
    start_x = (w - total_width) // 2
    caption_y = h - int(h * 0.08)

    pad = 14
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (start_x - pad, caption_y - th - pad),
                  (start_x + total_width + pad, caption_y + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    add_text_with_outline(frame, caption, (start_x, caption_y),
                          font_scale, (255, 255, 255), thickness)

    if laugh_resized is not None:
        ex = start_x + tw + gap + emoji_size // 2
        ey = caption_y - th // 2
        frame = overlay_emoji(frame, laugh_resized, ex, ey)

    if emoji_laugh is not None:
        center_size = int(min(w, h) * 0.15)
        center_emoji = cv2.resize(emoji_laugh, (center_size, center_size),
                                  interpolation=cv2.INTER_LANCZOS4)
        frame = overlay_emoji(frame, center_emoji, w // 2, h // 2)

    wm_scale = w / 900.0
    (wmw, wmh), _ = cv2.getTextSize(watermark_text, font, wm_scale, 2)
    wm_overlay = frame.copy()
    cv2.putText(wm_overlay, watermark_text,
                ((w - wmw) // 2, h // 2 - int(h * 0.12)),
                font, wm_scale, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.addWeighted(wm_overlay, 0.25, frame, 0.75, 0, frame)

    badge_text = "Laughify Hub"
    badge_scale = w / 1250.0
    badge_thickness = 2
    (bdw, bdh), _ = cv2.getTextSize(badge_text, font, badge_scale, badge_thickness)
    badge_pad = 12
    bx1 = w - bdw - badge_pad * 2 - 24
    by1 = 24
    bx2 = w - 24
    by2 = by1 + bdh + badge_pad * 2
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
    cv2.addWeighted(badge_overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
    add_text_with_outline(
        frame,
        badge_text,
        (bx1 + badge_pad, by2 - badge_pad),
        badge_scale,
        (255, 255, 255),
        badge_thickness,
    )

    progress = int((frame_idx / max(total_frames, 1)) * w)
    cv2.rectangle(frame, (0, h - 6), (progress, h), (0, 200, 255), -1)

    alpha = 0.6 if (frame_idx // 15) % 2 == 0 else 0.3
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, int(h * 0.10)), (0, 0, 0), -1)
    cv2.addWeighted(banner, alpha, frame, 1 - alpha, 0, frame)

    top_text = "Follow for more videos like this"
    b_scale = w / 900.0
    (btw, bth), _ = cv2.getTextSize(top_text, font, b_scale, 2)
    bx = (w - btw) // 2
    by = int(h * 0.075)

    if emoji_fire is not None:
        f_size = int(bth * 2)
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
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 0.5 or fps > 240:
        fps = probe_fps_ffprobe(input_path) or 0.0
    if fps < 0.5 or fps > 240:
        fps = 30.0
    return fps


def probe_has_audio(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return "audio" in out


def _audio_work_paths(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(AUDIO_WORK_DIR, exist_ok=True)
    raw_path = os.path.join(AUDIO_WORK_DIR, f"{base}_raw.wav")
    remixed_path = os.path.join(AUDIO_WORK_DIR, f"{base}_remixed.wav")
    return raw_path, remixed_path


def extract_audio(input_path, wav_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "2",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Extracted audio: {wav_path}")


def build_audio_remix_filter(pitch=PITCH_SHIFT, original_mix=ORIGINAL_MIX, effects_mix=EFFECTS_MIX):
    return (
        f"[0:a]"
        f"highpass=f=120,lowpass=f=9000,volume={original_mix}[orig];"
        f"[0:a]"
        f"rubberband=pitch={pitch},"
        f"aecho=0.75:0.72:55|110:0.42|0.28,"
        f"aecho=0.55:0.58:320|640|960:0.22|0.16|0.10,"
        f"asoftclip=type=tanh:param=0.72,"
        f"acompressor=threshold=-16dB:ratio=4:attack=8:release=120,"
        f"flanger=delay=8:depth=4:speed=0.35,"
        f"volume={effects_mix}[fx];"
        f"[orig][fx]amix=inputs=2:duration=first:dropout_transition=0,"
        f"alimiter=limit=0.95,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[outa]"
    )


def remix_audio(raw_wav_path, remixed_wav_path, pitch=PITCH_SHIFT):
    filter_graph = build_audio_remix_filter(pitch=pitch)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        raw_wav_path,
        "-filter_complex",
        filter_graph,
        "-map",
        "[outa]",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        remixed_wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Remixed audio: {remixed_wav_path}")


def prepare_remixed_audio(input_path):
    raw_path, remixed_path = _audio_work_paths(input_path)
    extract_audio(input_path, raw_path)
    remix_audio(raw_path, remixed_path)
    return remixed_path


def _transform_video_with_emojis(input_path, output_path, watermark, emoji_laugh, emoji_fire):
    cap = cv2.VideoCapture(input_path)
    fps = _resolve_fps(cap, input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width, output_height = OUTPUT_SIZE

    temp_video = "temp_no_audio.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (output_width, output_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.02) : int(h * 0.98), int(w * 0.02) : int(w * 0.98)]
        frame = cv2.resize(crop, (w, h))
        frame = apply_dynamic_reframe(frame, frame_idx, total)
        # frame = apply_wave_distortion(frame, frame_idx)
        noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frame = cv2.convertScaleAbs(
            frame,
            alpha=1.16 + 0.08 * np.sin(frame_idx / 19.0),
            beta=10 + int(6 * np.cos(frame_idx / 27.0)),
        )
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 1, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))
        # frame = cv2.flip(frame, 1)  # Mirror the frame (currently commented out)
        # frame = apply_color_grade(frame, frame_idx)
        frame = apply_blur_and_sharpen(frame, frame_idx)
        stylized_main = frame.copy()
        frame = apply_vignette(frame)
        frame = apply_scanlines(frame, frame_idx)
        frame = build_vertical_canvas(frame, frame_idx)
        # frame = add_rotated_preview(frame, stylized_main, frame_idx)
        frame = add_styled_border(frame, frame_idx)
        frame = add_overlays(frame, frame_idx, total, emoji_laugh, emoji_fire, watermark)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processing video... {frame_idx}/{total} frames")

    cap.release()
    out.release()

    remixed_audio = None
    if probe_has_audio(input_path):
        print("Processing audio...")
        remixed_audio = prepare_remixed_audio(input_path)
    else:
        print("No audio track found; exporting video only.")

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
    ]
    if remixed_audio:
        cmd.extend(["-i", remixed_audio])
    cmd.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if remixed_audio:
        cmd.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(output_path)
    subprocess.run(cmd, check=True)
    os.remove(temp_video)
    print(f"\n✅ Done! Saved to: {output_path}")


def transform_video(input_path, output_path, watermark="@YourChannel"):
    prepare_emojis()
    emoji_laugh = load_emoji("emojis/laugh.png", 128) if os.path.exists("emojis/laugh.png") else None
    emoji_fire = load_emoji("emojis/fire.png", 128) if os.path.exists("emojis/fire.png") else None
    _transform_video_with_emojis(input_path, output_path, watermark, emoji_laugh, emoji_fire)


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def list_videos(folder):
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


if __name__ == "__main__":
    transform_folder(
        input_dir="Downloaded_Videos",
        # input_dir="done_input",
        output_dir="output",
        watermark="Laughify Hub",
    )
