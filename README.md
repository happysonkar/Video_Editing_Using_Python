# Video Editor (`edit_video.py`)

Batch-process short-form videos with light visual transforms and social-style overlays (captions, emojis, watermark, progress bar, and a top hype banner). Outputs MP4 files with the original audio preserved.

## Features

### Visual transforms (every frame)

- Horizontal mirror flip
- Slight crop and resize back to original dimensions
- Subtle film grain (random noise)
- Brightness/contrast boost
- ~1° rotation

### Overlays

| Element | Description |
|--------|-------------|
| Bottom caption | Pill background, white outlined text (“Wait for the ending”) + laugh emoji |
| Center emoji | Large laugh emoji in the middle of the frame |
| Watermark | Semi-transparent channel handle (configurable) |
| Progress bar | Cyan bar along the bottom edge |
| Top banner | Blinking dark bar (first ~3 seconds) with “Follow for more videos like this”, fire emojis, and cyan text |

Emoji images are downloaded automatically on first run into an `emojis/` folder (laugh and fire PNGs).

### Audio & timing

- Frame rate is taken from the source video (OpenCV, then `ffprobe` if needed; defaults to 30 fps)
- Video is written via OpenCV, then remuxed with **ffmpeg** so audio stays in sync
- Original audio is copied when possible; falls back to AAC re-encode if the source codec is incompatible with MP4

## Requirements

- **Python 3.8+**
- **ffmpeg** and **ffprobe** on your `PATH`

```bash
pip install opencv-python numpy Pillow
```

On Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

## Project layout

```
edit/
├── edit_video.py    # Main script
├── input/           # Place source videos here
├── output/          # Edited videos written here as <name>_edited.mp4
└── emojis/          # Created automatically (laugh.png, fire.png)
```

Supported input extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.m4v`

## Usage

### Default (batch folder)

The script’s `__main__` block processes every video in `input/` and writes to `output/`:

```python
transform_folder(
    input_dir="input",
    output_dir="output",
    watermark="ComedyCentral_TV",
)
```

Run from the project directory:

```bash
mkdir -p input output
# Copy your videos into input/
python edit_video.py
```

### Single file

```python
from edit_video import transform_video

transform_video(
    input_path="input/my_clip.mp4",
    output_path="output/my_clip_edited.mp4",
    watermark="@YourChannel",
)
```

### Batch folder (programmatic)

```python
from edit_video import transform_folder

transform_folder(
    input_dir="path/to/videos",
    output_dir="path/to/output",
    watermark="@YourChannel",
)
```

## Customization

Edit these in `edit_video.py` as needed:

| What | Where |
|------|--------|
| Watermark text | `watermark` argument to `transform_video` / `transform_folder`, or the call at the bottom of the file |
| Caption, banner text, colors | `add_overlays()` |
| Crop, noise, rotation strength | `_transform_video_with_emojis()` loop |
| Emoji URLs | `prepare_emojis()` → `png_urls` dict |

To change input/output folders without editing the file, import and call `transform_folder()` from your own script or REPL.

## Output

- Filename pattern: `<original_basename>_edited.mp4`
- Codec: H.264 (`libx264`), `yuv420p`, `+faststart` for web playback
- A temporary `temp_no_audio.avi` is created during processing and removed when ffmpeg finishes

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `ffmpeg` / `ffprobe` not found | Install ffmpeg and ensure it is on `PATH` |
| No videos processed | Check that files are in `input/` and use a supported extension |
| Audio missing or out of sync | Confirm source has an audio track; script uses source FPS for muxing |
| ffmpeg audio copy fails | Script automatically retries with AAC 192k re-encode |

## License note

Emoji assets are fetched from public URLs (Twitter CDN for laugh, OpenMoji for fire). Verify licensing fits your use case if you redistribute edited videos commercially.
