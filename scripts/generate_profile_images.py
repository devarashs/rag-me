"""Generate the page's avatar, favicons and link-preview image from one photo.

Usage (Pillow is only needed here, so it is not a project dependency):

    uv run --with pillow python scripts/generate_profile_images.py images/pfp.JPG

Writes into web/, which Vercel serves from its CDN:

- avatar.webp           160px square, for the 52px header avatar at up to 3x DPR.
- favicon.ico           16/32/48px round crop, for browser tabs.
- icon-192.png          192px round crop, the modern <link rel="icon">.
- apple-touch-icon.png  180px square (iOS masks its own corners, and draws
                        transparency as black, so this one stays opaque).
- og-image.jpg          1200x630 link preview for LinkedIn, Slack, WhatsApp.

Output carries no EXIF or other metadata, so nothing from the camera is published.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

# Crop boxes as fractions of the source square (left, top, right, bottom), tuned
# for the current photo: the face sits near the centre. "portrait" keeps head and
# shoulders for larger images; "face" is tighter so a 16px favicon is still a face.
PORTRAIT_CROP = (0.21, 0.24, 0.79, 0.82)
FACE_CROP = (0.30, 0.28, 0.70, 0.68)

# Brand colours from web/styles.css (the indigo gradient and slate text).
GRADIENT_START = (85, 88, 232)
GRADIENT_END = (122, 79, 224)
WHITE = (255, 255, 255)
SOFT_WHITE = (226, 228, 250)


def crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop(
        (round(left * width), round(top * height), round(right * width), round(bottom * height))
    )


def square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def round_crop(image: Image.Image, size: int) -> Image.Image:
    """Circular crop with antialiased edges, drawn at 4x and scaled down."""
    scale = 4
    mask = Image.new("L", (size * scale, size * scale), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * scale - 1, size * scale - 1), fill=255)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    result = square(image, size).convert("RGBA")
    result.putalpha(mask)
    return result


def diagonal_gradient(width: int, height: int) -> Image.Image:
    gradient = Image.new("RGB", (width, height))
    pixels = gradient.load()
    span = width + height
    for y in range(height):
        for x in range(width):
            t = (x + y) / span
            pixels[x, y] = tuple(
                round(start + (end - start) * t)
                for start, end in zip(GRADIENT_START, GRADIENT_END, strict=True)
            )
    return gradient


def load_font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Windows and common Linux font paths; falls back to Pillow's built-in font.
    candidates = (
        ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def build_link_preview(portrait: Image.Image) -> Image.Image:
    """1200x630 card: brand gradient, round photo, name, role and call to action."""
    card = diagonal_gradient(1200, 630)
    photo = round_crop(portrait, 340)
    ring = Image.new("RGBA", (360, 360), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, 359, 359), fill=(255, 255, 255, 70))
    card.paste(ring, (90, 135), ring)
    card.paste(photo, (100, 145), photo)

    draw = ImageDraw.Draw(card)
    draw.text((510, 205), "Arash Salehkhah", font=load_font(66, bold=True), fill=WHITE)
    draw.text(
        (512, 290), "Senior Backend Engineer", font=load_font(38, bold=False), fill=SOFT_WHITE
    )
    draw.text(
        (512, 375),
        "Ask anything about my work,\nanswered from my own notes.",
        font=load_font(32, bold=False),
        fill=WHITE,
        spacing=10,
    )
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photo", type=Path, help="square-ish source photo")
    source = ImageOps.exif_transpose(Image.open(parser.parse_args().photo)).convert("RGB")

    portrait = crop_fraction(source, PORTRAIT_CROP)
    face = crop_fraction(source, FACE_CROP)

    square(portrait, 160).save(WEB_DIR / "avatar.webp", quality=82, method=6)
    round_crop(face, 48).save(WEB_DIR / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    round_crop(face, 192).save(WEB_DIR / "icon-192.png", optimize=True)
    square(face, 180).save(WEB_DIR / "apple-touch-icon.png", optimize=True)
    build_link_preview(portrait).save(
        WEB_DIR / "og-image.jpg", quality=85, optimize=True, progressive=True
    )

    for name in (
        "avatar.webp",
        "favicon.ico",
        "icon-192.png",
        "apple-touch-icon.png",
        "og-image.jpg",
    ):
        print(f"{name:22} {(WEB_DIR / name).stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
