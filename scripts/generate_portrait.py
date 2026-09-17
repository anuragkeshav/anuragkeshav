from pathlib import Path
from html import escape

import cv2
import numpy as np
from PIL import Image
from rembg import remove


# ============================================================
# SETTINGS
# ============================================================

COLS = 90
DISPLAY_WIDTH = 700

FONT_SIZE = 12.9
CHAR_W = 7.74

# Contrast
CLAHE_CLIP = 2.4

# Slightly darker shadows without crushing the face
GAMMA = 1.25

# ASCII brightness ramp
# Bright -> space
# Dark -> dense
RAMP = " .:-=+*#%@"

INPUT_FILE = Path("input/portrait.jpg")
OUTPUT_FILE = Path("assets/portrait.svg")


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading portrait...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing image: {INPUT_FILE}"
    )

image = Image.open(
    INPUT_FILE
).convert("RGBA")

img_w, img_h = image.size

print(
    f"Original image: "
    f"{img_w} x {img_h}"
)


# ============================================================
# REMOVE BACKGROUND
# ============================================================

print("Removing background...")

rgba = np.array(image)

removed = remove(rgba)

rgb = removed[:, :, :3]
alpha = removed[:, :, 3]


# ============================================================
# PORTRAIT CROP
# ============================================================
#
# The uploaded portrait is square.
#
# We intentionally crop:
#
#   - a little from left/right
#   - very little from top
#   - almost all of the shoulders
#
# This makes the FACE occupy much more of
# the ASCII grid.
# ============================================================

left = int(img_w * 0.10)
right = int(img_w * 0.90)

top = int(img_h * 0.02)
bottom = int(img_h * 0.99)

rgb = rgb[
    top:bottom,
    left:right
]

alpha = alpha[
    top:bottom,
    left:right
]

crop_h, crop_w = rgb.shape[:2]

print(
    f"Portrait crop: "
    f"{crop_w} x {crop_h}"
)


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
)


# ============================================================
# ASCII SIZE
# ============================================================

# The 0.48 correction compensates for the fact
# that monospace characters are taller than wide.

ROWS = max(
    1,
    int(
        COLS
        * (crop_h / crop_w)
        * 0.48
    )
)

# Keep facial detail in a useful range.
ROWS = max(
    52,
    min(
        ROWS,
        62
    )
)

print(
    f"ASCII grid: "
    f"{COLS} x {ROWS}"
)


# ============================================================
# RESIZE
# ============================================================

gray = cv2.resize(
    gray,
    (COLS, ROWS),
    interpolation=cv2.INTER_LANCZOS4
)

alpha_small = cv2.resize(
    alpha,
    (COLS, ROWS),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# BILATERAL FILTER
# ============================================================

print("Smoothing while preserving facial edges...")

gray = cv2.bilateralFilter(
    gray,
    5,
    35,
    35
)


# ============================================================
# CLAHE
# ============================================================

print("Enhancing local facial contrast...")

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP,
    tileGridSize=(6, 6)
)

gray = clahe.apply(gray)


# ============================================================
# UNSHARP MASK
# ============================================================
#
# Helps preserve:
#   eyes
#   eyebrows
#   nose edge
#   lips
#   jaw
#
# ============================================================

blur = cv2.GaussianBlur(
    gray,
    (0, 0),
    1.2
)

gray = cv2.addWeighted(
    gray,
    1.35,
    blur,
    -0.35,
    0
)

gray = np.clip(
    gray,
    0,
    255
).astype(np.uint8)


# ============================================================
# GAMMA / SHADOW CONTROL
# ============================================================

print("Balancing shadows and skin tones...")

v = (
    gray.astype(np.float32)
    / 255.0
)

v = np.power(
    v,
    GAMMA
)

gray = np.clip(
    v * 255.0,
    0,
    255
).astype(np.uint8)


# ============================================================
# BACKGROUND MASK
# ============================================================
#
# Everything outside the person becomes empty.
#
# This is what allows GitHub's dark background
# to show through.
# ============================================================

background = alpha_small < 35

gray[background] = 255


# ============================================================
# ASCII CONVERSION
# ============================================================

print("Converting portrait to ASCII...")

ascii_rows = []

for y in range(ROWS):

    row = []

    for x in range(COLS):

        # Transparent background
        if alpha_small[y, x] < 35:
            row.append(" ")
            continue

        brightness = int(
            gray[y, x]
        )

        # Convert brightness to ASCII index.
        #
        # 255 = bright = space
        # 0   = dark  = @

        index = int(
            (
                (255 - brightness)
                / 255.0
            )
            * (len(RAMP) - 1)
        )

        index = max(
            0,
            min(
                len(RAMP) - 1,
                index
            )
        )

        row.append(
            RAMP[index]
        )

    ascii_rows.append(
        "".join(row)
    )


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

def has_content(row):
    return any(
        c != " "
        for c in row
    )


while (
    ascii_rows
    and not has_content(
        ascii_rows[0]
    )
):
    ascii_rows.pop(0)


while (
    ascii_rows
    and not has_content(
        ascii_rows[-1]
    )
):
    ascii_rows.pop()


# ============================================================
# FIND CONTENT BOUNDS
# ============================================================

if not ascii_rows:
    raise RuntimeError(
        "ASCII conversion produced no visible content."
    )


min_x = len(ascii_rows[0])
max_x = 0

for row in ascii_rows:

    for x, char in enumerate(row):

        if char != " ":

            min_x = min(
                min_x,
                x
            )

            max_x = max(
                max_x,
                x
            )


# ============================================================
# CROP EMPTY HORIZONTAL SPACE
# ============================================================

padding = 2

min_x = max(
    0,
    min_x - padding
)

max_x = min(
    COLS - 1,
    max_x + padding
)

ascii_rows = [
    row[min_x:max_x + 1]
    for row in ascii_rows
]


# ============================================================
# FINAL DIMENSIONS
# ============================================================

FINAL_COLS = max(
    len(row)
    for row in ascii_rows
)

FINAL_ROWS = len(
    ascii_rows
)

CONTENT_WIDTH = (
    FINAL_COLS * CHAR_W
)

# Space around portrait
SIDE_PADDING = 80

SVG_WIDTH = (
    CONTENT_WIDTH
    + SIDE_PADDING * 2
)

SVG_HEIGHT = (
    FINAL_ROWS
    * FONT_SIZE
    * 1.05
)

CENTER_X = (
    SVG_WIDTH / 2
)


# ============================================================
# CREATE SVG
# ============================================================

print("Creating centered SVG...")

svg = []

svg.append(
    '<?xml version="1.0" encoding="UTF-8"?>'
)

svg.append(
    f'<svg '
    f'xmlns="http://www.w3.org/2000/svg" '
    f'width="{DISPLAY_WIDTH}" '
    f'viewBox="0 0 '
    f'{SVG_WIDTH:.2f} '
    f'{SVG_HEIGHT:.2f}" '
    f'preserveAspectRatio="xMidYMid meet">'
)


# ============================================================
# CLIP PATHS
# ============================================================

svg.append("<defs>")

for i, row in enumerate(ascii_rows):

    row_width = (
        len(row) * CHAR_W
    )

    row_left = (
        CENTER_X
        - row_width / 2
    )

    y = (
        i * FONT_SIZE
    )

    delay = (
        i * 0.09
    )

    clip_id = (
        f"row{i}"
    )

    svg.append(
        f'<clipPath id="{clip_id}">'
    )

    svg.append(
        f'<rect '
        f'x="{row_left:.2f}" '
        f'y="{y:.2f}" '
        f'width="0" '
        f'height="{FONT_SIZE * 1.4:.2f}">'
    )

    svg.append(
        f'<animate '
        f'attributeName="width" '
        f'from="0" '
        f'to="{row_width:.2f}" '
        f'dur="0.75s" '
        f'begin="{delay:.2f}s" '
        f'fill="freeze"/>'
    )

    svg.append(
        '</rect>'
    )

    svg.append(
        '</clipPath>'
    )

svg.append("</defs>")


# ============================================================
# TEXT GROUP
# ============================================================

svg.append(
    '<g '
    'font-family="DejaVu Sans Mono, '
    'Liberation Mono, monospace" '
    f'font-size="{FONT_SIZE}px" '
    'font-weight="400" '
    'fill="#E6EDF3" '
    'text-anchor="middle" '
    'xml:space="preserve">'
)


# ============================================================
# DRAW EVERY ROW
# ============================================================

for i, row in enumerate(ascii_rows):

    y = (
        (i + 1)
        * FONT_SIZE
    )

    safe_row = escape(
        row,
        quote=False
    )

    svg.append(
        f'<text '
        f'x="{CENTER_X:.2f}" '
        f'y="{y:.2f}" '
        f'clip-path="url(#row{i})">'
        f'{safe_row}'
        f'</text>'
    )


# ============================================================
# CLOSE SVG
# ============================================================

svg.append("</g>")
svg.append("</svg>")


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.write_text(
    "\n".join(svg),
    encoding="utf-8"
)


# ============================================================
# DONE
# ============================================================

print()
print(
    "=============================================="
)

print(
    "       FACE-FOCUSED ASCII PORTRAIT"
)

print(
    "=============================================="
)

print(
    f"Input:       {INPUT_FILE}"
)

print(
    f"Original:    {img_w} x {img_h}"
)

print(
    f"Crop:        {crop_w} x {crop_h}"
)

print(
    f"ASCII:       {FINAL_COLS} x {FINAL_ROWS}"
)

print(
    f"SVG width:   {SVG_WIDTH:.1f}"
)

print(
    f"Display:     {DISPLAY_WIDTH}px"
)

print(
    "Background:  TRANSPARENT"
)

print(
    "Text:        GitHub dark compatible"
)

print(
    "Centered:    YES"
)

print(
    "Animation:   ONE-TIME"
)

print(
    f"Output:      {OUTPUT_FILE}"
)

print(
    "=============================================="
)
