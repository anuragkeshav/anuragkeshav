from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


# ============================================================
# SETTINGS
# ============================================================

COLS = 90
DISPLAY_WIDTH = 460

# Monospace character measurements
CHAR_W = 7.74
FONT_SIZE = 12.9

# Image processing
CLAHE_CLIP = 3.0
DARKEN_POWER = 1.7

# ASCII ramp
# Bright pixels -> light characters
# Dark pixels   -> dense characters
RAMP = " .:-=+*#%@"

INPUT_FILE = Path("input/portrait.jpg")
OUTPUT_FILE = Path("assets/portrait.svg")


# ============================================================
# 1. LOAD IMAGE
# ============================================================

print("Loading image...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input image not found: {INPUT_FILE}"
    )

image = Image.open(INPUT_FILE).convert("RGBA")

original_width, original_height = image.size

print(
    f"Original image: "
    f"{original_width} x {original_height}"
)


# ============================================================
# 2. REMOVE BACKGROUND
# ============================================================

print("Removing background...")

rgba = np.array(image)

result = remove(rgba)

# RGB + alpha
rgb = result[:, :, :3]
alpha = result[:, :, 3]


# ============================================================
# 3. FIND SUBJECT BOUNDING BOX
# ============================================================

print("Detecting subject...")

ys, xs = np.where(alpha > 20)

if len(xs) == 0 or len(ys) == 0:
    raise RuntimeError(
        "No subject detected by rembg."
    )

x1 = int(xs.min())
y1 = int(ys.min())
x2 = int(xs.max()) + 1
y2 = int(ys.max()) + 1


# ============================================================
# 4. ADD PADDING AROUND SUBJECT
# ============================================================

subject_width = x2 - x1
subject_height = y2 - y1

pad_x = int(subject_width * 0.08)
pad_y = int(subject_height * 0.08)

x1 = max(0, x1 - pad_x)
y1 = max(0, y1 - pad_y)

x2 = min(rgb.shape[1], x2 + pad_x)
y2 = min(rgb.shape[0], y2 + pad_y)


# Crop both image and alpha
rgb = rgb[y1:y2, x1:x2]
alpha = alpha[y1:y2, x1:x2]

print(
    f"Subject crop: "
    f"{rgb.shape[1]} x {rgb.shape[0]}"
)


# ============================================================
# 5. COMPOSITE SUBJECT ON WHITE
# ============================================================

print("Creating white background...")

white = np.full_like(rgb, 255)

alpha_float = alpha.astype(np.float32) / 255.0

composite = (
    rgb.astype(np.float32)
    * alpha_float[:, :, None]
    +
    white.astype(np.float32)
    * (1.0 - alpha_float[:, :, None])
)

composite = np.clip(
    composite,
    0,
    255
).astype(np.uint8)


# ============================================================
# 6. GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    composite,
    cv2.COLOR_RGB2GRAY
)


# ============================================================
# 7. CALCULATE ASCII SIZE
# ============================================================

width = gray.shape[1]
height = gray.shape[0]

# Characters are narrower than they are tall.
# This compensates for the monospace character shape.

CHAR_ASPECT = CHAR_W / FONT_SIZE

ROWS = max(
    1,
    int(
        COLS
        * (height / width)
        * CHAR_ASPECT
    )
)

print(
    f"ASCII grid: "
    f"{COLS} columns x {ROWS} rows"
)


# ============================================================
# 8. RESIZE IMAGE TO ASCII GRID
# ============================================================

print("Resizing image...")

gray = cv2.resize(
    gray,
    (COLS, ROWS),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# 9. BILATERAL FILTER
# ============================================================

print("Applying bilateral filter...")

gray = cv2.bilateralFilter(
    gray,
    d=5,
    sigmaColor=50,
    sigmaSpace=50
)


# ============================================================
# 10. CLAHE
# ============================================================

print("Applying CLAHE...")

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP,
    tileGridSize=(8, 8)
)

gray = clahe.apply(gray)


# ============================================================
# 11. DARKENING CURVE
# ============================================================

print("Applying darkening curve...")

normalized = (
    gray.astype(np.float32) / 255.0
)

normalized = np.power(
    normalized,
    DARKEN_POWER
)

gray = np.clip(
    normalized * 255.0,
    0,
    255
).astype(np.uint8)


# ============================================================
# 12. CONVERT IMAGE TO ASCII
# ============================================================

print("Converting to ASCII...")

ascii_rows = []

for y in range(ROWS):

    row_chars = []

    for x in range(COLS):

        brightness = int(
            gray[y, x]
        )

        # Reverse brightness:
        # bright -> beginning of RAMP
        # dark   -> end of RAMP

        index = int(
            (255 - brightness)
            / 255.0
            * (len(RAMP) - 1)
        )

        index = max(
            0,
            min(
                len(RAMP) - 1,
                index
            )
        )

        row_chars.append(
            RAMP[index]
        )

    ascii_rows.append(
        "".join(row_chars)
    )


# ============================================================
# 13. CREATE SVG DIMENSIONS
# ============================================================

print("Creating SVG...")

svg_width = COLS * CHAR_W
svg_height = ROWS * FONT_SIZE * 1.05


# ============================================================
# 14. START SVG
# ============================================================

svg = []

svg.append(
    '<?xml version="1.0" encoding="UTF-8"?>'
)

svg.append(
    f'<svg '
    f'xmlns="http://www.w3.org/2000/svg" '
    f'width="{DISPLAY_WIDTH}" '
    f'viewBox="0 0 {svg_width:.2f} {svg_height:.2f}" '
    f'preserveAspectRatio="xMidYMid meet" '
    f'role="img">'
)


# ============================================================
# 15. WHITE BACKGROUND
# ============================================================

svg.append(
    '<rect '
    'x="0" '
    'y="0" '
    'width="100%" '
    'height="100%" '
    'fill="white"/>'
)


# ============================================================
# 16. SVG DEFINITIONS
# ============================================================

svg.append("<defs>")


for i in range(ROWS):

    clip_id = f"rowClip{i}"

    y = i * FONT_SIZE

    delay = i * 0.09

    svg.append(
        f'<clipPath id="{clip_id}">'
    )

    svg.append(
        f'<rect '
        f'x="0" '
        f'y="{y:.2f}" '
        f'width="0" '
        f'height="{FONT_SIZE * 1.3:.2f}">'
    )

    svg.append(
        f'<animate '
        f'attributeName="width" '
        f'from="0" '
        f'to="{svg_width:.2f}" '
        f'dur="0.8s" '
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
# 17. ASCII TEXT GROUP
# ============================================================

svg.append(
    '<g '
    'font-family="DejaVu Sans Mono, Liberation Mono, monospace" '
    f'font-size="{FONT_SIZE}px" '
    'font-weight="400" '
    'fill="black" '
    'xml:space="preserve">'
)


# ============================================================
# 18. ADD ASCII ROWS
# ============================================================

for i, row in enumerate(ascii_rows):

    y = (i + 1) * FONT_SIZE

    clip_id = f"rowClip{i}"

    # Escape XML-sensitive characters
    row = (
        row
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    svg.append(
        f'<text '
        f'x="0" '
        f'y="{y:.2f}" '
        f'clip-path="url(#{clip_id})">'
        f'{row}'
        f'</text>'
    )


# ============================================================
# 19. CLOSE SVG
# ============================================================

svg.append("</g>")
svg.append("</svg>")


# ============================================================
# 20. SAVE SVG
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
# 21. DONE
# ============================================================

print()
print("==============================================")
print("       PORTRAIT GENERATED SUCCESSFULLY")
print("==============================================")
print(f"Input:          {INPUT_FILE}")
print(f"Output:         {OUTPUT_FILE}")
print(
    f"Original size:  "
    f"{original_width} x {original_height}"
)
print(
    f"Subject crop:   "
    f"{width} x {height}"
)
print(f"ASCII columns:  {COLS}")
print(f"ASCII rows:     {ROWS}")
print(
    f"Display width:  "
    f"{DISPLAY_WIDTH}px"
)
print("Animation:      One-time typing")
print("==============================================")
