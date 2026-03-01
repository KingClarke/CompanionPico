import cv2
import numpy as np
import os
from pathlib import Path


#Convert Bitmaps to CPP Byte arrays (.cpp)
# #Black and White, colors & effects done at runtime
# 12fps * 8 faces = 96 frames
# *2 variations = 192 frames
# *2 idles & easter eggs = 382 frames
# *2.34kb per-frame = 894kb
# +296kb .u2f = 1.19mb
# 160*120 = 2400*1.024 = 2.34kb


INPUT_ROOT = Path("faces")
OUT_CPP = "faces_data.cpp"
OUT_H   = "faces_data.h"

WIDTH  = 160
HEIGHT = 120

def bmp_to_packed_bytes(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")

    if img.shape != (HEIGHT, WIDTH):
        raise RuntimeError(f"{path} is {img.shape}, expected {HEIGHT}x{WIDTH}")

    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    packed = np.packbits(bw, axis=1)
    return packed.tobytes()

cpp_lines = []
h_lines = []

cpp_lines.append('#include "faces_data.h"\n')
h_lines.append('#pragma once\n#include <stdint.h>\n')

for category in sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir()):
    cat_name = category.name
    frames = sorted(category.glob("*.bmp"))

    h_lines.append(f'extern const uint8_t* {cat_name}_frames[];')
    h_lines.append(f'extern const uint16_t {cat_name}_frame_count;\n')

    cpp_lines.append(f'// ===== {cat_name.upper()} =====')

    frame_names = []

    for i, bmp in enumerate(frames):
        data = bmp_to_packed_bytes(bmp)
        arr_name = f"{cat_name}_{i}"
        frame_names.append(arr_name)

        cpp_lines.append(f'const uint8_t {arr_name}[] = {{')

        for j in range(0, len(data), 12):
            chunk = data[j:j+12]
            cpp_lines.append(
                "  " + ", ".join(f"0x{b:02X}" for b in chunk) + ","
            )

        cpp_lines.append("};\n")

    cpp_lines.append(f'const uint8_t* {cat_name}_frames[] = {{')
    for name in frame_names:
        cpp_lines.append(f'  {name},')
    cpp_lines.append('};')

    cpp_lines.append(f'const uint16_t {cat_name}_frame_count = {len(frame_names)};\n')

with open(OUT_CPP, "w") as f:
    f.write("\n".join(cpp_lines))

with open(OUT_H, "w") as f:
    f.write("\n".join(h_lines))

print(" faces converted successfully :3")
