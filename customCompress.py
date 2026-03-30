# 120 * 160 rgb -> 19,200
# 19,200 * 0.5 byte -> 9.375kb

# 240 * 320 -> 38,400
# 76,800 * 0.5 byte -> 37.5kb

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import os

def sanitize_name(filename):
    # make safe C++ variable name, upper case, replace invalid chars with _
    name = os.path.splitext(os.path.basename(filename))[0]
    name = ''.join(c if c.isalnum() else '_' for c in name)
    return f"IMG_{name.upper()}"

def convert_to_pico_header(image_path, n_colors=16, saturation_boost=1.5, sample_scale=0.2):
    header_name = sanitize_name(image_path)

    # 1. Load and Resize for Pico (160x120)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pico = cv2.resize(img_rgb, (160, 120), interpolation=cv2.INTER_AREA)

    # 2. Boost Saturation & Brightness (HSV)
    hsv = cv2.cvtColor(img_pico, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_boost
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] *= 1.2
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 3. Downsample for Clustering
    small = cv2.resize(boosted, (0, 0), fx=sample_scale, fy=sample_scale)
    small_pixels = small.reshape((-1, 3))

    # 4. Extract Palette
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=383, n_init=3).fit(small_pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    palette[0] = [0, 0, 0]  # force first slot black

    # 5. Dithering
    flat_palette = palette.flatten().tolist()
    flat_palette += [0] * (768 - len(flat_palette))
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)
    pi_img = Image.fromarray(boosted)
    dithered_pi = pi_img.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
    indices = np.array(dithered_pi).flatten()

    # 6. Convert Palette to RGB565
    palette_565 = []
    for color in palette:
        r, g, b = color
        rgb565 = ((int(r) & 0xF8) << 8) | ((int(g) & 0xFC) << 3) | (int(b) >> 3)
        palette_565.append(f"0x{rgb565:04X}")

    # 7. Pack Pixels (4-bit)
    packed_data = []
    for i in range(0, len(indices), 2):
        byte = (indices[i] << 4) | (indices[i + 1] & 0x0F)
        packed_data.append(f"0x{byte:02X}")

    # 8. Write Header with struct
    with open(f"{header_name}.h", "w") as f:
        f.write(f"#ifndef {header_name.upper()}_H\n#define {header_name.upper()}_H\n#include <stdint.h>\n\n")
        f.write("struct Image {\n")
        f.write("    const uint16_t* palette;\n")
        f.write("    const uint8_t* data;\n")
        f.write("    int width;\n")
        f.write("    int height;\n};\n\n")
        f.write(f"const uint16_t {header_name}_palette[16] = {{{', '.join(palette_565)}}};\n\n")
        f.write(f"const uint8_t {header_name}_data[{len(packed_data)}] = {{\n    ")
        for i, b in enumerate(packed_data):
            f.write(b + (", " if i < len(packed_data) - 1 else ""))
            if (i + 1) % 16 == 0: f.write("\n    ")
        f.write("\n};\n\n")
        f.write(f"const Image {header_name} = {{ {header_name}_palette, {header_name}_data, 160, 120 }};\n\n")
        f.write("#endif\n")

    print(f"Saved {header_name}.h ({len(packed_data) / 1024:.2f} KB)")

# Example usage:
convert_to_pico_header("input.png")