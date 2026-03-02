import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

# 120 * 160 rgb -> 19,200
# 19,200 * 0.5 byte -> 9.375kb

# 240 * 320 -> 38,400
# 76,800 * 0.5 byte -> 37.5kb

def convert_to_pico_header(image_path, header_name, n_colors=16, saturation_boost=1.5):
    # 1. Load and Boost Saturation
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to our target resolution (160x120)
    img_rgb = cv2.resize(img_rgb, (320, 240), interpolation=cv2.INTER_LANCZOS4)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_boost
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Extract Best 16-Color Palette
    pixels = boosted.reshape((-1, 3))
    kmeans = MiniBatchKMeans(n_clusters=n_colors,
                             random_state=383,
                             n_init=3).fit(pixels)
    palette = np.uint8(kmeans.cluster_centers_)

    # 3. Floyd-Steinberg Dithering using Pillow
    # Create a palette image template (Pillow expects 768 values for a 256-color palette)
    flat_palette = palette.flatten().tolist()
    flat_palette += [0] * (768 - len(flat_palette))

    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)

    # Convert boosted RGB to Pillow and Quantize (Dither)
    pi_img = Image.fromarray(boosted)
    dithered_pi = pi_img.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
    indices = np.array(dithered_pi).flatten()

    # 4. Convert Palette to RGB565 (Pico 2 compression format)
    palette_565 = []
    for color in palette:
        r, g, b = color
        # RRRRRggg gggbbbbb
        # Red:   Take top 5 bits, shift to bits 11-15
        r_5 = (int(r) & 0xF8) << 8
        # Green: Take top 6 bits, shift to bits 5-10
        g_6 = (int(g) & 0xFC) << 3
        # Blue:  Take top 5 bits, shift to bits 0-4
        b_5 = (int(b) >> 3) & 0x1F

        rgb565 = (r_5 | g_6 | b_5) & 0xFFFF
        palette_565.append(f"0x{rgb565:04X}")

    # 5. Pack Pixels (4 bits each, 2 pixels per byte)
    packed_data = []
    for i in range(0, len(indices), 2):
        # First pixel in high 4 bits, second pixel in low 4 bits
        byte = (indices[i] << 4) | (indices[i + 1] & 0x0F)
        packed_data.append(f"0x{byte:02X}")

    # 6. Write the C++ Header File
    with open(f"{header_name}.h", "w") as f:
        f.write(f"#ifndef {header_name.upper()}_H\n#define {header_name.upper()}_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"// Image Palette (16 colors in RGB565)\n")
        f.write(f"const uint16_t {header_name}_palette[16] = {{\n    ")
        f.write(", ".join(palette_565))
        f.write("\n};\n\n")

        f.write(f"// Image Data (160x120 pixels, 4-bit indexed, packed 2 per byte)\n")
        f.write(f"const uint8_t {header_name}_data[{len(packed_data)}] = {{\n    ")
        # Format lines for readability
        for i, byte_hex in enumerate(packed_data):
            f.write(byte_hex + (", " if i < len(packed_data) - 1 else ""))
            if (i + 1) % 12 == 0: f.write("\n    ")
        f.write("\n};\n\n")
        f.write("#endif")

    print(f"Successfully converted to {header_name}.h")
    print(f"Final size: {len(packed_data)} bytes (Approx {len(packed_data) / 1024:.2f} KB)")


# Run the script
convert_to_pico_header("input.jpg", "image_asset")