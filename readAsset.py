import numpy as np
from PIL import Image
import re

def preview_from_header(header_path):
    with open(header_path, 'r') as f:
        content = f.read()

    # 1. Extract the Palette (RGB565)
    palette_match = re.search(r'palette\[16\] = \{(.*?)\};', content, re.DOTALL)
    palette_hex = palette_match.group(1).split(',')
    palette_565 = [int(x.strip(), 16) for x in palette_hex if x.strip()]

    # Convert RGB565 back to RGB888 for display
    palette_rgb = []
    for c in palette_565:
        r = ((c >> 11) & 0x1F) << 3
        g = ((c >> 5) & 0x3F) << 2
        b = (c & 0x1F) << 3
        palette_rgb.append([r, g, b])
    palette_rgb = np.array(palette_rgb, dtype=np.uint8)

    # 2. Extract the Packed Data
    data_match = re.search(r'data\[\d+\] = \{(.*?)\};', content, re.DOTALL)
    data_hex = data_match.group(1).split(',')
    packed_bytes = [int(x.strip(), 16) for x in data_hex if x.strip()]

    # 3. Unpack bits (4-bit to index)
    indices = []
    for byte in packed_bytes:
        indices.append(byte >> 4)  # High 4 bits (Pixel 1)
        indices.append(byte & 0x0F)  # Low 4 bits (Pixel 2)

    # 4. Reconstruct Image
    # 160 * 120 = 19200 pixels
    pixel_data = palette_rgb[indices]
    img_array = pixel_data.reshape((120, 160, 3))

    # 5. Show and Save
    final_img = Image.fromarray(img_array)
    # Scale up by 2x to show the "Pico Style" look
    final_img = final_img.resize((320, 240), Image.NEAREST)
    final_img.show()
    final_img.save("reconstructed_preview.png")
    print("Reconstruction complete. Saved as reconstructed_preview.png")

# Usage
preview_from_header("image_asset.h")