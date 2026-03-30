import numpy as np
from PIL import Image
import re

def preview_from_header(header_path):
    with open(header_path, 'r') as f:
        content = f.read()

    # 1. Extract the Palette (RGB565)
    palette_match = re.search(r'palette\[16\] = \{(.*?)\};', content, re.DOTALL)
    if not palette_match:
        raise ValueError("Palette not found in header.")
    palette_hex = palette_match.group(1).split(',')
    palette_565 = [int(x.strip(), 16) for x in palette_hex if x.strip()]

    # Convert RGB565 back to RGB888
    palette_rgb = []
    for c in palette_565:
        r = ((c >> 11) & 0x1F) << 3
        g = ((c >> 5) & 0x3F) << 2
        b = (c & 0x1F) << 3
        palette_rgb.append([r, g, b])
    palette_rgb = np.array(palette_rgb, dtype=np.uint8)

    # 2. Extract Packed Data
    data_match = re.search(r'data\[\d+\] = \{(.*?)\};', content, re.DOTALL)
    if not data_match:
        raise ValueError("Data array not found in header.")
    data_hex = data_match.group(1).split(',')
    packed_bytes = [int(x.strip(), 16) for x in data_hex if x.strip()]

    # 3. Unpack 4-bit indices
    indices = []
    for byte in packed_bytes:
        indices.append(byte >> 4)      # High nibble
        indices.append(byte & 0x0F)    # Low nibble

    # 4. Extract width and height from Image struct
    struct_match = re.search(r'struct Image\s+\w+\s*=\s*\{\s*\w+_palette,\s*\w+_data,\s*(\d+),\s*(\d+)\s*\};', content)
    if struct_match:
        width = int(struct_match.group(1))
        height = int(struct_match.group(2))
    else:
        width, height = 160, 120  # fallback

    # 5. Map indices to RGB
    pixel_data = palette_rgb[indices[:width*height]]  # safety slice
    img_array = pixel_data.reshape((height, width, 3))

    # 6. Show and Save
    final_img = Image.fromarray(img_array)
    final_img = final_img.resize((width*2, height*2), Image.NEAREST)  # scale for preview
    final_img.show()
    save_path = f"{header_path.replace('.h','')}_preview.png"
    final_img.save(save_path)
    print(f"Reconstruction complete. Saved as {save_path}")

# Usage example
preview_from_header("IMG_INPUT.h")