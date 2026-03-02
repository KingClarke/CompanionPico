import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
def simplify_to_vibrant_colors(image_path, output_path, n_colors=16, saturation_boost=1.75, sample_scale=0.20):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    # Boost saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_boost
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Downsample for clustering
    small = cv2.resize(boosted, (0, 0), fx=sample_scale, fy=sample_scale)
    small_pixels = small.reshape((-1, 3))

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=383,
        batch_size=10000,
    )
    kmeans.fit(small_pixels)

    # Use sklean to assign labels
    pixels = boosted.reshape((-1, 3))
    labels = kmeans.predict(pixels)

    palette = np.uint8(kmeans.cluster_centers_)
    simplified_pixels = palette[labels]
    simplified_image = simplified_pixels.reshape(h, w, 3)

    simplified_image_bgr = cv2.cvtColor(simplified_image, cv2.COLOR_HSV2RGB)
    cv2.imwrite(output_path, simplified_image_bgr)

    print("Top vibrant colors (RGB):")
    for color in palette:
        print(tuple(color))

    print(f"\nSaved to {output_path}")

# Example usage
simplify_to_vibrant_colors("input.jpg", "output.jpg")