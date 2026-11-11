import os
import numpy as np
from PIL import Image

# Paths to your images and txt mask files
img_dir = 'Dataset/images'
txt_mask_dir = 'Dataset/masks'
output_mask_dir = 'Dataset/masks_png'
os.makedirs(output_mask_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    if not (img_file.endswith('.png') or img_file.endswith('.jpg') or img_file.endswith('.jpeg')):
        continue
    base_name = img_file.rsplit('.', 1)[0]
    txt_mask_file = base_name + '.txt'
    txt_mask_path = os.path.join(txt_mask_dir, txt_mask_file)
    img_path = os.path.join(img_dir, img_file)
    output_mask_path = os.path.join(output_mask_dir, base_name + '_mask.png')

    # --- Load image to get dimensions ---
    img = Image.open(img_path)
    W, H = img.size
    mask = np.zeros((H, W), dtype=np.uint8)

    # --- Parse .txt mask ---
    if os.path.exists(txt_mask_path):
        with open(txt_mask_path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) < 5: continue  # Skip incomplete lines
                # Usually YOLO: class x_center y_center width height
                _, x_center, y_center, box_w, box_h = [float(v) for v in vals[:5]]
                # Convert to pixel coordinates
                x_c = int(x_center * W)
                y_c = int(y_center * H)
                bw = int(box_w * W)
                bh = int(box_h * H)
                x1 = max(0, x_c - bw // 2)
                y1 = max(0, y_c - bh // 2)
                x2 = min(W, x_c + bw // 2)
                y2 = min(H, y_c + bh // 2)
                mask[y1:y2, x1:x2] = 255  # Mark decay area as white

    # --- Save binary mask as PNG ---
    mask_img = Image.fromarray(mask)
    mask_img.save(output_mask_path)
    print(f"Saved mask: {output_mask_path}")

print("All masks converted to PNG.")
