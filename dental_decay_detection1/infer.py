# diagnostic_infer.py
import torch
import numpy as np
from PIL import Image
from main import FasterRCNNLightning
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_MODEL = None

def diagnostic_infer(image: np.ndarray, ckpt_path: str = '3.ckpt'):
    """See what the model is actually detecting"""
    global _MODEL
    if _MODEL is None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _MODEL = FasterRCNNLightning.load_from_checkpoint(ckpt_path)
        _MODEL.to(device)
        _MODEL.eval()
    
    # Preprocess
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image
    
    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert('RGB')
    pil = pil.resize((512, 512), resample=Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = _MODEL([input_tensor])
    
    if not outputs or len(outputs) == 0:
        print("No detections at all!")
        return image, 0.0, "No detections"
    
    preds = outputs[0]
    
    print(f"Raw detections: {len(preds['scores'])}")
    for i, (score, label) in enumerate(zip(preds['scores'], preds['labels'])):
        print(f"Detection {i}: Score={score:.3f}, Label={label}")
    
    # Filter by confidence
    mask = preds['scores'] > 0.3
    filtered_scores = preds['scores'][mask]
    filtered_labels = preds['labels'][mask]
    filtered_boxes = preds['boxes'][mask]
    
    print(f"After confidence filtering: {len(filtered_scores)}")
    for i, (score, label) in enumerate(zip(filtered_scores, filtered_labels)):
        print(f"Filtered {i}: Score={score:.3f}, Label={label}")
    
    return image, 0.0, f"Found {len(filtered_scores)} detections"

# Test it
if __name__ == "__main__":
    # Test with a sample image
    test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    diagnostic_infer(test_image)