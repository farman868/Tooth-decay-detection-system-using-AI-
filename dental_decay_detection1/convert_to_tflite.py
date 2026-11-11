import torch
import os
import tensorflow as tf
from unet_model import UNet

def convert():
    print("🚀 Starting High-Precision Conversion...")

    # 1. Load the PyTorch Model
    # Ensure unet_dental.pth and unet_model.py are in this folder!
    try:
        model = UNet() 
        model.load_state_dict(torch.load('unet_dental.pth', map_location='cpu'))
        model.eval()
        print("✅ PyTorch weights loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Export PyTorch to ONNX
    # We use Opset 18 because your version of PyTorch requires it
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(model, dummy_input, "temp_model.onnx", 
                      opset_version=18, 
                      input_names=['input'], 
                      output_names=['output'])
    print("✅ Step 1: PyTorch to ONNX successful.")

    # 3. ONNX to TensorFlow (Using the CLI for stability)
    print("⏳ Step 2: Converting ONNX to TensorFlow...")
    os.system("python -m tf2onnx.convert --onnx temp_model.onnx --output temp_tf_model --tag serve")
    
    if not os.path.exists("temp_tf_model"):
        print("❌ Error: TensorFlow model folder was not created.")
        return
    print("✅ Step 2: ONNX to TensorFlow successful.")

    # 4. TensorFlow to TFLite (The Final Android file)
    print("⏳ Step 3: Finalizing TFLite file...")
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf_model")
    
    # We use Float16 to keep the model 'Alive' (No Scale: 0.0!)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    # 5. Save the prize
    with open('unet_dental_final.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("\n" + "="*40)
    print("🚀 SUCCESS! 'unet_dental_final.tflite' is ready.")
    print("="*40)
    print("Action: Copy this file to app/src/main/assets/")

if __name__ == "__main__":
    convert()