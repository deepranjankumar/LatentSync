import cv2
import torch
import numpy as np

# Ensure correct import for CodeFormer
import sys
import os

# Ensure the correct path for CodeFormer
CODEFORMER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CodeFormer"))
BASICSR_PATH = os.path.join(CODEFORMER_PATH, "basicsr")

sys.path.append(CODEFORMER_PATH)
sys.path.append(BASICSR_PATH)

# Import CodeFormer
from CodeFormer.basicsr.archs.codeformer_arch import CodeFormer

def load_sr_model(method):
    """Loads the appropriate super-resolution model."""
    if method == "CodeFormer":
        model = CodeFormer()
        model.eval()  # Set to evaluation mode
        return model
    elif method == "GFPGAN":
        # Load GFPGAN model (if applicable)
        pass  
    return None

def apply_super_resolution(model, image):
    """Applies super-resolution to an image using the specified model."""
    if isinstance(model, CodeFormer):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            output_tensor = model(image_tensor)
        output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        return cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return image  # If no model, return original image
