import torch
from torchvision import transforms, models as tvmodels
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import numpy as np
import os

# --- DYNAMIC PATHS ---
# Get the base directory dynamically (the directory containing this file is 'src')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)  # cervical_multimodal directory

# Path where the trained PyTorch model is located (assuming it's in a 'models' folder)
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_cnn.pth")
GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, "cervical", "static", "cervical", "uploads", "gradcam")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load Model ---
def load_model():
    """Load the model and configure the final layer based on saved class count."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Image model not found at {MODEL_PATH}")
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    
    # Handle both direct state_dict saves and full checkpoint dictionaries
    if isinstance(ckpt, dict):
        saved_classes = ckpt.get('classes', None)
        state_dict = ckpt.get('state_dict', ckpt)
    else:
        # If ckpt is directly the state_dict
        saved_classes = None
        state_dict = ckpt

    # Determine number of classes
    if saved_classes is None:
        print("⚠️ Warning: Class names not found in checkpoint. Inferring from model structure.")
        # Try to infer from fc.weight shape in state_dict
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            print("⚠️ Warning: Could not infer number of classes. Assuming 5 classes.")
            num_classes = 5
    else:
        num_classes = len(saved_classes)

    model = tvmodels.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load state dict, handling potential prefix issues
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device).eval()
    return model

# --- Generate Grad-CAM ---
def generate_gradcam(img_input, filename_base):
    """
    Generate Grad-CAM overlay and save to the predefined GRADCAM_OUTPUT_DIR.
    Args:
        img_input (str or PIL.Image): Image path or image object
        filename_base (str): Base name for the output file
    Returns:
        str: Path to the saved Grad-CAM image
    """
    try:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        else:
            img = img_input

        model = load_model()
        cam_extractor = SmoothGradCAMpp(model, target_layer=model.layer4[-1])

        input_tensor = transform(img).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = int(output.argmax(dim=1).item())

        activation_map = cam_extractor(pred_class, output)
        activation = activation_map[0].squeeze().cpu().numpy()

        # Normalize the activation map
        act = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        heatmap = (act * 255).astype('uint8')
        heat_pil = Image.fromarray(heatmap).convert('L')

        # ✅ Ensure input image is still a PIL image
        img_resized = img.resize((224, 224))
        overlay = overlay_mask(img_resized, heat_pil, alpha=0.5)

        # Save the image
        output_filename = f"gradcam_{filename_base}"
        save_path = os.path.join(GRADCAM_OUTPUT_DIR, output_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, overlay)

        print(f"✅ Grad-CAM visualization saved to: {save_path}")
        return save_path
    
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return ""

# --- Example usage ---
#if __name__ == '__main__':
    #TEST_IMAGE_PATH = r'C:\Users\user\OneDrive\Desktop\002_05.jpg'
    #if os.path.exists(TEST_IMAGE_PATH):
       # base_name = os.path.basename(TEST_IMAGE_PATH)
        #generate_gradcam(TEST_IMAGE_PATH, base_name)
    #else:
       # print(f"❌ Test image not found at {TEST_IMAGE_PATH}. Please provide a valid path to test.")
