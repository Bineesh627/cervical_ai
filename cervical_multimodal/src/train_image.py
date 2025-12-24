# train_cnn_updated.py - PyTorch Training Script with Fixed Paths

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image # Used in the predict_image function

# --- Configuration and Path Setup ---

# Input data directory containing 'train', 'val', 'test' folders
DATA_ROOT_DIR = r'C:\Users\user\OneDrive\Desktop\cervical-ai\cervical_multimodal\data' 
# Output directory where the model will be saved
MODEL_SAVE_DIR = r'C:\Users\user\OneDrive\Desktop\cervical-ai\cervical_multimodal\models'

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 

# Full path for the saved model
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "image_cnn.pth") 

batch_size = 32
img_size = 224
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ResNet Normalization
normalize_params = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(*normalize_params)
])
val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(*normalize_params)
])

# --- Model and Data Functions ---

def get_datasets():
    # Load datasets from the new root directory
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, "val"), transform=val_transform)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, "test"), transform=val_transform)
    return train_ds, val_ds, test_ds

def build_model(num_classes):
    # Use ResNet-18 with pre-trained weights
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True) 

    # Freeze base model weights (optional, but good for transfer learning)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final Fully Connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --- Training Loop ---

def train_model():
    try:
        train_ds, val_ds, test_ds = get_datasets()
    except Exception as e:
        print(f"❌ Error loading datasets. Check if the directory structure is correct inside: {DATA_ROOT_DIR}")
        print(f"Details: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # DYNAMIC CLASS DISCOVERY
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"🚀 Detected {num_classes} classes: {class_names}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimize the un-frozen parameters (the new model.fc layer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) 

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training Phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(outputs.argmax(dim=1) == labels)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_corrects += torch.sum(outputs.argmax(dim=1) == labels)
                
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} TrainAcc: {epoch_acc:.4f} ValAcc: {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": class_names
            }, MODEL_PATH)
            print(f"💾 Saved best image model to: {MODEL_PATH}")
            
    print("Training complete. Best val acc:", best_acc.item())

# --- Prediction Function ---

def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}. Run train_model() first.")
        return 0.0, "Unknown"
        
    try:
        # Load the checkpoint
        ckpt = torch.load(MODEL_PATH, map_location=device)
        saved_class_names = ckpt.get("classes", [])
        num_classes_saved = len(saved_class_names)
        
        # Build model structure to match the saved weights
        model = build_model(num_classes_saved).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True) 
        model.eval()
        
    except Exception as e:
        print(f"❌ Error during model loading/initialization: {e}")
        return 0.0, "Loading Error"

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(*normalize_params)
    ])
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Input image file not found at {image_path}")
        return 0.0, "Image Not Found"
        
    try:
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            
        max_idx = int(probs.argmax())
        prob = float(probs.max())
        
        # Use the class names saved with the model
        label = saved_class_names[max_idx] if saved_class_names else str(max_idx)
        return prob, label
        
    except Exception as e:
        print(f"❌ Error during image processing or inference: {e}")
        return 0.0, "Inference Error"

if __name__ == "__main__":
    train_model()
    # You can add a prediction test here after training completes.
    # test_image_path = os.path.join(DATA_ROOT_DIR, "test", "Dyskeratotic", "some_file.jpg")
    # if os.path.exists(test_image_path):
    #     prob, label = predict_image(test_image_path)
    #     print(f"\nPrediction: {label} with confidence {prob*100:.2f}%")