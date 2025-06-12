import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from train_unet import yolo_to_mask

def load_and_preprocess_image(image_path, transform):
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # Apply transformations
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    return image, image_tensor

def predict_mask(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # Remove batch dimension and get first channel
    return mask

def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # True mask
    plt.subplot(132)
    plt.imshow(true_mask, cmap='gray')
    plt.title('True Mask')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(133)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('best_unet_model.pth'))
    model.eval()
    
    # Define transformations
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create output directory for results
    os.makedirs('test_results', exist_ok=True)
    
    # Get validation images
    val_image_dir = 'valid/images'
    val_label_dir = 'valid/labels'
    image_files = [f for f in os.listdir(val_image_dir) if f.endswith('.jpg')]
    
    # Test on 5 random images
    np.random.seed(42)  # For reproducibility
    test_images = np.random.choice(image_files, 5, replace=False)
    
    for img_file in test_images:
        # Load and preprocess image
        image_path = os.path.join(val_image_dir, img_file)
        label_path = os.path.join(val_label_dir, img_file.replace('.jpg', '.txt'))
        
        # Load original image and get its dimensions
        original_image = np.array(Image.open(image_path).convert("RGB"))
        img_height, img_width = original_image.shape[:2]
        
        # Get true mask
        true_mask = yolo_to_mask(label_path, img_width, img_height)
        
        # Preprocess image for model
        image, image_tensor = load_and_preprocess_image(image_path, transform)
        
        # Get prediction
        pred_mask = predict_mask(model, image_tensor, device)
        
        # Resize prediction to original image size
        pred_mask = Image.fromarray(pred_mask).resize((img_width, img_height))
        pred_mask = np.array(pred_mask)
        
        # Visualize and save results
        save_path = os.path.join('test_results', f'result_{img_file.replace(".jpg", ".png")}')
        visualize_prediction(original_image, true_mask, pred_mask, save_path)
        print(f"Saved result for {img_file}")

if __name__ == '__main__':
    main() 