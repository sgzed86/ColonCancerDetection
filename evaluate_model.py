import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from train_unet import yolo_to_mask
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm

def load_and_preprocess_image(image_path, transform):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image)
    image_tensor = augmented['image']
    return image, image_tensor

def predict_mask(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    return mask

def calculate_metrics(y_true, y_pred, threshold=0.5):
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true.flatten(), y_pred_binary.flatten())
    precision = precision_score(y_true.flatten(), y_pred_binary.flatten())
    recall = recall_score(y_true.flatten(), y_pred_binary.flatten())
    f1 = f1_score(y_true.flatten(), y_pred_binary.flatten())
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_roc_curve(fpr, tpr, auc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
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
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Get validation images
    val_image_dir = 'valid/images'
    val_label_dir = 'valid/labels'
    image_files = [f for f in os.listdir(val_image_dir) if f.endswith('.jpg')]
    
    # Initialize lists to store predictions and ground truth
    all_true_masks = []
    all_pred_masks = []
    
    # Process all validation images
    print("Processing validation images...")
    for img_file in tqdm(image_files):
        # Load and preprocess image
        image_path = os.path.join(val_image_dir, img_file)
        label_path = os.path.join(val_label_dir, img_file.replace('.jpg', '.txt'))
        
        # Load original image and get its dimensions
        original_image = np.array(Image.open(image_path).convert("RGB"))
        img_height, img_width = original_image.shape[:2]
        
        # Get true mask
        true_mask = yolo_to_mask(label_path, img_width, img_height)
        
        # Preprocess image for model
        _, image_tensor = load_and_preprocess_image(image_path, transform)
        
        # Get prediction
        pred_mask = predict_mask(model, image_tensor, device)
        
        # Resize prediction to original image size
        pred_mask = Image.fromarray(pred_mask).resize((img_width, img_height))
        pred_mask = np.array(pred_mask)
        
        # Store masks
        all_true_masks.append(true_mask)
        all_pred_masks.append(pred_mask)
    
    # Convert lists to numpy arrays
    all_true_masks = np.array(all_true_masks)
    all_pred_masks = np.array(all_pred_masks)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_true_masks, all_pred_masks)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Plot and save ROC curve
    roc_save_path = os.path.join('evaluation_results', 'roc_curve.png')
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], roc_save_path)
    print(f"\nROC curve saved to: {roc_save_path}")
    
    # Save metrics to file
    metrics_save_path = os.path.join('evaluation_results', 'metrics.txt')
    with open(metrics_save_path, 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
    print(f"Metrics saved to: {metrics_save_path}")

if __name__ == '__main__':
    main() 