import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from unet_model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def yolo_to_mask(yolo_path, img_width, img_height):
    try:
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        
        if not os.path.exists(yolo_path):
            return mask
        
        with open(yolo_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1))
            x2 = max(0, min(x2, img_width-1))
            y2 = max(0, min(y2, img_height-1))
            
            # Fill the mask
            mask[y1:y2, x1:x2] = 1.0
        
        return mask
    except Exception as e:
        logging.error(f"Error processing YOLO file {yolo_path}: {str(e)}")
        return np.zeros((img_height, img_width), dtype=np.float32)

class ColonPolypDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        logging.info(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.images[idx])
            label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
            
            # Load image
            image = np.array(Image.open(img_path).convert("RGB"))
            img_height, img_width = image.shape[:2]
            
            # Convert YOLO labels to mask
            mask = yolo_to_mask(label_path, img_width, img_height)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask
        except Exception as e:
            logging.error(f"Error loading image {self.images[idx]}: {str(e)}")
            raise

def get_training_augmentation():
    train_transform = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, masks in train_pbar:
            try:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except Exception as e:
                logging.error(f"Error in training step: {str(e)}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, masks in val_pbar:
                try:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks.unsqueeze(1))
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception as e:
                    logging.error(f"Error in validation step: {str(e)}")
                    continue
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logging.info(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        logging.info(f'Train Loss: {train_loss:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            logging.info('Model saved!')
        logging.info('-' * 50)

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Create datasets
        train_dataset = ColonPolypDataset(
            image_dir='train/images',
            label_dir='train/labels',
            transform=get_training_augmentation()
        )
        
        val_dataset = ColonPolypDataset(
            image_dir='valid/images',
            label_dir='valid/labels',
            transform=get_validation_augmentation()
        )
        
        logging.info(f"Number of training images: {len(train_dataset)}")
        logging.info(f"Number of validation images: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        # Initialize model
        model = UNet(n_channels=3, n_classes=1).to(device)
        logging.info("Model initialized")
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, device=device)
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 