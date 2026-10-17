#!/usr/bin/env & python3
"""
BEV Training Script
Trains the BEV detection and velocity models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
import argparse
import os
from modular_demo import BEVBackbone, DetectionHead, VelocityHead, load_camera_images

class BEVDataset(Dataset):
    """Custom dataset for BEV training"""
    def __init__(self, nusc, samples, transform=None):
        self.nusc = nusc
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        camera_images = load_camera_images(self.nusc, sample)

        # Convert to tensors
        images = []
        for img in camera_images:
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            images.append(img)

        # Dummy targets (in real training, load from annotations)
        targets = {
            'cls_targets': torch.randn(6, 1, 16, 16),
            'reg_targets': torch.randn(6, 7, 16, 16),
            'vel_targets': torch.randn(6, 2, 16, 16)
        }

        return torch.stack(images), targets

def train_bev_model(model_type='detection', epochs=10, batch_size=2, lr=1e-4):
    """Train BEV model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Initialize models
    backbone = BEVBackbone().to(device)

    if model_type == 'detection':
        head = DetectionHead().to(device)
        criterion_cls = nn.BCEWithLogitsLoss()
        criterion_reg = nn.SmoothL1Loss()
    elif model_type == 'velocity':
        head = VelocityHead().to(device)
        criterion_vel = nn.SmoothL1Loss()
    else:
        raise ValueError("Model type must be 'detection' or 'velocity'")

    # Load dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='./data', verbose=False)
    train_samples = nusc.sample[:int(0.8 * len(nusc.sample))]  # 80% for training
    val_samples = nusc.sample[int(0.8 * len(nusc.sample)):]    # 20% for validation

    train_dataset = BEVDataset(nusc, train_samples)
    val_dataset = BEVDataset(nusc, val_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    if model_type == 'detection':
        optimizer = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    else:
        optimizer = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=lr)

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        # Training
        backbone.train()
        head.train()

        train_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets_cls = targets['cls_targets'].to(device)
            targets_reg = targets['reg_targets'].to(device)
            targets_vel = targets['vel_targets'].to(device)

            optimizer.zero_grad()

            # Forward pass
            bev_features = backbone.extract_camera_features(images.view(-1, 3, 224, 224))
            bev_features = backbone.lift_to_bev(bev_features.unsqueeze(0))

            if model_type == 'detection':
                cls_logits, reg_preds = head(bev_features)
                loss_cls = criterion_cls(cls_logits, targets_cls)
                loss_reg = criterion_reg(reg_preds, targets_reg)
                loss = loss_cls + loss_reg
            else:  # velocity
                vel_preds = head(bev_features)
                loss = criterion_vel(vel_preds, targets_vel)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        backbone.eval()
        head.eval()

        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets_cls = targets['cls_targets'].to(device)
                targets_reg = targets['reg_targets'].to(device)
                targets_vel = targets['vel_targets'].to(device)

                bev_features = backbone.extract_camera_features(images.view(-1, 3, 224, 224))
                bev_features = backbone.lift_to_bev(bev_features.unsqueeze(0))

                if model_type == 'detection':
                    cls_logits, reg_preds = head(bev_features)
                    loss_cls = criterion_cls(cls_logits, targets_cls)
                    loss_reg = criterion_reg(reg_preds, targets_reg)
                    loss = loss_cls + loss_reg
                else:
                    vel_preds = head(bev_features)
                    loss = criterion_vel(vel_preds, targets_vel)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'best_{model_type}_model.pth')

    print(f"Training complete! Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BEV Model')
    parser.add_argument('--model', type=str, default='detection', choices=['detection', 'velocity'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    train_bev_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
