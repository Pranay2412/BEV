import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nuscenes.nuscenes import NuScenes
from modular_demo import BEVBackbone, DetectionHead, VelocityHead, load_camera_images
import cv2

def visualize_bev_pipeline(sample_idx=10, model_path=None):
    """Visualize the complete BEV pipeline"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    backbone = BEVBackbone().to(device)
    detector = DetectionHead().to(device)
    velocity = VelocityHead().to(device)

    # Load trained weights if available
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        detector.load_state_dict(checkpoint.get('detector_state_dict', {}))
        velocity.load_state_dict(checkpoint.get('velocity_state_dict', {}))
        print("Loaded trained model weights")

    backbone.eval()
    detector.eval()
    velocity.eval()

    # Load dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='./data', verbose=False)
    sample = nusc.sample[sample_idx]

    # Load camera images
    camera_images = load_camera_images(nusc, sample)
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    # Run inference
    with torch.no_grad():
        bev_features = backbone(camera_images)
        cls_logits, reg_preds = detector(bev_features)
        velocities = velocity(bev_features)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))

    # 1. Camera inputs
    for i in range(6):
        plt.subplot(4, 6, i+1)
        plt.imshow(camera_images[i])
        plt.title(f"{camera_names[i]}")
        plt.axis('off')

    # 2. BEV feature maps
    for i in range(6):
        plt.subplot(4, 6, 6+i+1)
        bev_feat = bev_features[i, 0].cpu().numpy()  # First channel
        plt.imshow(bev_feat, cmap='viridis')
        plt.title(f"BEV Features {i+1}")
        plt.axis('off')

    # 3. Detection outputs
    for i in range(6):
        plt.subplot(4, 6, 12+i+1)
        cls_map = torch.sigmoid(cls_logits[i, 0]).cpu().numpy()
        plt.imshow(cls_map, cmap='Reds', vmin=0, vmax=1)
        plt.title(f"Detection {i+1}")
        plt.axis('off')

    # 4. Velocity outputs
    for i in range(6):
        plt.subplot(4, 6, 18+i+1)
        vel_magnitude = torch.sqrt(velocities[i, 0]**2 + velocities[i, 1]**2).cpu().numpy()
        plt.imshow(vel_magnitude, cmap='Blues')
        plt.title(f"Velocity {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'bev_pipeline_visualization_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional detailed plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # BEV feature analysis
    bev_combined = bev_features.mean(dim=0).cpu().numpy()  # Average across cameras
    axes[0, 0].imshow(bev_combined[0], cmap='viridis')
    axes[0, 0].set_title('Combined BEV Features')
    axes[0, 0].axis('off')

    # Detection heatmap
    det_combined = torch.sigmoid(cls_logits.mean(dim=0)[0]).cpu().numpy()
    im = axes[0, 1].imshow(det_combined, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Detection Confidence')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])

    # Velocity field
    vel_x = velocities.mean(dim=0)[0].cpu().numpy()
    vel_y = velocities.mean(dim=0)[1].cpu().numpy()
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    axes[0, 2].imshow(vel_mag, cmap='coolwarm')
    axes[0, 2].set_title('Velocity Magnitude')
    axes[0, 2].axis('off')

    # Feature statistics
    axes[1, 0].hist(bev_combined.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('BEV Feature Distribution')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Frequency')

    # Detection statistics
    axes[1, 1].hist(det_combined.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 1].set_title('Detection Confidence Distribution')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Frequency')

    # Velocity statistics
    axes[1, 2].hist(vel_mag.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 2].set_title('Velocity Magnitude Distribution')
    axes[1, 2].set_xlabel('Velocity')
    axes[1, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'bev_analysis_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # MMDetection-style BEV top-down result
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_mmdet_style_bev(ax, bev_features, cls_logits, reg_preds, velocities)
    plt.savefig(f'bev_mmdet_style_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Top-down 3D BEV conversion image
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    bev_top = bev_features.mean(dim=0).cpu().numpy()
    x = np.linspace(0, bev_top.shape[2] - 1, bev_top.shape[2])
    y = np.linspace(0, bev_top.shape[1] - 1, bev_top.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = bev_top[0]
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.9)
    ax.set_title('Top-down BEV 3D Surface')
    ax.set_xlabel('BEV X')
    ax.set_ylabel('BEV Y')
    ax.set_zlabel('Feature Strength')
    ax.view_init(elev=75, azim=-45)
    plt.savefig(f'bev_topdown_3d_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Visualization complete!")
    print(f"📊 BEV Features shape: {bev_features.shape}")
    print(f"📊 Detection outputs shape: {cls_logits.shape}")
    print(f"📊 Velocity outputs shape: {velocities.shape}")
    print(f"📊 Feature stats - Mean: {bev_combined.mean():.3f}, Std: {bev_combined.std():.3f}")
    print(f"📊 Detection stats - Max conf: {det_combined.max():.3f}, Mean conf: {det_combined.mean():.3f}")
    print(f"📊 BEV result image saved to bev_mmdet_style_sample_{sample_idx}.png")
    print(f"📊 Top-down 3D BEV image saved to bev_topdown_3d_sample_{sample_idx}.png")


def rotated_box(cx, cy, w, l, yaw):
    corners = np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]])
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners.dot(R.T) + np.array([cx, cy])


def draw_mmdet_style_bev(ax, bev_features, cls_logits, reg_preds, velocities, score_thr=0.35, max_boxes=20):
    bev_top = bev_features.mean(dim=0).cpu().numpy()[0]
    conf_map = torch.sigmoid(cls_logits.mean(dim=0)[0]).cpu().numpy()
    avg_reg = reg_preds.mean(dim=0).cpu().numpy()
    vel = velocities.mean(dim=0).cpu().numpy()

    ax.imshow(bev_top, cmap='gray', origin='lower')
    ax.imshow(conf_map, cmap='Reds', alpha=0.35, origin='lower', vmin=0, vmax=1)

    coords = np.argwhere(conf_map > score_thr)
    if coords.shape[0] > 0:
        scores = conf_map[coords[:, 0], coords[:, 1]]
        order = np.argsort(scores)[::-1][:max_boxes]

        for idx in order:
            y, x = coords[idx]
            score = scores[idx]
            w = np.clip(np.abs(avg_reg[3, y, x]) * 3 + 1, 0.5, bev_top.shape[1] / 2)
            l = np.clip(np.abs(avg_reg[4, y, x]) * 3 + 1, 0.5, bev_top.shape[0] / 2)
            yaw = avg_reg[6, y, x]
            box = rotated_box(x + 0.5, y + 0.5, w, l, yaw)
            patch = patches.Polygon(box, closed=True, edgecolor='yellow', facecolor='none', linewidth=1.5, alpha=0.9)
            ax.add_patch(patch)
            ax.text(x + 0.5, y + 0.5, f"{score:.2f}", color='white', fontsize=6, ha='center', va='center')

    skip = 3
    h, w = vel[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(0, w, skip), np.arange(0, h, skip))
    u = vel[0][::skip, ::skip]
    v = vel[1][::skip, ::skip]
    ax.quiver(grid_x, grid_y, u, v, color='cyan', angles='xy', scale_units='xy', scale=0.5, width=0.0025, alpha=0.8)

    ax.set_title('MMDetection-style BEV Result')
    ax.axis('off')


def create_bev_gif(sample_indices=[10, 11, 12, 13, 14]):
    """Create animated GIF showing BEV processing over multiple frames"""
    import imageio

    images = []
    for idx in sample_indices:
        visualize_bev_pipeline(idx)
        # In a real implementation, you'd save individual frames and combine them
        # For now, just process each sample
        print(f"Processed sample {idx}")

    print("GIF creation would require saving individual frames and combining with imageio")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Visualize BEV Results')
    parser.add_argument('--sample_idx', type=int, default=10, help='Sample index to visualize')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')

    args = parser.parse_args()

    visualize_bev_pipeline(args.sample_idx, args.model_path)