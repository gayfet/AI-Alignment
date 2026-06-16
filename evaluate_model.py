import os
import csv
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CNN Architecture (Must match training exactly)
# ==========================================
class AlignmentCNN(nn.Module):
    def __init__(self):
        super(AlignmentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x

# ==========================================
# 2. Evaluation Logic
# ==========================================
def evaluate_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "TestData")
    csv_path = os.path.join(test_dir, "labels.csv")
    model_path = os.path.join(base_dir, "wafer_alignment_model.pth")

    if not os.path.exists(test_dir):
        print("Error: 'TestData' folder not found. Please generate it first.")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignmentCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode (turns off gradients/dropout)

    max_shift = 30.0
    max_rot = 15.0

    actuals_x, actuals_y, actuals_rot = [], [], []
    preds_x, preds_y, preds_rot = [], [], []
    errors_x, errors_y, errors_rot = [], [], []

    print("Running Inference on Test Data...")

    with torch.no_grad(): # Crucial: tells PyTorch not to calculate gradients, saving memory
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Get Ground Truth
                true_x = float(row['shift_x'])
                true_y = float(row['shift_y'])
                true_rot = float(row['rotation_deg'])
                
                # 2. Load and format image
                img_path = os.path.join(test_dir, row['filename'])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0) # Add channel
                img = np.expand_dims(img, axis=0) # Add batch dimension -> (1, 1, 256, 256)
                
                image_tensor = torch.tensor(img, dtype=torch.float32).to(device)
                
                # 3. Predict
                output = model(image_tensor).cpu().numpy()[0]
                
                # 4. De-normalize back to physical units (Pixels and Degrees)
                pred_x = output[0] * max_shift
                pred_y = output[1] * max_shift
                pred_rot = output[2] * max_rot

                # 5. Log Data
                actuals_x.append(true_x); preds_x.append(pred_x); errors_x.append(abs(true_x - pred_x))
                actuals_y.append(true_y); preds_y.append(pred_y); errors_y.append(abs(true_y - pred_y))
                actuals_rot.append(true_rot); preds_rot.append(pred_rot); errors_rot.append(abs(true_rot - pred_rot))

    # ==========================================
    # 3. Calculate Metrics
    # ==========================================
    mae_x = np.mean(errors_x)
    mae_y = np.mean(errors_y)
    mae_rot = np.mean(errors_rot)

    print("\n--- Model Performance (Mean Absolute Error) ---")
    print(f"X Alignment Error: {mae_x:.4f} pixels")
    print(f"Y Alignment Error: {mae_y:.4f} pixels")
    print(f"Rotation Error:    {mae_rot:.4f} degrees")
    print("---------------------------------------------")

    # ==========================================
    # 4. Generate Visual Scatter Plots
    # ==========================================
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AI vs Ground Truth', fontsize=14)

    # Helper function to plot
    def plot_scatter(ax, actual, pred, title, unit):
        ax.scatter(actual, pred, alpha=0.6, color='blue')
        # Draw the "Perfect Prediction" diagonal line
        limits = [min(min(actual), min(pred)), max(max(actual), max(pred))]
        ax.plot(limits, limits, color='red', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel(f'Actual Offset ({unit})')
        ax.set_ylabel(f'Predicted Offset ({unit})')
        ax.grid(True)

    plot_scatter(axs[0], actuals_x, preds_x, f'X Shift (MAE: {mae_x:.2f}px)', 'pixels')
    plot_scatter(axs[1], actuals_y, preds_y, f'Y Shift (MAE: {mae_y:.2f}px)', 'pixels')
    plot_scatter(axs[2], actuals_rot, preds_rot, f'Rotation (MAE: {mae_rot:.2f}°)', 'degrees')

    plt.tight_layout()
    plot_path = os.path.join(base_dir, "evaluation_scatter.png")
    plt.savefig(plot_path)
    print(f"\nEvaluation complete. Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate_model()