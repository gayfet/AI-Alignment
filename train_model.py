import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt # <-- New plotting library

# ==========================================
# 1. Dataset Loader
# ==========================================
class WaferDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "labels.csv")
        self.data = []

        # Maximums defined in our generator script for normalization
        self.max_shift = 30.0
        self.max_rot = 15.0

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'filename': row['filename'],
                    'shift_x': float(row['shift_x']),
                    'shift_y': float(row['shift_y']),
                    'rotation': float(row['rotation_deg'])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item['filename'])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize image pixels to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Normalize targets to roughly [-1, 1]
        target_x = item['shift_x'] / self.max_shift
        target_y = item['shift_y'] / self.max_shift
        target_rot = item['rotation'] / self.max_rot
        
        image_tensor = torch.tensor(img, dtype=torch.float32)
        target_tensor = torch.tensor([target_x, target_y, target_rot], dtype=torch.float32)
        
        return image_tensor, target_tensor

# ==========================================
# 2. Convolutional Neural Network
# ==========================================
class AlignmentCNN(nn.Module):
    def __init__(self):
        super(AlignmentCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten = nn.Flatten()
        
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x

# ==========================================
# 3. Training Loop & Graph Generation
# ==========================================
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "TrainingData")
    
    dataset = WaferDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = AlignmentCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    epoch_losses = [] # <-- Array to hold the loss history
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss) # <-- Log the efficiency
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

    # Save the trained model weights
    save_path = os.path.join(base_dir, "wafer_alignment_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

    # ------------------------------------------
    # Generate the Loss Graph
    # ------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Plot epochs on the X axis, loss on the Y axis
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    
    # Label the graph
    plt.title('AI Training Efficiency (Loss per Epoch)')
    plt.xlabel('Epoch (Iteration)')
    plt.ylabel('Mean Squared Error (Lower is Better)')
    plt.grid(True)
    
    # Save the graph as an image
    plot_path = os.path.join(base_dir, "training_loss_plot.png")
    plt.savefig(plot_path)
    print(f"Training graph saved to {plot_path}")

if __name__ == "__main__":
    train_model()