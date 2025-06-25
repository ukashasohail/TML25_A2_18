import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image
import requests
import io
import base64
import json
import os
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configuration and Hyperparameters
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
TOKEN = "17506180"
PORT = "9947"
SEED = "33893953"
BATCH_SIZE = 64
EPOCHS = 2
TEMPERATURE = 0.5
LEARNING_RATE = 3e-4
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

class TaskDataset(Dataset):
    def __init__(self, ids=None, imgs=None, labels=None, transform=None):
        # Initialize with provided data or empty lists if not provided (for loading from .pt)
        self.ids = ids if ids is not None else []
        self.imgs = imgs if imgs is not None else []
        self.labels = labels if labels is not None else []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

def convert_to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

def get_train_transform():
    return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2980, 0.2962, 0.2987],
                std=[0.2886, 0.2875, 0.2889]
            )
        ])

def get_eval_transform():
    return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Lambda(convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2980, 0.2962, 0.2987],
                std=[0.2886, 0.2875, 0.2889]
            )
        ])
    
# 3. Data Preparation and Augmentation
class ContrastiveStealingDataset(Dataset):
    def __init__(self, images, representations, labels=None, transform=None):
        self.images = images
        self.representations = representations
        self.labels = labels if labels is not None else []
        self.transform = transform or get_train_transform()
    
    def __getitem__(self, idx):
        img = self.images[idx]
        rep = self.representations[idx]
        label = self.labels[idx] if self.labels else None
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, rep, label
    
    def __len__(self):
        return len(self.images)

# 4. Model Architecture (ResNet-18)
class StolenEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # modify the first conv layer in the og resnet
        self.encoder.maxpool = nn.Identity() # modify the maxpool layer in the og resnet
        # our custom proejction head - input features should be 512
        # A small multi-layer perceptron that projects the 512-dimensional output of ResNet-18 to a higher-dimensional space (2048), applies batch normalization and ReLU, then projects down to output_dim 
        self.projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.projector(x), p=2, dim=1) # pass through projection head
    
# 5. InfoNCE Contrastive Loss Function - alternative to mse loss
class ContStealNTXent(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j, t):
        """
        z_i, z_j: [N, D] are stolen encoder views
        t:        [N, D] is our target representations
        """
        device = z_i.device
        N = z_i.size(0)
        
        # Normalize all representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        t = F.normalize(t, dim=1)
        
        # Combine representations: [3N, D]
        reps = torch.cat([z_i, z_j, t], dim=0)
        
        # Compute similarity matrix: [3N, 3N]
        sim_matrix = torch.mm(reps, reps.t()) / self.temperature
        
        # Create positive mask
        pos_mask = torch.zeros((3*N, 3*N), dtype=torch.bool, device=device)
        
        # Set positive pairs
        # View1-View2
        pos_mask[torch.arange(N), torch.arange(N, 2*N)] = True
        pos_mask[torch.arange(N, 2*N), torch.arange(N)] = True
        
        # View1-Target
        pos_mask[torch.arange(N), torch.arange(2*N, 3*N)] = True
        pos_mask[torch.arange(2*N, 3*N), torch.arange(N)] = True
        
        # View2-Target
        pos_mask[torch.arange(N, 2*N), torch.arange(2*N, 3*N)] = True
        pos_mask[torch.arange(2*N, 3*N), torch.arange(N, 2*N)] = True
        
        # Mask self-similarity
        self_mask = torch.eye(3*N, dtype=torch.bool, device=device)
        valid_mask = ~self_mask
        
        # Compute loss
        numerator = torch.sum(
            sim_matrix[pos_mask].view(3*N, -1), 
            dim=1, 
            keepdim=True
        )
        
        denominator = torch.logsumexp(
            sim_matrix[valid_mask].view(3*N, -1), 
            dim=1, 
            keepdim=True
        )
        
        loss = - (numerator - denominator).mean()
        return loss
    
def load_and_combine_batches(dataset):    
    """Loads the data representations that were queried from the API"""
    # Count of items in folder of stealing_data
    total_representation_batches = len(os.listdir('stealing_data'))
    
    for batch_idx in range(total_representation_batches):
        start_idx = batch_idx * 1000
        end_idx = (batch_idx + 1) * 1000
        
        print(f"Loading representations representations_{end_idx}.pkl")
        with open(f'stealing_data/representations_{end_idx}.pkl', 'rb') as f:
            batch_reps = pickle.load(f)
        batch_images = [dataset.imgs[i] for i in range(start_idx, end_idx)]
        batch_labels = [dataset.labels[i] for i in range(start_idx, end_idx)]
        batch_reps = torch.stack([torch.tensor(rep) for rep in batch_reps])
        print(f"Loaded {len(batch_images)} images and their representations for batch {batch_idx+1}")
        
        # Create training dataset (images, representations, and labels) for current batch + all previous
        if batch_idx == 0:
            cumulative_images = batch_images
            cumulative_reps = batch_reps
            cumulative_labels = batch_labels
        else:
            cumulative_images.extend(batch_images)
            cumulative_reps = torch.cat([cumulative_reps, batch_reps])
            cumulative_labels.extend(batch_labels)
        
        train_dataset = ContrastiveStealingDataset(
            cumulative_images, cumulative_reps, cumulative_labels
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, 
            shuffle=True, drop_last=True
        )
            
    return train_dataset, train_loader, cumulative_reps, cumulative_images
    
# 6. Train Function
def train(model, train_loader, optimizer, criterion, num_epochs=EPOCHS, device=device, save_path=SAVE_DIR):
    model.to(device)
    epoch_losses = []
    # Training Loop
    for epoch in range(num_epochs):
        print("\nStarting training...\n")
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for view1, view2, target_rep, label in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)
            target_rep = target_rep.to(device)
            
            optimizer.zero_grad()
            
            # Forward passes
            z1 = model(view1)
            z2 = model(view2)
            
            # Loss calculations
            loss = criterion(z1, z2, target_rep)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}\n")
        
    print("\nSaving model checkpoint\n")
    ckpt_path = f"{save_path}/stolen_encoder.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")
    
    return epoch_losses
        
        

# ---------- EVALUATION FUNCTIONS ----------
def plot_loss_curve(losses):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

def visualize_tsne(representations, labels, title):
    """Visualize representations using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(representations)
    
    # Normalize labels to fit within the range of the colormap
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)} # This ensures that the labels are valid for the colormap
    normalized_labels = [label_to_index[label] for label in labels]
    
    plt.figure(figsize=(10, 8))
    
    # cmap='tab10' colormap is used to assign distinct colors to each class
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=normalized_labels, cmap='tab10', alpha=0.6)
    
    # Ticks corresponding to the unique class labels
    plt.colorbar(scatter, ticks=range(len(unique_labels)))
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def run_tsne_visualization(model, dataset, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Computing t-SNE"):
            view1, view2, rep, label = dataset[i]
            for img in [view1, view2]:
                img = img.unsqueeze(0).to(device)
                pred = model(img).cpu()
                preds.append(pred.squeeze(0))
                labels.append(label)
    preds = torch.stack(preds).numpy()
    visualize_tsne(preds, labels, "Stolen Model Embeddings t-SNE")


# 2. Query API for Representations (if not already done)
def model_stealing(images, port, save_path=None):
    endpoint = "/query"
    url = f"http://34.122.51.94:{port}" + endpoint
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)
    
    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})
    if response.status_code == 200:
        # save as pickle file if save_path is provided
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(response.json(), f)
            print(f"Representations saved to {save_path}")
        return response.json()["representations"]
    else:
        raise Exception(f"Query failed: {response.status_code}, {response.text}")


if __name__ == "__main__":
    
    print(f"Training will run on: {device}")
    
    # Create directory for saving models
    os.makedirs("models", exist_ok=True)

    # 1. Load Surrogate Dataset
    dataset: TaskDataset = torch.load("./ModelStealingPub.pt", map_location=device)
    print(f"Loaded surrogate dataset with {len(dataset)} images")
    
    # 3. Data Preparation and Augmentation
    train_dataset, train_loader, cumulative_reps, cumulative_images = load_and_combine_batches(dataset=dataset)
    
    # 4. Model Architecture (ResNet-18)
    model = StolenEncoder().to(device)
    
    # 5. Contrastive Loss Function
    criterion = ContStealNTXent(temperature=TEMPERATURE)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Train model
    epoch_losses = train(model, train_loader, optimizer, criterion)
    
    plot_loss_curve(epoch_losses)
    run_tsne_visualization(model, train_dataset, device)
