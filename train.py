import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
from tqdm import tqdm
import json
import torch.nn.functional as F
from torch.cuda import amp

import config
import dataset
from model import UNet

# --- Training Configuration ---
BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
MODEL_SAVE_PATH = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def print_batch_stats(batch_data: torch.Tensor, batch_name: str):
    """Prints a quick summary of a data batch."""
    print(f"  > Stats for '{batch_name}' batch:")
    print(f"    Shape: {batch_data.shape}")
    if batch_data.numel() == 0:
        print("    Batch is empty!")
        return
    print(f"    Min:   {batch_data.min().item():.4f}")
    print(f"    Max:   {batch_data.max().item():.4f}")
    print(f"    Mean:  {batch_data.mean().item():.4f}")
    print(f"    NaNs:  {torch.isnan(batch_data).sum().item()}")
    print(f"    Infs:  {torch.isinf(batch_data).sum().item()}")

def masked_weighted_loss(y_pred, y_true, land_mask, shipping_mask):
    """
    Aggressive Loss for the Final Run.
    """
    diff = y_pred - y_true
    mse = diff ** 2
    
    # Broadcast masks
    land_mask_broadcast = land_mask.unsqueeze(1)
    shipping_mask_broadcast = shipping_mask.unsqueeze(1)

    # --- CHANGE 1: The Sledgehammer (Asymmetry) ---
    # If y_true > y_pred (Under-prediction), we punish it severely.
    # Previous was 4.0 (1+3). Now we use 10.0 (1+9).
    # This FORCES the model to predict ice if it's even slightly unsure.
    under_prediction_mask = (y_true > y_pred).float()
    bias_weight = 1.0 + (under_prediction_mask * 9.0) 
    
    # --- CHANGE 2: Route Priority ---
    # We keep this high to protect the shipping lanes.
    route_weight = 1.0 + (shipping_mask_broadcast * 4.0)
    
    # --- CHANGE 3: Compound Penalty ---
    # We Multiply them. 
    # If you under-predict on a shipping route, weight is 10 * 5 = 50x.
    total_weight = bias_weight * route_weight
    
    water_mask = 1.0 - land_mask_broadcast
    loss = mse * total_weight * water_mask
    
    valid_pixels = water_mask.sum() + 1e-6 
    return loss.sum() / valid_pixels

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training")
    
    for i, batch in enumerate(loop):
        if i == 0:
            print("\n--- First Batch Sanity Check (Train) ---")
            print_batch_stats(batch['x'], "Input (x)")
            print_batch_stats(batch['y'], "Target (y)")
            
        x = batch['x'].to(device)
        y_true = batch['y'].to(device)
        land_mask = batch['land_mask'].to(device)
        shipping_mask = batch['shipping_mask'].to(device)
        
        with amp.autocast(enabled=(device == 'cuda')):
            y_pred = model(x)
            #loss = loss_fn(y_pred, y_true, land_mask)
            loss = loss_fn(y_pred, y_true, land_mask, shipping_mask)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def validate_one_epoch(model, loader, loss_fn, device):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    loop = tqdm(loader, desc="Validating")
    
    with torch.no_grad():
        for i, batch in enumerate(loop):
            if i == 0:
                print("\n--- First Batch Sanity Check (Val) ---")
                print_batch_stats(batch['x'], "Input (x)")
                print_batch_stats(batch['y'], "Target (y)")
            
            x = batch['x'].to(device)
            y_true = batch['y'].to(device)
            land_mask = batch['land_mask'].to(device)
            shipping_mask = batch['shipping_mask'].to(device)
            
            with amp.autocast(enabled=(device == 'cuda')):
                y_pred = model(x)
                #loss = loss_fn(y_pred, y_true, land_mask)
                loss = loss_fn(y_pred, y_true, land_mask, shipping_mask)
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    return total_loss / len(loader)

def main():
    print("--- Starting Training ---")
    print(f"Using device: {DEVICE}")

    print("Loading training dataset...")
    train_dataset = dataset.GreatLakesDataset(is_train=True)
    weather_stats = train_dataset.weather_stats
    
    print("\nLoading validation dataset...")
    val_dataset = dataset.GreatLakesDataset(is_train=False, weather_stats=weather_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print("\n--- Data Loading Complete ---")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    model = UNet(in_channels=dataset.N_INPUT_CHANNELS, out_channels=dataset.N_OUTPUT_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
    scaler = amp.GradScaler(enabled=(DEVICE == 'cuda'))
    
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, masked_weighted_loss, DEVICE, scaler)
        avg_val_loss = validate_one_epoch(model, val_loader, masked_weighted_loss, DEVICE)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} Complete: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()