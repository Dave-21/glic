import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
import os
from tqdm import tqdm
import json
import torch.nn.functional as F
from torch.cuda import amp # --- NEW: For Mixed Precision ---

import config
#from dataset import GreatLakesDataset, N_INPUT_CHANNELS, N_OUTPUT_CHANNELS
import dataset # <-- Use new dataset
from model import UNet 

# --- Training Configuration ---
# --- OPTIMIZED ---
# Increased batch size and workers for better hardware utilization.
BATCH_SIZE = 16
VAL_BATCH_SIZE = 16 # Keep validation batch size consistent
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
MODEL_SAVE_PATH = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0 # Rule of thumb: 4x number of GPUs. Adjust based on CPU.
#TEMP_LOSS_WEIGHT = 0.001

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def print_batch_stats(batch_data: torch.Tensor, batch_name: str):
    """Prints a quick summary of a data batch."""
    print(f"  > Stats for '{batch_name}' batch:")
    print(f"    Shape: {batch_data.shape}")
    
    # *****************************************************************
    # --- ENHANCED DEBUGGING ---
    # *****************************************************************
    if batch_data.numel() == 0:
        print("    Batch is empty!")
        return
        
    print(f"    Min:   {batch_data.min().item():.4f}")
    print(f"    Max:   {batch_data.max().item():.4f}")
    print(f"    Mean:  {batch_data.mean().item():.4f}")
    print(f"    NaNs:  {torch.isnan(batch_data).sum().item()}")
    print(f"    Infs:  {torch.isinf(batch_data).sum().item()}")
    # *****************************************************************

def masked_loss(y_pred, y_true, land_mask):
    """
    Calculates MSE loss, ignoring pixels that are land.
    y_pred, y_true: [B, 3, H, W]
    land_mask: [B, H, W]
    """
    
    # Apply the mask (where land_mask is 1, loss is 0)
    # land_mask == 0 -> Water/Ice (Keep)
    # land_mask == 1 -> Land (Ignore)
    mask = 1.0 - land_mask
    
    # Ensure mask is broadcastable
    # y_pred/y_true shape: [B, 3, H, W]
    # mask shape: [B, H, W] -> [B, 1, H, W]
    mask = mask.unsqueeze(1) 
    
    # Calculate loss only on non-land pixels
    loss = F.mse_loss(y_pred * mask, y_true * mask, reduction='sum')
    
    # Normalize by the number of non-land pixels
    # (B * 3 * H * W)
    num_pixels = mask.sum()
    if num_pixels > 0:
        loss = loss / num_pixels
    else:
        # Avoid division by zero if batch is all land
        loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    return loss

def main():
    print("--- Starting Training ---")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Datasets ---
    print("Loading training dataset...")
    train_dataset = dataset.GreatLakesDataset(is_train=True)
    
    # Get weather stats from training set to pass to validation set
    weather_stats = train_dataset.weather_stats
    
    print("\nLoading validation dataset...")
    val_dataset = dataset.GreatLakesDataset(is_train=False, weather_stats=weather_stats)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=VAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print("\n--- Data Loading Complete ---")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # --- 2. Initialize Model, Loss, Optimizer ---
    model = UNet(
        in_channels=dataset.N_INPUT_CHANNELS,
        out_channels=dataset.N_OUTPUT_CHANNELS
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
    
    # --- NEW: Initialize GradScaler for Mixed Precision ---
    scaler = amp.GradScaler(enabled=(DEVICE == 'cuda'))
    
    best_val_loss = float('inf')

    # --- 3. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc="Training")
        
        for i, batch in enumerate(train_loop):
            # First batch sanity check
            if i == 0:
                print("\n--- First Batch Sanity Check (Train) ---")
                print_batch_stats(batch['x'], "Input (x)")
                print_batch_stats(batch['y'], "Target (y)")
                
            x = batch['x'].to(DEVICE)
            y_true = batch['y'].to(DEVICE)
            land_mask = batch['land_mask'].to(DEVICE)
            
            # --- NEW: Mixed Precision Forward Pass ---
            with amp.autocast(enabled=(DEVICE == 'cuda')):
                y_pred = model(x)
                loss = masked_loss(y_pred, y_true, land_mask)
            
            # --- NEW: Mixed Precision Backward Pass ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for i, batch in enumerate(val_loop):
                if i == 0:
                    print("\n--- First Batch Sanity Check (Val) ---")
                    print_batch_stats(batch['x'], "Input (x)")
                    print_batch_stats(batch['y'], "Target (y)")
                
                x = batch['x'].to(DEVICE)
                y_true = batch['y'].to(DEVICE)
                land_mask = batch['land_mask'].to(DEVICE)
                
                # --- NEW: Mixed Precision Forward Pass (Validation) ---
                with amp.autocast(enabled=(DEVICE == 'cuda')):
                    y_pred = model(x)
                    loss = masked_loss(y_pred, y_true, land_mask)
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} Complete: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()