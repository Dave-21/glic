import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
import os
from tqdm import tqdm

import config
from dataset import GreatLakesDataset, N_TIMESTEPS, N_INPUT_CHANNELS, N_OUTPUT_CHANNELS
from model import UNet # Import our new 2D UNet

# --- Training Configuration ---
BATCH_SIZE = 16
VAL_BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
TEMP_LOSS_WEIGHT = 0.001

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- THIS IS YOUR NEW FUNCTION ---
def masked_loss(y_pred, y_true, land_mask):
    """
    Calculates a weighted, masked MSE loss for 2 channels: ice and temp.
    y_pred, y_true are shape [B, 2, H, W]
    land_mask is shape [B, H, W]
    """
    # [B, 1, H, W]
    land_mask = land_mask.unsqueeze(1)
    water_mask = 1.0 - land_mask

    # Separate channels
    y_pred_ice, y_pred_temp = y_pred[:, 0:1], y_pred[:, 1:2]
    y_true_ice, y_true_temp = y_true[:, 0:1], y_true[:, 1:2]

    # ---- NEW: give extra weight to pixels that actually have ice ----
    # (so the model is punished much more for missing ice than for
    # correctly predicting open water)
    ice_presence = (y_true_ice > 0.05).float()      # 1 where ice, 0 where no-ice
    ice_weight   = 1.0 + 4.0 * ice_presence         # 5x weight on ice pixels

    water_mask_ice   = water_mask * ice_weight
    denom_ice        = torch.sum(water_mask_ice) + 1e-8
    denom_temp       = torch.sum(water_mask) + 1e-8

    # Ice loss (normalized 0–1)
    sq_error_ice       = (y_pred_ice - y_true_ice) ** 2
    masked_sq_error_ice = sq_error_ice * water_mask_ice
    loss_ice           = torch.sum(masked_sq_error_ice) / denom_ice

    # Temp loss (raw °C) – unchanged weighting
    sq_error_temp        = (y_pred_temp - y_true_temp) ** 2
    masked_sq_error_temp = sq_error_temp * water_mask
    loss_temp            = torch.sum(masked_sq_error_temp) / denom_temp

    total_loss = loss_ice + (TEMP_LOSS_WEIGHT * loss_temp)
    return total_loss
# --- END YOUR NEW FUNCTION ---

def train():
    print(f"Starting training on {DEVICE}")
    
    # --- 1. Load Data ---
    print("Loading data...")
    all_dates = [
        datetime.date(2019, 1, 11) + datetime.timedelta(days=i)
        for i in range(N_TIMESTEPS)
    ]
    
    valid_start_dates = all_dates[:-1]
    
    if not valid_start_dates:
        raise ValueError("Not enough data to form a single (T, T+1) pair.")

    train_dates = valid_start_dates[:-3]
    val_dates = valid_start_dates[-3:]

    print(f"Training on {len(train_dates)} sequences, validating on {len(val_dates)} sequences.")

    train_dataset = GreatLakesDataset(train_dates, is_train=True)
    weather_stats = train_dataset.weather_stats
    val_dataset = GreatLakesDataset(
        val_dates, 
        is_train=False, 
        weather_stats=weather_stats
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # --- 2. Initialize Model ---
    # This line automatically uses N_INPUT_CHANNELS = 7 from the dataset import
    print(f"Initializing model: {N_INPUT_CHANNELS} in channels, {N_OUTPUT_CHANNELS} out channels.")
    
    model = UNet(in_channels=N_INPUT_CHANNELS, out_channels=N_OUTPUT_CHANNELS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    
    best_val_loss = float('inf')
    
    # --- 3. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc="Training")
        
        for batch in train_loop:
            x = batch['x'].to(DEVICE)
            y_true = batch['y'].to(DEVICE)
            land_mask = batch['land_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(x)
            
            loss = masked_loss(y_pred, y_true, land_mask)
            
            if loss > 0:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for batch in val_loop:
                x = batch['x'].to(DEVICE)
                y_true = batch['y'].to(DEVICE)
                land_mask = batch['land_mask'].to(DEVICE)
                
                y_pred = model(x)
                loss = masked_loss(y_pred, y_true, land_mask)
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} Complete: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Val Loss: {best_val_loss:.6f})")

    print("--- Training Complete ---")

if __name__ == "__main__":
    train()