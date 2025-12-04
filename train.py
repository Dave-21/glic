import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda import amp
from pathlib import Path
import datetime
import logging

import config as base_config
import dataset
from model import UNet


# This script is designed to be called from sweeper.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

NUM_WORKERS = 4


def print_batch_stats(batch_data: torch.Tensor, batch_name: str):
    """Prints a quick summary of a data batch."""
    logging.debug(f"  > Stats for '{batch_name}' batch:")
    if not isinstance(batch_data, torch.Tensor) or batch_data.numel() == 0:
        logging.debug("    Batch is empty or not a tensor!")
        return
    logging.debug(f"    Shape: {batch_data.shape}")
    logging.debug(f"    Min:   {batch_data.min().item():.4f}")
    logging.debug(f"    Max:   {batch_data.max().item():.4f}")
    logging.debug(f"    Mean:  {batch_data.mean().item():.4f}")
    logging.debug(f"    NaNs:  {torch.isnan(batch_data).sum().item()}")
    logging.debug(f"    Infs:  {torch.isinf(batch_data).sum().item()}")


# --- Loss Functions ---

class IntervalLoss(nn.Module):
    """
    Loss function that only penalizes predictions outside a valid range.
    Used for handling categorical/stepped ground truth (e.g. NIC codes).
    """
    def __init__(self):
        super().__init__()
        # Define the valid ranges for each "center point" value found in the data.
        # Based on SIGRID-3 codes.
        self.ranges = {
            0.05: (0.0, 0.10),
            0.125: (0.0, 0.15),
            0.225: (0.15, 0.30),
            0.40: (0.30, 0.70), # 1st Stage Thin
            0.50: (0.30, 0.70), # 2nd Stage Thin
            0.60: (0.30, 0.70), # 2nd Stage Thin (Alt)
            0.75: (0.70, 1.20),
            0.95: (0.70, 1.20),
            1.60: (1.20, 2.50), # Cap at 2.5m
            2.00: (1.20, 3.00),
            2.50: (1.20, 5.00),
        }
        
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        # Ensure we compute loss in float32 for stability and to avoid type mismatch
        # Autocast might provide float16 preds, but targets are float32.
        pred = pred.float()
        target = target.float()
        
        loss = torch.zeros_like(pred)
        
        # 1. Exact matches (0.0) or unknown values -> Standard L1
        # For 0.0 (no ice), we want strict L1/MSE to 0.
        mask_zero = (target < 0.01)
        loss[mask_zero] = F.mse_loss(pred[mask_zero], target[mask_zero], reduction='none')
        
        # 2. For each known "step", apply deadzone
        processed_mask = mask_zero.clone()
        
        for center_val, (low, high) in self.ranges.items():
            # Find pixels with this target value (approximate float match)
            mask = (torch.abs(target - center_val) < 0.01)
            if not mask.any():
                continue
                
            # Predictions inside range -> 0 loss
            # Predictions < low -> (low - pred)^2
            # Predictions > high -> (pred - high)^2
            
            p = pred[mask]
            l = torch.zeros_like(p)
            
            under = p < low
            over = p > high
            
            l[under] = (low - p[under]) ** 2
            l[over] = (p[over] - high) ** 2
            
            loss[mask] = l
            processed_mask |= mask
            
        # 3. Catch-all for values not in our table (e.g. interpolation artifacts) -> Standard MSE
        remaining = ~processed_mask
        if remaining.any():
            loss[remaining] = F.mse_loss(pred[remaining], target[remaining], reduction='none')
            
        return loss.mean()

def masked_weighted_loss(
    y_pred, y_true_concentration, y_true_thickness, y_true_thickness_class, land_mask, shipping_mask, edge_mask, x, config
):
    """
    Calculates a composite loss for multi-task learning, configured by the sweep.
    Includes categorical thickness loss and edge mask weighting.
    """
    y_pred_concentration = y_pred["concentration"]
    y_pred_thickness = y_pred["thickness"]
    
    # Sanitize ground truth
    y_true_concentration = torch.nan_to_num(y_true_concentration, nan=0.0)
    y_true_thickness = torch.nan_to_num(y_true_thickness, nan=0.0)
    
    ICE_THRESHOLD = 0.15

    water_mask = 1.0 - land_mask.unsqueeze(1)
    valid_pixels_conc = water_mask.sum() * dataset.N_TIMESTEPS + 1e-8

    # --- 1. Concentration Loss ---
    if config.get("loss_type_conc", "mse") == "l1":
        loss_conc_unweighted = torch.abs(y_pred_concentration - y_true_concentration)
    else:
        loss_conc_unweighted = (y_pred_concentration - y_true_concentration) ** 2

    # --- Weighting Scheme ---
    # a) Recall Focus
    ice_gt_mask = (y_true_concentration > ICE_THRESHOLD).float()
    water_pred_mask = (y_pred_concentration < ICE_THRESHOLD).float()
    missed_ice_mask = ice_gt_mask * water_pred_mask
    recall_focus_weight = 1.0 + (missed_ice_mask * 10.0)

    # b) Shipping Routes (T+1 only)
    shipping_mask_broadcast = shipping_mask.unsqueeze(1)
    route_weight_t1 = 1.0 + (shipping_mask_broadcast * 4.0)
    
    # c) Edge Mask Weighting (New)
    # edge_mask is (B, H, W). Broadcast to (B, T, H, W)
    edge_mask_broadcast = edge_mask.unsqueeze(1)
    edge_weight = 1.0 + (edge_mask_broadcast * 10.0) # 10x penalty on edges

    # d) Combine weights
    final_weights = recall_focus_weight * edge_weight
    final_weights[:, 0:1, :, :] *= route_weight_t1
    
    # e) Apply weights
    base_loss_unreduced = loss_conc_unweighted * final_weights * water_mask
    base_loss = base_loss_unreduced.sum() / valid_pixels_conc

    # --- 2. Physics-Informed Penalty ---
    pred_conc_t1 = y_pred_concentration[:, 0:1, :, :]
    ice_t0 = x[:, 0, :, :].unsqueeze(1)
    growth = pred_conc_t1 - ice_t0
    excess_growth = torch.relu(growth - config["max_ice_growth_per_day"])
    physics_penalty = (excess_growth**2) * water_mask[:, 0:1, :, :]
    physics_loss = physics_penalty.sum() / (water_mask[:, 0:1, :, :].sum() + 1e-8)

    loss_concentration = base_loss + (config["physics_loss_weight"] * physics_loss)

    # --- 3. Thickness Loss ---
    final_thickness_mask = water_mask 
    valid_pixels_thick = final_thickness_mask.sum() + 1e-8

    if valid_pixels_thick > 1e-7:
        if config.get("num_classes", 1) > 1:
            # --- Categorical Cross Entropy ---
            # y_pred_thickness: (B, T, C, H, W)
            
            # Compute targets on-the-fly to ensure shape correctness
            # y_true_thickness: (B, T, H, W)
            t_targets = torch.zeros_like(y_true_thickness, dtype=torch.long)
            t_targets[(y_true_thickness > 0.001) & (y_true_thickness < 0.10)] = 1
            t_targets[(y_true_thickness >= 0.10) & (y_true_thickness < 0.30)] = 2
            t_targets[(y_true_thickness >= 0.30) & (y_true_thickness < 0.70)] = 3
            t_targets[(y_true_thickness >= 0.70) & (y_true_thickness < 1.20)] = 4
            t_targets[y_true_thickness >= 1.20] = 5
            
            y_true_thickness_class = t_targets
            
            # Flatten B and T for CE Loss: (N, C, H, W) vs (N, H, W)
            B, T, C, H, W = y_pred_thickness.shape
            p_flat = y_pred_thickness.view(B * T, C, H, W)
            t_flat = y_true_thickness_class.view(B * T, H, W)
            
            # Weights need to be flattened too: (N, H, W)
            # We use final_weights (includes edge/recall/shipping)
            w_flat = final_weights.view(B * T, H, W)
            
            # Mask needs expansion to T dimension before flattening
            # final_thickness_mask is (B, 1, H, W) -> (B, T, H, W)
            m_expanded = final_thickness_mask.expand(B, T, H, W)
            m_flat = m_expanded.reshape(B * T, H, W) 
            
            # CE Loss with reduction='none' to apply pixel weights
            ce_loss = F.cross_entropy(p_flat, t_flat, reduction='none') # (N, H, W)
            
            # CE Loss with reduction='none' to apply pixel weights
            ce_loss = F.cross_entropy(p_flat, t_flat, reduction='none') # (N, H, W)
            
            # Apply weights and mask
            weighted_ce = ce_loss * w_flat * m_flat
            loss_thickness = weighted_ce.sum() / (m_flat.sum() + 1e-8)
            
        elif config.get("loss_type_thick") == "interval":
            # Interval Loss (Regression)
            mask_broadcast = final_thickness_mask.expand_as(y_pred_thickness).bool()
            p_flat = y_pred_thickness[mask_broadcast]
            t_flat = y_true_thickness[mask_broadcast]
            loss_fn_thick = IntervalLoss()
            loss_thickness = loss_fn_thick(p_flat, t_flat)
            
        else:
            # Standard Regression (MSE/L1)
            if config.get("loss_type_thick", "smooth_l1") == "mse":
                loss_thickness_unweighted = (y_pred_thickness - y_true_thickness) ** 2
            else:
                loss_thickness_unweighted = F.smooth_l1_loss(
                    y_pred_thickness, y_true_thickness, reduction="none"
                )
            
            if config.get("thickness_loss_ice_weight", False):
                ice_thickness_weight = y_true_thickness.clamp(min=0.1)
                loss_thickness_unweighted = loss_thickness_unweighted * ice_thickness_weight

            loss_thickness_unweighted = loss_thickness_unweighted * final_weights
            loss_thickness = (
                loss_thickness_unweighted * final_thickness_mask
            ).sum() / valid_pixels_thick
    else:
        loss_thickness = 0.0

    # --- 4. Combine Losses ---
    total_loss = loss_concentration + (
        config["thickness_loss_weight"] * loss_thickness
    )

    return total_loss, loss_concentration, loss_thickness, physics_loss

def train_one_epoch(model, loader, optimizer, config, device, scaler):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training")

    for i, batch in enumerate(loop):
        # DEBUG: Check if we even enter the loop
        if i == 0:
            logging.info("Entered training loop. Processing first batch...")

        x = batch["x"].to(device)
        y_true_concentration = batch["y"].to(device)
        y_true_thickness = batch.get(
            "y_thickness", torch.zeros_like(y_true_concentration)
        ).to(device)
        y_true_thickness_class = batch.get(
            "y_thickness_class", torch.zeros_like(y_true_concentration).long()
        ).to(device)
        land_mask = batch["land_mask"].to(device)
        shipping_mask = batch["shipping_mask"].to(device)
        edge_mask = batch.get(
            "edge_mask", torch.zeros_like(shipping_mask)
        ).to(device)

        # DEBUG SHAPES
        #print(f"y_conc: {y_true_concentration.shape}, y_thick: {y_true_thickness.shape}")

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            y_pred = model(x)
            # Unpack the tuple returned by masked_weighted_loss
            loss, _, _, _ = masked_weighted_loss(
                y_pred,
                y_true_concentration,
                y_true_thickness,
                y_true_thickness_class,
                land_mask,
                shipping_mask,
                edge_mask,
                x,
                config,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate_one_epoch(model, loader, config, device):
    model.eval()
    total_loss = 0.0
    loop = tqdm(loader, desc="Validating")

    with torch.no_grad():
        for i, batch in enumerate(loop):
            x = batch["x"].to(device)
            y_true_concentration = batch["y"].to(device)
            y_true_thickness = batch.get(
                "y_thickness", torch.zeros_like(y_true_concentration)
            ).to(device)
            y_true_thickness_class = batch.get(
                "y_thickness_class", torch.zeros_like(y_true_concentration).long()
            ).to(device)
            land_mask = batch["land_mask"].to(device)
            shipping_mask = batch["shipping_mask"].to(device)
            edge_mask = batch.get(
                "edge_mask", torch.zeros_like(shipping_mask)
            ).to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                y_pred = model(x)
                loss, _, _, _ = masked_weighted_loss(
                    y_pred,
                    y_true_concentration,
                    y_true_thickness,
                    y_true_thickness_class,
                    land_mask,
                    shipping_mask,
                    edge_mask,
                    x,
                    config,
                )

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            if base_config.DEBUG_MODE and i == 0:
                import matplotlib.pyplot as plt
                base_config.DEBUG_DIR.mkdir(exist_ok=True)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                for j in range(3):
                    axes[0, j].imshow(y_true_concentration[0, j].cpu().numpy())
                    axes[0, j].set_title(f"True Conc T+{j+1}")
                    axes[1, j].imshow(y_pred["concentration"][0, j].cpu().numpy())
                    axes[1, j].set_title(f"Pred Conc T+{j+1}")
                plt.savefig(base_config.DEBUG_DIR / f"validation_epoch_{i}.png")
                plt.close()


    return total_loss / len(loader)


def run_experiment(config: dict) -> str:
    """
    Runs a single, full training experiment based on a configuration dictionary.
    Returns the path to the best saved model.
    """
    run_id = config["run_id"]
    model_save_dir = Path(base_config.MODEL_PATH).parent / run_id
    os.makedirs(model_save_dir, exist_ok=True)
    import json
    best_model_path = model_save_dir / "best_model.pth"
    
    # Save configuration for reproducibility and evaluation
    with open(model_save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    logging.info(f"--- Starting Experiment: {run_id} ---")
    logging.info(f"Using device: {DEVICE}")

    # --- DataLoaders ---
    logging.info("Loading datasets...")
    # validation_start_date = datetime.date(2019, 4, 1)
    # User requested more training data, so pushing validation back to mid-April
    validation_start_date = datetime.date(2019, 4, 15)
    
    # Pre-load training data (filtered, so hopefully fits in RAM) for max speed
    train_dataset = dataset.ConfigurableFastTensorDataset(
        is_train=True,
        val_start_date=validation_start_date,
        shipping_routes_only=True,
        pre_load=False,
        min_ice_threshold=config.get("min_ice_threshold", 0.15),
        min_thickness_threshold=config.get("min_thickness_threshold", 0.1),
        stratify_mode=config.get("stratify_mode", False),
        stratify_ratio=config.get("stratify_ratio", 0.5),
        stratify_threshold=config.get("stratify_threshold", 0.3),
    )
    
    # Lazy-load validation data to be safe with memory
    val_dataset = dataset.ConfigurableFastTensorDataset(
        is_train=False, 
        val_start_date=validation_start_date,
        pre_load=False,
    )

    # Use FileAwareSampler for both lazy-loaded datasets
    train_sampler = dataset.FileAwareSampler(train_dataset)
    val_sampler = dataset.FileAwareSampler(val_dataset)
    
    # Dynamic Worker Allocation
    import psutil
    total_ram = psutil.virtual_memory().total / (1024 ** 3) # GB
    available_ram = psutil.virtual_memory().available / (1024 ** 3) # GB
    
    config_workers = config.get("num_workers", -1)
    
    if config_workers >= 0:
        num_workers = config_workers
        logging.info(f"Using configured num_workers: {num_workers}")
    else:
        # Auto-detect
        # Adjusted for 16GB RAM system with batch_size=8
        if available_ram > 24:
            num_workers = 4
        elif available_ram > 6:
            num_workers = 2
        else:
            num_workers = 0
        logging.info(f"Auto-detected num_workers: {num_workers} (Available RAM: {available_ram:.1f} GB)")

    # Pin memory only if we have workers and enough RAM headroom
    # With 16GB RAM, pinning might be risky if we are close to the edge, but let's try if > 8GB free
    pin_memory = (num_workers > 0) and (available_ram > 8)
    persistent = (num_workers > 0)
    prefetch = 2 if num_workers > 0 else None

    logging.info(f"DataLoader config: workers={num_workers}, persistent={persistent}, pin={pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Model ---
    model = UNet(
        in_channels=dataset.N_INPUT_CHANNELS,
        out_channels=dataset.N_OUTPUT_CHANNELS,
        n_filters=config["n_filters"],
        dropout_rate=config["dropout_rate"],
        depth=config["unet_depth"],
        use_attention=config.get("use_attention", False),
        use_physics_gate=config.get("use_physics_gate", False),
        num_classes=config.get("num_classes", 1),
    ).to(DEVICE)

    # --- Optimizer ---
    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=config["scheduler_patience"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # --- Training Loop ---
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        logging.info(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
        print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")

        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, config, DEVICE, scaler
        )
        avg_val_loss = validate_one_epoch(model, val_loader, config, DEVICE)

        scheduler.step(avg_val_loss)

        msg = f"Epoch {epoch+1} Complete: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}"
        logging.info(msg)
        print(msg)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Model improved. Saved to {best_model_path}")
            
    logging.info(f"--- Experiment {run_id} Complete ---")
    return str(best_model_path)


def main():
    """
    Main function to run a single training with default parameters.
    """
    print("--- Running Single Training with Default Config ---")
    default_config = base_config.get_default_config()
    run_experiment(default_config)


if __name__ == "__main__":
    main()