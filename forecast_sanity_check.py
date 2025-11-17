# in a visualize_val_example.py
import torch, datetime
import matplotlib.pyplot as plt

from dataset import GreatLakesDataset, HISTORY_DAYS, FORECAST_DAYS, N_INPUT_CHANNELS, N_OUTPUT_CHANNELS
from model import UNet
from train import masked_mse_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

all_dates = [datetime.date(2019, 1, 11) + datetime.timedelta(days=i)
             for i in range(21)]
# pick one validation start date
target_date = all_dates[16]  # adjust to a known val date

ds = GreatLakesDataset([target_date], is_train=False)
sample = ds[0]
x = sample['x'].unsqueeze(0).to(DEVICE)      # (1, C, H, W) with 3-day history stacked in C
y_true = sample['y'].numpy()                 # (1, FORECAST_DAYS, H, W)

model = UNet(in_channels=N_INPUT_CHANNELS, out_channels=N_OUTPUT_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

with torch.no_grad():
    y_pred = model(x).cpu().numpy()          # (1, FORECAST_DAYS, H, W)

y_pred = y_pred[0]      # (FORECAST_DAYS, H, W)
y_true = y_true[0]      # (FORECAST_DAYS, H, W)

fig, axes = plt.subplots(2, FORECAST_DAYS, figsize=(12, 6))
for t in range(FORECAST_DAYS):
    axes[0, t].imshow(y_true[t], vmin=0, vmax=1)
    axes[0, t].set_title(f"True T+{t+1}")
    axes[0, t].axis("off")
    axes[1, t].imshow(y_pred[t], vmin=0, vmax=1)
    axes[1, t].set_title(f"Pred T+{t+1}")
    axes[1, t].axis("off")

plt.tight_layout()
plt.show()
