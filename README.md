# Great Lakes Ice Challenge (GLIC)
---
Great Lakes Ice Forecasting Model using Python with PyTorch to train a UNet to predict ice concentration for the next 3 days.

## Validating the training pipeline

The repository does not ship the large training assets used by the real
competition.  To exercise the end-to-end pipeline locally you can synthesize a
small dataset and run a short CPU-only training loop:

1. Install the Python dependencies listed in `requirements.txt` (PyTorch and
the geospatial stack are required).
2. Generate the fixtures:

   ```bash
   python scripts/generate_synthetic_training_data.py
   ```

   This populates the `datasets/` folder with 21 days of deterministic mock
   data that matches the paths expected in `config.py`.
3. Launch the quick smoke test, which reuses the real `train.py` module but
   limits the run to a single epoch with tiny batches:

   ```bash
   python scripts/run_quick_train.py
   ```

The training loop will build the land mask, load the generated weather/state
inputs, run one epoch, and emit a checkpoint under `checkpoints/`.
