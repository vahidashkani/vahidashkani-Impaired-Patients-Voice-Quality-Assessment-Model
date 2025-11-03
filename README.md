# Automated Severity and Breathiness Assessment of Disordered Speech Using a Speech Foundation Model
For transparency, all hyperparameters, model configurations, and data splits are documented here:

Model: Whisper-Small (encoder frozen) + GRBASPredictor (3-layer LSTM, attention, adapters)

Feature Inputs: (i) Whisper embeddings, (ii) Mel + Δ + Δ² features (40 × 3 = 120 dims)

Target: CAPE-V Grade (continuous, 0–4 scale)

Loss: L1Loss

Optimizer: AdamW

Scheduler: ReduceLROnPlateau

Batch Size: 1

Max Epochs: 200
