# Automated Severity and Breathiness Assessment of Disordered Speech Using a Speech Foundation Model
In this study, we propose a novel automated model for speech quality estimation that objectively evaluates perceptual dysphonia severity and breathiness in audio samples, demonstrating strong agreement with expert ratings. The proposed model integrates Whisper ASR embeddings with Mel spectrograms augmented by second-order delta features combined with a sequential-attention fusion network feature mapping path. This hybrid approach enhances the model’s sensitivity to phonetic, high level feature rep-resentation and spectral variations, enabling more accurate predictions of perceptual speech quality. A sequential-attention fusion network feature mapping module captures long-range dependencies through the multi-head attention network, while LSTM layers refine the learned representations by modeling temporal dynamics. Comparative analysis against state-of-the-art methods for dysphonia assessment demonstrates our model’s better generalization across test samples. Our findings underscore the effec-tiveness of ASR-derived embeddings alongside the deep feature mapping structure in speech quality assessment, offering a promising pathway for advancing automated evaluation systems.



For transparency, all hyperparameters, model configurations, and data splits are documented here:

Model: Whisper-Small (encoder frozen) + GRBASPredictor (3-layer LSTM, attention, adapters)

Feature Inputs: (i) Whisper embeddings, (ii) Mel + Δ + Δ² features (40 × 3 = 120 dims)

Target: CAPE-V Grade (continuous, 0–4 scale)

Loss: L1Loss

Optimizer: AdamW

Scheduler: ReduceLROnPlateau

Batch Size: 1

Max Epochs: 200



# Overview of the proposed structure for quality measurement
<img width="469" height="313" alt="image" src="https://github.com/user-attachments/assets/93b7ad22-4e63-4222-9c53-3be77f4c4acc" />

Overview of the proposed adapters structure
<img width="223" height="328" alt="image" src="https://github.com/user-attachments/assets/121f15e6-0138-4921-b13d-e401deeee0d6" />

Proposed feature mapping block
<img width="477" height="288" alt="image" src="https://github.com/user-attachments/assets/63cbbad0-971c-41dc-abb3-250b8580d39f" />


