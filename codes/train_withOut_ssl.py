#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transformers import HubertModel
from transformers import AutoFeatureExtractor, WhisperModel

from model_withOut_ssl import GRBASPredictor
from dataset import PVQDDataset

# Directories
datadir = "/home/nca/Downloads/cape_v_breathiness/dataset/"
ckptdir = '/home/nca/Downloads/cape_v_breathiness/checkpoint_proposed_model/'

if not os.path.exists(ckptdir):
    os.makedirs(ckptdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

# PVQD-S
trainlist = os.path.join(datadir, 'sets/16k_slice_label/normalized_main/normalized_capeVbreathiness_train_segmented.csv')
validlist = os.path.join(datadir, 'sets/16k_slice_label/normalized_main/normalized_capeVbreathiness_dev_segmented.csv')

# ASR module
asr_model = WhisperModel.from_pretrained("openai/whisper-small", output_attentions=True)
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
decoder_input_ids = decoder_input_ids.to(device)

# Freeze the pre-trained weight
#for param in asr_model.parameters():
    #param.requires_grad = False


features_out = 768
grbas_dim = 1  # 1: Grade  5: GRBAS
multi_indicator = False

trainset = PVQDDataset(datadir, trainlist, feature_extractor, multi_indicator)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)

validset = PVQDDataset(datadir, validlist, feature_extractor, multi_indicator)
validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=1, collate_fn=validset.collate_fn)

net = GRBASPredictor(asr_model, decoder_input_ids, features_out, grbas_dim)
net = net.to(device)


criterion = nn.L1Loss(reduction='sum')
#criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss(reduction='sum')  # Or 'sum'
class LogCoshLoss(nn.Module):
    def forward(self, outputs, targets):
        diff = outputs - targets
        return torch.mean(torch.log(torch.cosh(diff)))

#criterion = LogCoshLoss()


#########################################
#########################################
# Define the Generalized Smooth L1 Loss
'''class GeneralizedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-5):
        """
        Generalized Smooth L1 Loss.
        Args:
            beta (float): Threshold to switch between L1 and L2 loss.
            epsilon (float): Small value to stabilize the log function.
        """
        super(GeneralizedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        diff = outputs - targets
        abs_diff = torch.abs(diff)
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * (diff ** 2) / (self.beta + self.epsilon),
            abs_diff - 0.5 * self.beta
        )
        return torch.mean(loss)

criterion = GeneralizedSmoothL1Loss(beta=1.0, epsilon=1e-5)'''
############################################
############################################
#optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-6, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

# Early stopping settings
orig_patience = 20
patience = orig_patience
PREV_VAL_LOSS = float('inf')

# Lists to track losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(1, 500):
    STEPS = 0
    net.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        asr_mel_features, inputs, mel_specgrams, labels, filenames = data
        
        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        mel_specgrams = mel_specgrams.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(asr_mel_features, inputs, mel_specgrams)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        STEPS += 1
        running_loss += loss.item()
    
    avg_train_loss = running_loss / STEPS
    train_losses.append(avg_train_loss)
    print(f'Epoch: {epoch} Averaged train loss: {avg_train_loss}')
    
    epoch_val_loss = 0.0
    net.eval()

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # Validation
    VALSTEPS = 0
    for i, data in enumerate(validloader, 0):
        VALSTEPS += 1
        asr_mel_features, inputs, mel_specgrams, labels, filenames = data
        
        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        mel_specgrams = mel_specgrams.to(device)
        
        outputs = net(asr_mel_features, inputs, mel_specgrams)
        loss = criterion(outputs, labels)
        epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / VALSTEPS
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    print(f'Averaged val loss: {avg_val_loss}')
    
    if avg_val_loss < PREV_VAL_LOSS:
        print('Loss has decreased')
        PREV_VAL_LOSS = avg_val_loss
        PATH = os.path.join(ckptdir, f'ckpt_{epoch}.pth')
        torch.save(net.state_dict(), PATH)
        patience = orig_patience
    else:
        patience -= 1
        if patience == 0:
            print(f'Loss has not decreased for {orig_patience} epochs; early stopping at epoch {epoch}')
            break

print('Finished Training')

# Plot the training and validation loss trends
epochs_trained = len(train_losses)  # Number of epochs before early stopping

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs_trained + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs_trained + 1), val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Trends')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_trends.png')  # Save the plot as an image
plt.show()




























