#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:47:10 2024

@author: nca
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_withOut_ssl import GRBASPredictor
from dataset import PVQDDataset
import numpy as np
from transformers import WhisperModel, AutoFeatureExtractor, HubertModel



def count_parameters(model):
    """
    Counts the total and trainable parameters in a model.
    Args:
        model (torch.nn.Module): The model to count parameters for.
    Returns:
        total_params (int): Total number of parameters.
        trainable_params (int): Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# TESubjectiveRatings
# Hardcoded configuration
datadir = '/home/nca/Downloads/cape_v_breathiness/dataset/'
checkpoint = '/home/nca/Downloads/cape_v_breathiness/checkpoint_proposed_model/ckpt_6.pth'
outfile = '/home/nca/Downloads/cape_v_breathiness/checkpoint_proposed_model/Normalized_capeVbreathiness_test22.csv'

# Model parameters
asr_model_name = "openai/whisper-small"
feature_extractor_name = "openai/whisper-small"
ssl_out_dim = 768
grbas_dim = 1  # 1 or 5
multi_indicator = False

# Load models
print('Loading checkpoint')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asr_model = WhisperModel.from_pretrained(asr_model_name, output_attentions=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
decoder_input_ids = decoder_input_ids.to(device)

model = GRBASPredictor(asr_model, decoder_input_ids, ssl_out_dim, grbas_dim).to(device)
model.eval()
# Load model state
model.load_state_dict(torch.load(checkpoint))


##################################################
# Count parameters
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
##################################################


# Data loading
wavdir = os.path.join(datadir, 'wav')
validlist = os.path.join(datadir, 'sets/16k_slice_label/normalized_main/normalized_capeVbreathiness_test_segmented.csv')
validset = PVQDDataset(wavdir, validlist, feature_extractor, multi_indicator)
validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0, collate_fn=validset.collate_fn)

# Initialize loss function and containers
total_loss = 0.0
num_steps = 0.0
predictions = {}  # filename : prediction
#criterion = nn.MSELoss(reduction='sum')
class LogCoshLoss(nn.Module):
    def forward(self, outputs, targets):
        diff = outputs - targets
        return torch.mean(torch.log(torch.cosh(diff)))
#criterion = LogCoshLoss()
criterion = nn.L1Loss(reduction='sum')


print('Starting prediction')

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

Loss = []
for i, data in enumerate(validloader, 0):
    asr_mel_features, inputs, mel_specgrams, labels, filenames = data
    asr_mel_features = asr_mel_features.to(device)
    inputs = inputs.to(device)
    mel_specgrams = mel_specgrams.to(device)
    labels = labels.to(device)

    outputs = model(asr_mel_features, inputs, mel_specgrams)
    loss = criterion(labels, outputs)
    outputs = outputs.cpu().detach().numpy()[0]

    Loss.append(loss.cpu().detach().numpy())
    predictions[filenames[0]] = outputs  # batch size = 1

print(np.mean(Loss))

# Read true labels
true_G = {}
with open(validlist, 'r') as validf:
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        if multi_indicator:
            G = [float(i) for i in parts[1:]]
        else:
            G = float(parts[1])
        true_G[uttID] = G

# Write results to file
index = "BL05"
num = 0
with open(outfile, 'w') as ans:
    for k, v in predictions.items():
        if index == k.split('.')[0].split('/')[1].split('_')[0]:
            if multi_indicator:
                outl = k.split('.')[0] + ',' + str(num) + ',' + str(v[0]) + ',' + str(true_G[k][0])  + '\n'
            else:
                outl = k.split('.')[0] + ',' + str(num) + ',' + str(v) + ',' + str(true_G[k]) + '\n'
            ans.write(outl)
        else:
            index = k.split('.')[0].split('/')[1].split('_')[0]
            num += 1
            if multi_indicator:
                outl = k.split('.')[0] + ',' + str(num) + ',' + str(v[0]) + ',' + str(true_G[k][0]) + '\n'
            else:
                outl = k.split('.')[0] + ',' + str(num) + ',' + str(v) + ',' + str(true_G[k]) + '\n'
            ans.write(outl)
