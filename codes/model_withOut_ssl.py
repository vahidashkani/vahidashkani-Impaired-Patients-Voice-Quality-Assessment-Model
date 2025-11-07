#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import torch
import torch.nn as nn

class GRBASPredictor(nn.Module):
    def __init__(self, asr_model, decoder_input_ids, ssl_out_dim, grbas_dim):
        super(GRBASPredictor, self).__init__()
        self.asr_model = asr_model
        self.decoder_input_ids = decoder_input_ids
        self.bottleneck_dim = 240  # Increased dimension for better representation
        self.features_out = 768
        self.grbas_dim = grbas_dim

        self.asr_weight = nn.Parameter(torch.rand(6), requires_grad=True)

        # Shared Layers (Bidirectional LSTM and Dense)
        self.shared_bilstm = nn.LSTM(input_size=360, hidden_size=128, num_layers=3, bidirectional=False, dropout=0.3)
        self.shared_dense = nn.Sequential(
            nn.Linear(128, 128, bias=True),  # Adjusted input size to match LSTM output
            nn.Dropout(0.3),
            nn.GELU()
        )

        # Enhanced Attention Layers
        self.mha_1 = nn.MultiheadAttention(embed_dim=128, num_heads=16)
        self.norm_1 = nn.LayerNorm(128)
        self.mha_2 = nn.MultiheadAttention(embed_dim=128, num_heads=16)
        self.norm_2 = nn.LayerNorm(128)

        # Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(128, 256),  # Increased layer dimensions
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.norm_4 = nn.LayerNorm(128)

        # Quality Stream
        self.quality_frame_score = nn.Linear(128, 1, bias=True)
        self.quality_sigmoid = nn.Sigmoid()
        self.quality_pooling = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        # Improved Adapters
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.features_out, self.bottleneck_dim),
                nn.GELU(),
                nn.LayerNorm(self.bottleneck_dim),
                nn.Dropout(0.1)
            )
            for _ in range(6)
        ])

    def forward(self, asr_mel_feature, wav, mel_features):
        mel_features = mel_features.reshape(1, -1, 120)

        x = self.asr_model(asr_mel_feature.squeeze(0), decoder_input_ids=self.decoder_input_ids, output_hidden_states=True)
        asr_hidden_state = x.encoder_hidden_states

        asr_states = [adapter(asr_hidden_state[-i]) for i, adapter in enumerate(self.adapter_layers, start=1)]
        asr_new_state = torch.stack(asr_states, dim=-1)

        asr_weight = nn.functional.softmax(self.asr_weight.unsqueeze(-1), dim=0)
        asr_x = torch.matmul(asr_new_state, asr_weight).squeeze(-1)

        _, T, _ = mel_features.shape
        all_features = torch.cat((asr_x[:, :T, :], mel_features), dim=-1)

        shared_out, _ = self.shared_bilstm(all_features)
        shared_out = self.shared_dense(shared_out)

        mha_out_1, _ = self.mha_1(shared_out, shared_out, shared_out)
        shared_out = self.norm_1(shared_out + mha_out_1)

        mha_out_2, _ = self.mha_2(shared_out, shared_out, shared_out)
        shared_out = self.norm_2(shared_out + mha_out_2)

        feedforward_out = self.feedforward(shared_out)
        shared_out = self.norm_4(shared_out + feedforward_out)

        quality_frame_score = self.quality_frame_score(shared_out)
        quality_frame = self.quality_sigmoid(quality_frame_score).permute(0, 2, 1)
        quality_frame = quality_frame 
        quality_avg = self.quality_pooling(quality_frame).squeeze(-1)
        return quality_avg








































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    