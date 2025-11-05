import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertModel


# HuBERT 
class HubertEmotionClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained_model="facebook/hubert-base-ls960",
                 mlp_hidden=256, dropout=0.3, mlp_layers=1, finetune=False):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(pretrained_model)
        hidden = self.hubert.config.hidden_size  # 768

        if not finetune: 
            for p in self.hubert.parameters():
                p.requires_grad = False

        layers = []
        dim_in = hidden
        for _ in range(max(0, mlp_layers - 1)):
            layers += [nn.Linear(dim_in, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim_in = mlp_hidden
        layers += [nn.Linear(dim_in, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_values):  
        out = self.hubert(input_values).last_hidden_state
        pooled = out.mean(dim=1)  # [B, 768]
        return self.classifier(pooled)



# Mel 
class MelCNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes=4, n_mels=64, dropout=0.3, conv_layers=3, mlp_hidden=256):
        super().__init__()
        c1, c2, c3 = 32, 64, 128

        self.conv1 = nn.Conv2d(1, c1, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(c3)

        self.use_c3 = conv_layers >= 3
        self.use_c2 = conv_layers >= 2

        mel_down = n_mels // 8 if conv_layers >= 3 else (n_mels // 4 if conv_layers == 2 else n_mels // 2)
        feat_dim = (c3 if conv_layers >= 3 else (c2 if conv_layers == 2 else c1)) * mel_down

        self.fc1 = nn.Linear(feat_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B,1,M,T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # [B,c1,M/2,T/2]

        if self.use_c2:
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)  # [B,c2,M/4,T/4]

        if self.use_c3:
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, 2)  # [B,c3,M/8,T/8]

        x = x.mean(dim=-1)     #[B,C,M’]
        x = x.flatten(1)       # [B, C*M’]
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
