import torch.nn as nn


class CNN1D_LayerWise(nn.Module):
    #CNN with layer-wise feature extraction
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Define layers separately for hooks
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        
        self.penultimate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
        # Storage for layer features
        self.layer_features = {}
        self.last_features = None
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Layer 1
        x = self.conv1(x)
        self.layer_features['conv1'] = self.global_pool(x).flatten(1).detach()
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        self.layer_features['conv2'] = self.global_pool(x).flatten(1).detach()
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3(x)
        self.layer_features['conv3'] = self.global_pool(x).flatten(1).detach()
        
        # Layer 4
        x = self.conv4(x)
        self.layer_features['conv4'] = self.global_pool(x).flatten(1).detach()
        
        # Penultimate
        x = self.global_pool(x).flatten(1)
        feats = self.penultimate(x)
        self.layer_features['penultimate'] = feats.detach()
        
        self.last_features = feats.detach()
        
        return self.classifier(feats)
