import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.feature_transform = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.classifier = nn.Linear(256, num_classes)
        self.num_classes = num_classes
        self.last_features = None

    def get_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)

        # Workaround for MPS adaptive pooling limitation
        if x.device.type == "mps":
            x = x.cpu()
            x = self.global_pool(x)
            x = x.to("mps")
        else:
            x = self.global_pool(x)

        x = x.flatten(1)
        feats = self.feature_transform(x)
        return feats


    def forward(self, x):
        feats = self.get_features(x)
        self.last_features = feats.detach()
        return self.classifier(feats)
