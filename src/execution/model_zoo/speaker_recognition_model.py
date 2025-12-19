import torch.nn as nn

class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpeakerClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x