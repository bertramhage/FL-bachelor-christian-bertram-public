from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, args):
        """
        CNN for brain-tumour MRI. Adapted from https://github.com/kampmichael/FedDC.

        Args:
            args: Arguments object containing:
                - img_size (int): Size of the input images (assumed square).
                - n_classes (int): Number of output classes.
                - in_channels (int, optional): Number of input channels (default is 3).
        
        """
        super().__init__()
        in_ch = getattr(args, "in_channels", 3)

        # feature extractor
        self.conv1 = nn.Conv2d(in_ch,  32, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(8, 128)

        self.pool  = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d(1)

        # classifier head
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(128, 1)
        
    def forward(self, x):
        """ Forward pass through the CNN."""
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
