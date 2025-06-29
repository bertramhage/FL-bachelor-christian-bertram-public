import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args):
        """LSTM model for sequence data. PyTorch implementation of https://github.com/YerevaNN/mimic3-benchmarks.
        
        Args:
            args: Arguments object containing:
                - n_features (int): Number of input features
                - dropout (float, optional): Dropout rate (default is 0.3)
        
        """
        super(LSTM, self).__init__()
        
        input_size = args.n_features
        dim = 16
        dropout = args.getattr(args, 'dropout', 0.3)
        
        # Bidirectional LSTM with units = dim/2 per direction (8 each)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=dim // 2,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Unidirectional LSTM with hidden size = dim (16)
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        # Additional dropout (after final LSTM layer)
        self.dropout_extra = nn.Dropout(dropout)
        
        # FC layer and sigmoid activation
        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the LSTM model."""
        batch_size = x.size(0)
        
        num_directions = 2
        h0_1 = torch.zeros(num_directions, batch_size, self.lstm1.hidden_size, device=x.device)
        c0_1 = torch.zeros(num_directions, batch_size, self.lstm1.hidden_size, device=x.device)
        
        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(1, batch_size, self.lstm2.hidden_size, device=x.device)
        c0_2 = torch.zeros(1, batch_size, self.lstm2.hidden_size, device=x.device)
        
        out2, _ = self.lstm2(out1, (h0_2, c0_2))

        last_output = out2[:, -1, :]
        
        last_output = self.dropout_extra(last_output) # Additional dropout
        
        final_output = self.fc(last_output)
        return final_output