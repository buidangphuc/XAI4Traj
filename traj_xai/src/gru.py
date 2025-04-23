import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUBlock(nn.Module):
    def __init__(
        self, hidden_size, num_layers, ff_dim, ff_dim2, rate=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.ff_dim2 = ff_dim2
        self.rate = rate
        
        # Input projection to transform input features to hidden_size
        self.input_projection = nn.Linear(3, hidden_size)  # 3 features: x, y, t
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Normalization and dropout layers
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(rate)
        
        # Feed-forward network
        self.conv1 = nn.Conv1d(hidden_size, ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(ff_dim, ff_dim2, kernel_size=1)

    def forward(self, inputs, training=None, mask=None):
        # Handle both 3D and 4D input tensors
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        
        # Reshape input to ensure it has the right format for processing
        if len(input_shape) == 3:
            # If input is [batch_size, seq_len, features]
            features = input_shape[2]
            inputs_reshaped = inputs  # Already in the right shape
        elif len(input_shape) == 4:
            # If input is [batch_size, seq_len, features, channels]
            features, channels = input_shape[2], input_shape[3]
            inputs_reshaped = inputs.reshape(batch_size, seq_len, features * channels)
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Project input to hidden dimension if dimensions don't match
        if features != self.hidden_size:
            projected_inputs = self.input_projection(inputs_reshaped)
        else:
            # If dimensions already match
            projected_inputs = inputs_reshaped
        
        # Create a mask for padded sequences if provided
        if mask is not None:
            # Create packed sequence for masking in GRU
            lengths = mask.sum(dim=1).cpu()
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                projected_inputs, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            # Apply GRU to packed sequence
            gru_output, _ = self.gru(packed_inputs)
            # Unpack the sequence
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(
                gru_output, 
                batch_first=True,
                total_length=seq_len
            )
        else:
            # Apply GRU to normal sequence
            gru_output, _ = self.gru(projected_inputs)
        
        # Apply layer normalization
        out_norm = self.layernorm(gru_output)
        
        # Apply dropout
        out_drop = self.dropout(out_norm)
        
        # Transpose for 1D convolution which expects [batch, channels, seq_len]
        out_drop_t = out_drop.transpose(1, 2)
        out_conv1 = F.relu(self.conv1(out_drop_t))
        out_conv2 = self.conv2(out_conv1)
        
        # Transpose back to [batch, seq_len, channels]
        out_conv2 = out_conv2.transpose(1, 2)
        
        # Residual connection
        return out_conv2 + projected_inputs


def build_model(
    n_classes,
    input_shape,
    hidden_size,
    num_layers,
    ff_dim,
    num_gru_blocks,
    mlp_units,
    dropout=0.0,
    mlp_dropout=0.0,
    mask=None,
    training=None,
) -> nn.Module:
    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Calculate dimensions
            self.hidden_size = hidden_size
            
            # Masking layer equivalent - handled in forward pass
            self.mask_value = mask
            
            # GRU blocks
            self.gru_blocks = nn.ModuleList([
                GRUBlock(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    ff_dim=ff_dim,
                    ff_dim2=hidden_size,  # Match hidden_size for residual connections
                    rate=dropout
                ) for _ in range(num_gru_blocks)
            ])
            
            # MLP layers
            layers = []
            input_dim = hidden_size
            for dim in mlp_units:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(mlp_dropout))
                input_dim = dim
            
            self.mlp = nn.Sequential(*layers)
            
            # Output layer
            last_dim = mlp_units[-1] if mlp_units else hidden_size
            self.output_layer = nn.Linear(last_dim, n_classes)
            
        def forward(self, x):
            # Handle both 3D and 4D inputs
            if len(x.shape) == 3:
                # Input is [batch_size, seq_len, features]
                batch_size, seq_len = x.shape[0], x.shape[1]
            elif len(x.shape) == 4:
                # Input is [batch_size, seq_len, features, channels]
                batch_size, seq_len = x.shape[0], x.shape[1]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            # Apply masking if needed - create mask tensor for GRU
            if self.mask_value is not None:
                # Create a mask that's True for valid positions and False for masked positions
                if len(x.shape) == 3:
                    mask = (x != self.mask_value).any(dim=-1)
                else:  # 4D input
                    mask = (x != self.mask_value).any(dim=-1).any(dim=-1)
            else:
                mask = None
            
            # Process through GRU blocks
            x_transformed = x
            for gru_block in self.gru_blocks:
                x_transformed = gru_block(x_transformed, mask=mask)
            
            # Global average pooling - taking mean over the sequence dimension
            x_pooled = x_transformed.mean(dim=1)
            
            # MLP layers
            x_mlp = self.mlp(x_pooled)
            
            # Output layer with softmax activation
            output = self.output_layer(x_mlp)
            output = F.softmax(output, dim=-1)
            
            return output
    
    return GRUModel()
