import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(
        self, head_size, num_heads, ff_dim, ff_dim2, rate=0.1, input_dim=3
    ):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ff_dim2 = ff_dim2
        self.rate = rate
        self.embed_dim = head_size * num_heads
        self.input_dim = input_dim

        # Input projection to transform input features to embed_dim
        # Note: We'll initialize this when we know the input dimension
        self.input_projection = None
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=rate,
            batch_first=True
        )
        
        # Normalization and dropout layers
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
        # Feed-forward network
        self.conv1 = nn.Conv1d(self.embed_dim, ff_dim, kernel_size=1)
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
        
        # Create input projection if needed or if input dimension changed
        if self.input_projection is None or inputs_reshaped.shape[2] != self.input_dim:
            self.input_dim = inputs_reshaped.shape[2]
            self.input_projection = nn.Linear(self.input_dim, self.embed_dim).to(inputs.device)
        
        # Project input to embedding dimension
        projected_inputs = self.input_projection(inputs_reshaped)
        
        # Process attention mask
        attn_mask = None
        key_padding_mask = None
        if mask is not None:
            # For key_padding_mask, True indicates positions to mask
            key_padding_mask = ~mask.bool()
        
        # Apply layer normalization before attention (pre-norm)
        out_norm1 = self.layernorm1(projected_inputs)
        
        # Apply multi-head attention
        out_att, _ = self.mha(
            out_norm1, out_norm1, out_norm1,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        
        # Apply dropout and residual connection
        out_drop1 = self.dropout1(out_att)
        res = out_drop1 + projected_inputs
        
        # Feed-forward network with layer normalization (pre-norm)
        out_norm2 = self.layernorm2(res)
        
        # Transpose for 1D convolution which expects [batch, channels, seq_len]
        out_norm2_t = out_norm2.transpose(1, 2)
        out_conv1 = F.relu(self.conv1(out_norm2_t))
        out_drop2 = self.dropout2(out_conv1)
        out_conv2 = self.conv2(out_drop2)
        
        # Transpose back to [batch, seq_len, channels]
        out_conv2 = out_conv2.transpose(1, 2)
        
        # Final residual connection
        return out_conv2 + res


def build_model(
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0.0,
    mlp_dropout=0.0,
    mask=None,
    training=None,
) -> nn.Module:
    # Calculate input dimension from input_shape
    if len(input_shape) == 2:
        input_dim = input_shape[1]  # [seq_len, features]
    elif len(input_shape) == 3:
        input_dim = input_shape[1] * input_shape[2]  # [seq_len, features, channels]
    else:
        input_dim = input_shape[-1]  # Default to last dimension
    
    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Calculate dimensions
            self.embed_dim = head_size * num_heads
            
            # Masking layer equivalent - handled in forward pass
            self.mask_value = mask
            
            # Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    head_size=head_size,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    ff_dim2=self.embed_dim,  # Match embedding dim for residual connections
                    rate=dropout,
                    input_dim=input_dim
                ) for _ in range(num_transformer_blocks)
            ])
            
            # MLP layers
            layers = []
            input_dim_mlp = self.embed_dim
            for dim in mlp_units:
                layers.append(nn.Linear(input_dim_mlp, dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(mlp_dropout))
                input_dim_mlp = dim
            
            self.mlp = nn.Sequential(*layers)
            
            # Output layer
            last_dim = mlp_units[-1] if mlp_units else self.embed_dim
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
            
            # Apply masking if needed - create mask tensor for attention
            if self.mask_value is not None:
                # Create a mask that's True for valid positions and False for masked positions
                if len(x.shape) == 3:
                    mask = (x != self.mask_value).any(dim=-1)
                else:  # 4D input
                    mask = (x != self.mask_value).any(dim=-1).any(dim=-1)
            else:
                mask = None
            
            # Process through transformer blocks
            # Each transformer block handles the projection internally
            x_transformed = x
            for transformer_block in self.transformer_blocks:
                x_transformed = transformer_block(x_transformed, mask=mask)
            
            # Global average pooling - taking mean over the sequence dimension
            x_pooled = x_transformed.mean(dim=1)
            
            # MLP layers
            x_mlp = self.mlp(x_pooled)
            
            # Output layer with softmax activation
            output = self.output_layer(x_mlp)
            output = F.softmax(output, dim=-1)
            
            return output
    
    return TransformerModel()