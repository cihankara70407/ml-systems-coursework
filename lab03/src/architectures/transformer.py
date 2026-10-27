import torch
import torch.nn as nn

class CihanSelfAttentionViT(nn.Module):
    """
    Vision Transformer with Conv2d Patch Tokenizer
    Splits image into 16 visual tokens with 64-dim embedding
    """
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=7)
        self.pos_bias = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        tokens = self.patch_proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_bias
        hidden = self.encoder(tokens)
        pooled = torch.mean(hidden, dim=1)
        return self.classifier(pooled)
