import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        # Projection: divides image into patches and projects to embedding dimension
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x)           # (B, E, H/patch, W/patch)
        x = x.flatten(2)           # (B, E, N_patches)
        x = x.transpose(1, 2)      # (B, N_patches, E)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_size=128, num_classes=10,
                 depth=6, heads=8, dropout=0.1):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels=3,
                                          patch_size=patch_size,
                                          emb_size=emb_size,
                                          img_size=img_size)
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # Positional embeddings for cls + patches
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, emb_size))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 3, img_size, img_size)
        B = x.size(0)
        x = self.patch_embed(x)         # (B, N_patches, E)

        # Prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 1+N_patches, E)

        # Add positional embedding
        x = x + self.pos_embed                         # (B, 1+N_patches, E)

        # Transformer
        x = self.transformer(x)                        # (B, 1+N_patches, E)

        # Classification on cls token
        cls_output = x[:, 0]                           # (B, E)
        logits = self.mlp_head(cls_output)             # (B, num_classes)
        return logits

    def save(self, filepath):
        """
        Save the state dict to the given filepath.
        """
        torch.save(self.state_dict(), str(filepath))
