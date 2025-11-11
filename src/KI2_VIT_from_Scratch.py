import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

#1. Divide Input Image into patches and flatten them. This is needed to keep positional Information of whats next to each other
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=256, img_size=32):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Linear(in_channels * patch_size * patch_size, emb_size)
        
    def forward(self, x):
        # Split into patches
        batch_size, channels, height, width = x.size()
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, self.num_patches, -1)
        # Linear projection
        x = self.projection(x)
        return x
    
# Multi head self attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        # Multi-head attention expects embed_dim to be divisible by num_heads
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True)
        
        # Linear layer to project the input embeddings to queries, keys, and values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        
    def forward(self, x):
        # Project input to queries, keys, values
        qkv = self.qkv(x)  # (batch_size, num_patches, emb_size * 3)
        
        # Split into q, k, v and reshape for multi-head attention
        qkv = qkv.chunk(3, dim=-1)  # tuple of (batch_size, num_patches, emb_size)
        
        # MultiheadAttention expects input of shape (batch_size, seq_length, embed_dim)
        q, k, v = qkv
        
        # Compute multi-head attention output
        attn_output, _ = self.attention(q, k, v)
        
        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_size),
            nn.GELU(),
            nn.Linear(ff_hidden_size, emb_size),
            nn.Dropout(p=dropout)
        )
        self.norm2 = nn.LayerNorm(emb_size)
        
    def forward(self, x):
        # Self-Attention layer
        x = x + self.attention(x)
        x = self.norm1(x)
        
        # Feed-Forward Network
        x = x + self.ff(x)
        x = self.norm2(x)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, emb_size=256, num_classes=10, num_heads=8, num_layers=6, ff_hidden_size=512, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.dropout1 = nn.Dropout(p=dropout)       #Dropout after positional emb
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, num_heads, ff_hidden_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)
        x = self.dropout1(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        cls_token_final = x[:, 0]
        out = self.mlp_head(cls_token_final)
        return out




