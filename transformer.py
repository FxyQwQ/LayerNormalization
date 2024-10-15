import torch
import torch.nn as nn
import torch.nn.functional as F
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(features, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert(heads * self.head_dim == embed_size)

        self.q = nn.Linear(embed_size, embed_size, bias=False)
        self.k = nn.Linear(embed_size, embed_size, bias=False)
        self.v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        "the shape of x is (batch, length, embed)"

        "after init, embed is split to head and head_dim"

        N = x.shape[0]
        length = x.shape[1]
        q = self.q(x).view(N, length, self.heads, self.head_dim)
        k = self.k(x).view(N, length, self.heads, self.head_dim)
        v = self.v(x).view(N, length, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [q, k])
        attention = F.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql, nlhd -> nqhd", [attention, v]).reshape(N, length, self.embed_size)

        return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.norm1(attention + x)  # Residual connection
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)    # Residual connection
        return x

# 示例参数
embed_size = 64  # 嵌入维度
heads = 8         # 注意力头数
dropout = 0.1     # Dropout 比例
forward_expansion = 4  # 前馈网络扩展因子

# 创建 Transformer 块
transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)

# 示例输入 (序列长度, 批量大小, 嵌入维度)
x = torch.randn(10, 32, embed_size)  # 10 是序列长度，32 是批量大小
output = transformer_block(x)
print(output.shape)  # 期望输出形状: (10, 32, 256)