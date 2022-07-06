"""
DateTime: 2021.11.29
Written By: Dr. Zhu
Recorded By: Hatimwen
"""
import paddle
import paddle.nn as nn
from mask_1129 import generate_mask

paddle.set_device('cpu')

class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_size = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_size(x)  # [n, embed_dim, h', w']
        x = x.flatten(2)  # [n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1])  # [n, h'*w, embed_dim]
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape   # _ : num_patches

        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)    # [b, h/2, w/2, 4c]
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)

        return x

class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B, h//ws, w//ws, ws, ws, c]
    x = x.reshape([-1, window_size, window_size, C])
    # [B * num_patches, ws, ws, c]
    return x

def windows_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H / window_size * W / window_size))
    x = windows.reshape([B, H//window_size, W//window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x

class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])   # [B, num_heads, num_patches, dim_head]
        return x

    def forward(self, x, mask=None):
        B, N, C = x.shape
        # x: [B, num_patches, embed_dim]
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)

        ##### BEGIN CLASS 6: Mask
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows, num_patches, num_patches]
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
            attn = attn.reshape([B//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])
            # attn: [B, num_windows, num_heads, num_patches, num_patches]
            # mask: [1, num_windows, 1,         num_patches, num_patches]
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
        ##### END CLASS 6: Mask


        out = paddle.matmul(attn, v)
        # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3])
        # [B, num_patches, num_heads, dim_head] num_heads * dim_head = embed_dim
        out = out.reshape([B, N, C])
        out = self.proj(out)
        return out

class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

        # CLASS 6
        if self.shift_size > 0:
            attn_mask = generate_mask(window_size=self.window_size,
                                      shift_size=self.shift_size,
                                      input_resolution=self.resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape

        h = x
        x = self.attn_norm(x)
        
        x = x.reshape([B, H, W, C])

        ##### BEGIN CLASS 6
        # Shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # Compute window attn
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        # Shift back
        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = paddle.roll(x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        ##### END CLASS 6
        

        # [B, H, W, C]
        x = x.reshape([B, H*W, C])

        x = self.attn(x)

        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x

def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block_w_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    swin_block_sw_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    print('image shape = [4, 3, 224, 224]')
    out = patch_embedding(t)    # [4, 56, 56, 96]
    print('patch_embedding out shape = ', out.shape)
    out = swin_block_w_msa(out)
    out = swin_block_sw_msa(out)
    print('swin_block out shape = ', out.shape)
    out = patch_merging(out)
    print('patch_merging out shape = ', out.shape)

if __name__ == '__main__':
    main()