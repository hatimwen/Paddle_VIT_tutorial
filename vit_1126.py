"""
DateTime: 2021.11.26
Written By: Dr. Zhu
Recorded By: Hatimwen
"""
import paddle
import paddle.nn as nn

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        self.class_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.))

        self.position_embedding = paddle.create_parameter(
            shape=[1, n_patches+1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [n, c, h, w]
        class_tokens = self.class_token.expand([x.shape[0], -1, -1])
        # class_tokens = self.class_token.expand([x.shape[0], 1, self.embed_dim])  # for batch
        x = self.patch_embedding(x)    #[n, embed_dim, h', w']
        x = x.flatten(2) # [n, embed_dim, h' * w']
        x = x.transpose([0, 2, 1]) # [n, h' * w, embed_dim]
        x = paddle.concat([class_tokens, x], axis=1)
        
        x = x + self.position_embedding
        x = self.dropout(x)
        return x

class Attention(nn.Layer):
    """multi-head self attention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=True, dropout=0., attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3)

        self.proj = nn.Linear(self.all_head_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multi_head(self, x):
        # N: num_patches
        # x: [B, N, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x: [B, num_heads, N, head_dim]
        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)

        # q, k, v: [B, num_heads, N, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True)    # q * k^T
        attn = self.scale * attn
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)
        # attn :[B, num_heads, N, N]

        out = paddle.matmul(attn, v)    # softmax(scale(q * k^T)) * v
        out = out.transpose([0, 2, 1, 3])
        # out: [B, N, num_heads, head_dim]
        out = out.reshape([B, N, -1])

        out = self.proj(out)
        out = self.dropout(out)
        return out

class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim=768, num_heads=4, qkv_bias=True, mlp_ratio=40, dropout=0., attention_dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)

    def forward(self, x):
        h = x   # residual
        x = self.attn_norm(x)
        h = self.attn(h)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        h = self.mlp(x)
        x = x + h
        return x

class Encoder(nn.Layer):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer()
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class VisualTransformer(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.encoder = Encoder(embed_dim, depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [N, C, H, W]
        x = self.patch_embedding(x) # [N, embed_dim, h', w']
        # x = x.flatten(2) # [N, embed_dim, h' * w'] h' * w' = num_patches
        # x = x.transpose([0, 2, 1]) # [N, num_patches, embed_dim]
        x = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x

def main():
    vit = VisualTransformer()
    print(vit)
    paddle.summary(vit, input_size=(4, 3, 224, 224))


if __name__ == '__main__':
    main()