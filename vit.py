"""
DateTime: 2021.11.24
Written By: Dr. Zhu
Recorded By: Hatimwen
"""
import paddle as paddle
import paddle.nn as nn
# from PIL import Image
from paddle.nn.layer.common import Identity

paddle.set_device('cpu')

class MLp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super(MLp, self).__init__()
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

class Encoder(nn.Layer):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.attn = Identity() #TODO
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        h = self.attn(h)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        h = self.mlp(x)
        x = x + h
        return x



class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2D(in_channels,
                                    embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                    bias_attr=False)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x) # [n, embed_dim, h', w']
        x = x.flatten(2) # [n, embed_dim, h' * w']
        x = x.transpose([0, 2, 1]) # [n, h' * w, embed_dim]
        x = self.drop_out(x)
        return x

class ViT(nn.Layer):
    def __init__(self):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [Encoder(16) for _ in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)   # 10:num_classes
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        # layernorm usually used here
        # [n, h' * w', embed_dim]
        x = x.transpose([0, 2, 1])
        x = self.avgpool(x)    # [n, embed_dim, 1]
        x = x.flatten(1)       # [n, embed_dim]
        x = self.head(x)
        return x

def main():
    # random img:
    # img = np.random.randint(0, 255, [28, 28], dtype=np.uint8)
    # sample = paddle.to_tensor(img, dtype='float32')
    # sample = sample.reshape([1, 1, 28, 28])

    # patch_embed = PatchEmbedding(28, 7, 1, 1)
    # out = patch_embed(sample)
    # print(out)
    # print(out.shape)

    # mlp = MLp(1)
    # out = mlp(out)
    # print(out)
    # print(out.shape)

    t = paddle.randn([4, 3, 224, 224])
    vit = ViT()
    out = vit(t)
    print(out)
    print(type(out))
    print(out.shape)

if __name__ == "__main__":
    main()