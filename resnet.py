"""
DateTime: 2021.11.23
Written By: Dr. Zhu
Recorded By: Hatimwen
"""
import paddle
import paddle.nn as nn

#paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Layer):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        ## 补充代码
        self.conv1 = nn.Conv2D(in_dim, out_dim, 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm(out_dim)
        self.conv2 = nn.Conv2D(out_dim, out_dim, 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm(out_dim)
        self.relu = nn.ReLU()

        if stride == 2 or in_dim != out_dim:
            self.downsample = nn.Sequential(*[
                nn.Conv2D(in_dim, out_dim, 1, stride=stride),
                nn.BatchNorm(out_dim)
            ])
        else:
            self.downsample = Identity()

    def forward(self, x):
        ## 补充代码
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = x + identity
        x = self.relu(x)
        return x 


class ResNet18(nn.Layer):
    def __init__(self, in_dim=64, num_classes=1000):
        super().__init__()
        ## 补充代码
        self.in_dim = in_dim
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm(in_dim)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_dim, n_blocks, stride=1):
        ## 补充代码
        layer_list = []
        layer_list.append(Block(self.in_dim, out_dim, stride))
        self.in_dim = out_dim
        for _ in range(1, n_blocks):
            layer_list.append(Block(self.in_dim, out_dim))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        ## 补充代码
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

         

def main():
    model = ResNet18()
    print(model)
    x = paddle.randn([2, 3, 32, 32])
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
