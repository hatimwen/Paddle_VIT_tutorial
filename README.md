# Paddle_VIT_tutorial

Baidu PaddlePaddle `从零开始学视觉Transformer` Dr. Zhu's codes.

[English](./README_en.md) | 简体中文

课程链接：https://aistudio.baidu.com/aistudio/course/introduce/25102?directly=1&shared=1

官方代码链接：https://github.com/BR-IDL/PaddleViT/tree/develop/edu

同步上课讲的一些代码，纯手敲，仅供参考，有问题可以一起交流学习。

具体时间线及对应代码如下：

- Class #0, 2021.11.23

    resnet18 实现 [resnet.py](./resnet.py)

- Class #1, 2021.11.24

    开始搭建ViT [vit.py](./vit.py)

- Class #2, 2021.11.25

    Multi-Head Self Attention [attention.py](./attention.py)

- Class #3, 2021.11.26

    实现一个ViT模型 [vit_1126.py](./vit_1126.py)

- Class #4, 2021.11.27

    实现DeiT [deit/deit.py](./deit/deit.py)

    图像输入网络前的步骤——图像处理 [deit/transforms.py](./deit/transforms.py)

- Class #5, 2021.11.28

    图像窗口上的注意力机制 [swin_transformer/main_1128.py](./swin_transformer/main_1128.py)

- Class #6, 2021.11.29

    注意力掩码 Attention Mask [swin_transformer/mask_1129.py](./swin_transformer/mask_1129.py)

    实现Swin Transformer 的 SwinBlock [swin_transformer/main_1129.py](./swin_transformer/main_1129.py)

- Class #7, 2021.11.30

    实现 Swin Transformer [swin_transformer/main_1130.py](./swin_transformer/main_1130.py)

    数据加载过程——迭代器的实现 [iterator_1130/tmp.py](./iterator_1130/tmp.py)

- Class #8, 2021.11.31

    [PaddleViT](https://github.com/BR-IDL/PaddleViT) 中配置文件的加载逻辑 [load_config](./load_config/)

- Class #9, 2021.12.1

    PaddlePaddle 进行多机多卡训练 [distributed/main.py](./distributed/main.py)

- Class #10, 2021.12.2

    实现 DETR [detr](./detr/)

感谢百度飞桨~加油！
