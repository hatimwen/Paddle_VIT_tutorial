# Paddle_VIT_tutorial

English | [简体中文](./README.md)

This repo contains some codes recorded from the online course, [Learn Vision Transformer from Scratch](https://aistudio.baidu.com/aistudio/course/introduce/25102?directly=1&shared=1), which was lectured by  [Dr. Zhu](https://github.com/xperzy), Baidu PaddlePaddle.

If you have any questions, please feel free to contact me.

Official code：https://github.com/BR-IDL/PaddleViT/tree/develop/edu

Timeline and corresponding codes:

- Class #0, 2021.11.23

    Implementation of resnet18. [resnet.py](./resnet.py)

- Class #1, 2021.11.24

    Let's build a ViT! [vit.py](./vit.py)

- Class #2, 2021.11.25

    Multi-Head Self Attention. [attention.py](./attention.py)

- Class #3, 2021.11.26

    Implementation of ViT. [vit_1126.py](./vit_1126.py)

- Class #4, 2021.11.27

    Implementation of DeiT. [deit/deit.py](./deit/deit.py)

    Before feeding to a net: Image Preprocess. [deit/transforms.py](./deit/transforms.py)

- Class #5, 2021.11.28

    Window Attention. [swin_transformer/main_1128.py](./swin_transformer/main_1128.py)

- Class #6, 2021.11.29

    Attention Mask. [swin_transformer/mask_1129.py](./swin_transformer/mask_1129.py)

    Implementation of SwinBlock, a block of Swin Transformer. [swin_transformer/main_1129.py](./swin_transformer/main_1129.py)

- Class #7, 2021.11.30

    Implementation of Swin Transformer. [swin_transformer/main_1130.py](./swin_transformer/main_1130.py)

    Used to load data: Iterator. [iterator_1130/tmp.py](./iterator_1130/tmp.py)

- Class #8, 2021.11.31

    How does [PaddleViT](https://github.com/BR-IDL/PaddleViT) set and load configs? [load_config](./load_config/)

- Class #9, 2021.12.1

    Distributed training for PaddlePaddle. [distributed/main.py](./distributed/main.py)

- Class #10, 2021.12.2

    Implementation of DETR. [detr](./detr/)

Thanks a lot for what Baidu PaddlePaddle have done! Fighting!
