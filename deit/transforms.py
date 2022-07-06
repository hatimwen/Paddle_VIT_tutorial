"""
DateTime: 2021.11.27
Written By: Dr. Zhu
Recorded By: Hatimwen
"""
import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as T
paddle.set_device('cpu')

def crop(img, region):
    cropped_img = T.crop(img, *region)
    return cropped_img

class CenterCrop():
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        cw, ch = self.size
        crop_top = int(round((h - ch) / 2.))
        crop_left = int(round((w - cw) / 2.))
        return crop(img, (crop_top, crop_left, ch, cw))

class Resize():
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return T.resize(img, self.size)

class ToTensor():
    def __init__(self):
        pass
    def __call__(self, img):
        w, h = img.size
        img = paddle.to_tensor(np.array(img))
        if img.dtype == paddle.uint8:
            img = paddle.cast(img, paddle.float32) / 255.
        # img = img.transpose([2, 0, 1])
        return img

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def main():
    img = Image.open('deit_1127/wenht.jpg')
    img = img.convert('L')
    transforms = Compose([Resize([256, 256]), 
                          CenterCrop([112, 112]),
                          ToTensor()])
    out = transforms(img)
    print(out)
    print(out.shape)

if __name__ == '__main__':
    main()