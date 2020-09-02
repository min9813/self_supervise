import cv2
import numpy as np
import imgaug.augmenters as iaa
from torchvision.transforms import transforms


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, p=0.5, _min=0.1, _max=2.0, image_wh=None):
        self.min = _min
        self.max = _max
        # kernel size is set to be 10% of the image height/width
        if image_wh is None:
            self.kernel_size = None
        else:
            kw = image_wh[0] // 10
            kw = kw + 1 - (kw & 1)

            kh = image_wh[1] // 10
            kh = kh + 1 - (kh & 1)

            self.kernel_size = (kw, kh)
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)
        h, w, c = sample.shape
        if self.kernel_size is None:
            kh = h // 10
            kh = kh + 1 - kh & 1
            kw = w // 10 + 1 - w & 1
            kw = kw + 1 - kw & 1
            kernel_size = (kw, kh)
        else:
            kernel_size = self.kernel_size

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * \
                np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, kernel_size, sigma)

        return sample

    def __str__(self):
        return "GaussianBlur(min={}, max={}, kernel_size={}, p={})\n".format(self.min, self.max, self.kernel_size, self.p)


def get_aug_trans(use_color_aug, use_shape_aug,  mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    if use_color_aug:
        c_aug = iaa.Sequential(
            [
                iaa.Sometimes(
                    p=0.8,
                    then_list=[
                        iaa.MultiplyAndAddToBrightness(),
                        iaa.AddToHueAndSaturation(
                            value_hue=(-150, 150), value_saturation=(-150, 150)),
                        iaa.GammaContrast((0.25, 4)),
                    ]
                ),
                iaa.Sometimes(
                    p=0.8,
                    then_list=[
                        iaa.CoarseDropout(
                            p=(0.2, 0.3), size_percent=(0.1, 0.25))
                    ]
                ),
                iaa.Sometimes(
                    p=0.2,
                    then_list=[iaa.Grayscale(alpha=(0.0, 1.0))]
                ),
                iaa.Sometimes(
                    p=0.2,
                    then_list=[iaa.GaussianBlur(sigma=(1.0, 2.0))]
                )
            ]
        )
    else:
        c_aug = None

    if use_shape_aug:
        s_aug = iaa.Sequential(
            [
                iaa.Sometimes(
                    p=0.7,
                    then_list=[iaa.Affine(
                        scale=(0.7, 1.3), translate_percent=(-0.2, 0.2), rotate=(-10, 10))]
                )
            ]
        )
    else:
        s_aug = None

    return transform, c_aug, s_aug


def get_aug_trans_torch_light(use_color_aug, use_shape_aug, image_size=84, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    if use_color_aug:
        c_aug = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),

            # transforms.RandomAffine()
        ])
    else:
        c_aug = None

    if use_shape_aug:
        s_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                image_size, scale=(0.5, 1.0), ratio=(3./4., 4./3.))
        ])
    else:
        s_aug = None

    return trans, c_aug, s_aug


def get_aug_trans_torch_strong(use_color_aug, use_shape_aug, image_size=84, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    if use_color_aug:
        c_aug = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5, _min=0.1, _max=2.0, image_wh=[image_size, image_size]),
        ])
    else:
        c_aug = None

    if use_shape_aug:
        s_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                image_size, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
        ])
    else:
        s_aug = None

    return trans, c_aug, s_aug


if __name__ == "__main__":
    trans = get_aug_trans_torch("train")
    print(trans)
    print(isinstance(trans, transforms.Compose))
