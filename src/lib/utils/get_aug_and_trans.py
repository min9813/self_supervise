import imgaug.augmenters as iaa
from torchvision.transforms import transforms


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
                        p=(0.2,0.3), size_percent=(0.1, 0.25))
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
                    then_list=[iaa.Affine(scale=(0.7,1.3), translate_percent=(-0.2,0.2), rotate=(-10,10))]
                )
            ]
        )
    else:
        s_aug = None

    return transform, c_aug, s_aug
