from torchvision.transforms import transforms


def get_aug_trans(use_color_aug, use_shape_aug, use_mix_aug, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # range [0.0, 1.0] -> [-1.0,1.0]
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform
