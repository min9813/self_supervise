import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, padding=1, act="relu", bn=True):
        assert act in ("relu", "sigmoid")
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        if bn:
            self.bn = nn.BatchNorm2d(output_ch)
        else:
            self.bn = None
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        return h


class ConvTransposeBlock(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, padding=1, act="relu", bn=True):
        assert act in ("relu", "sigmoid", "tanh")
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        if bn:
            self.bn = nn.BatchNorm2d(output_ch)
        else:
            self.bn = None
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        return h


class LinearBlock(nn.Module):

    def __init__(self, input_ch, output_ch, act="relu", bn=True, bias=False):
        assert act in ("relu", "sigmoid", "tanh")
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_ch, output_ch, bias=bias)
        if bn:
            self.bn = nn.BatchNorm1d(output_ch)
        else:
            self.bn = None
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        h = self.linear(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        return h


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 2, 2)


class Scaler(nn.Module):

    def __init__(self, dim):
        super(Scaler, self).__init__()
        self.in_features = dim
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1, dim))

        nn.init.normal_(self.scale)
        nn.init.normal_(self.bias)

    def forward(self, input):
        # print(type(self.scale.data), type(self.bias.data), self.scale.device)
        return input * self.scale.abs() + self.bias.abs()

    def extra_repr(self):
        return 'in_features={}'.format(
            self.in_features
        )


class MapSigmaAbs(nn.Module):

    def __init__(self, input_dim, z_dim, bn=False):
        super(MapSigmaAbs, self).__init__()
        self.bn = bn
        if self.bn:
            self.linear = LinearBlock(input_dim, z_dim, act="sigmoid")
        else:
            self.linear = nn.Linear(input_dim, z_dim)
            self.sigmoid = nn.Sigmoid()
        # self.scaler = Scaler(z_dim)
        self.scale = nn.Parameter(torch.Tensor(1, z_dim))
        self.bias = nn.Parameter(torch.Tensor(1, z_dim))

        nn.init.normal_(self.scale)
        nn.init.normal_(self.bias)

    def forward(self, x):
        x = self.linear(x)
        if not self.bn:
            x = self.sigmoid(x)
        # x = self.scaler(x)
        x = x * self.scale.abs() + self.bias.abs()
        return x


class MapSigmaSquare(nn.Module):

    def __init__(self, input_dim, z_dim, bn=False):
        super(MapSigmaSquare, self).__init__()
        self.bn = bn
        if self.bn:
            self.linear = LinearBlock(input_dim, z_dim, act="sigmoid")
        else:
            self.linear = nn.Linear(input_dim, z_dim)
            self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.ones(z_dim))
        self.bias = nn.Parameter(torch.Tensor(z_dim))

        nn.init.normal_(self.scale)
        nn.init.normal_(self.bias)

    def forward(self, x):
        x = self.linear(x)
        if not self.bn:
            x = self.sigmoid(x)

        # x = self.scale.abs() * x + self.bias.abs()
        return x


class Encoder(nn.Module):

    def __init__(self, input_ch=512, z_dim=128, layers=[512, 256, 256], sigma_bn=False):
        super(Encoder, self).__init__()

        linear_layers = [LinearBlock(input_ch, layers[0], bn=True)]
        for idx, layer in enumerate(layers[1:]):
            linear_layers.append(LinearBlock(layers[idx], layer, bn=True))
        self.models = nn.Sequential(*linear_layers)
        # self.conv1 = ConvBlock(input_ch, 32, 4, 2, 1)
        # self.conv2 = ConvBlock(32, 64, 4, 2, 1)
        # self.conv3 = ConvBlock(64, 128, 4, 2, 1)
        # self.conv4 = ConvBlock(128, 256, 4, 2, 1)

        self.mean_fc = nn.Linear(layers[-1], z_dim)
        self.sigma_fc = MapSigmaAbs(layers[-1], z_dim, bn=sigma_bn)
        nn.init.xavier_normal_(self.mean_fc.weight)
        # nn.init.xavier_normal_(self.mean_fc.bias)

    def forward(self, x):

        h = self.models(x)
        mean_v = self.mean_fc(h)
        sigma_v = self.sigma_fc(h)

        return mean_v, sigma_v


class Decoder(nn.Module):

    def __init__(self, output_ch=512, z_dim=128, layers=[256, 256, 512]):
        super(Decoder, self).__init__()

        linear_layers = [LinearBlock(z_dim, layers[0], bn=True)]
        for idx, layer in enumerate(layers[1:]):
            linear_layers.append(LinearBlock(layers[idx], layer, bn=True))
        self.models = nn.Sequential(*linear_layers)

        self.output = nn.Linear(layers[-1], output_ch)

    def forward(self, z):
        h = self.models(z)
        h = self.output(h)

        return h


class VAE(nn.Module):

    def __init__(self, input_ch, z_dim=64, layers=[512, 256, 256]):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_ch, z_dim, layers=layers)
        self.decoder = Decoder(input_ch, z_dim, layers=layers[::-1])
        self.z_dim = z_dim

    def forward(self, x):
        mean_v, sigma_v = self.encoder(x)
        # std_v = log_sigma_v.mul(0.5).exp_() # std = sqrt(var) <=> log std = 1/2 * log var
        randn = self.genereate_random(mean_v.shape, x.dtype, x.device)

        std_v = sigma_v.sqrt()
        z = std_v * randn + mean_v

        z = self.decoder(z)

        return z, mean_v, sigma_v

    def genereate_random(self, size, dtype, device=torch.device("cuda")):
        return torch.randn(size, dtype=dtype, device=device)


class VariationalModule(nn.Module):

    def __init__(self, input_ch, z_dim=64, layers=[512, 256, 256]):
        super(VariationalModule, self).__init__()
        self.encoder = Encoder(input_ch, z_dim, layers=layers)

    def forward(self, x):
        mean_v, sigma_v = self.encoder(x)

        randn = self.genereate_random(mean_v.shape, x.dtype, x.device)

        std_v = sigma_v.sqrt()
        z = std_v * randn + mean_v

        return z, mean_v, sigma_v

    def genereate_random(self, size, dtype, device=torch.device("cuda")):
        return torch.randn(size, dtype=dtype, device=device)


if __name__ == "__main__":
    # vae = VAE(3)
    # x = torch.randn(1, 3, 32, 32)
    # x_hat, mean_v, std_v = vae(x)
    # print(x_hat.size())
    # print(mean_v.size())
    # print(std_v.size())
    x = torch.randn(1, 128).cuda()
    vae = VAE(128, 64, layers=[128])
    vae.cuda().eval()
    # for name, _ in vae.named_children():
        # print(name)
    print(vae)
    z, mean_v, sigma_v = vae(x)
    # print(sigma_v)
    print(z.size())
