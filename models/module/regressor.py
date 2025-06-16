import math
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPRegressor, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x):
        x = self.global_pool(x)
        x = self.fc(x)
        return x


class ConvRegressor(nn.Module):
    def __init__(
        self,
        in_channels,
        image_size,
        hidden_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        super(MLPRegressor, self).__init__()

        self.flatten = nn.Flatten()

        self.downsample = []
        num_downsamples = int(math.log2(image_size))  # 6
        for i in range(num_downsamples):
            self.downsample.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.downsample.append(norm_layer(in_channels))
            self.downsample.append(nn.ReLU())
        self.downsample = nn.Sequential(*self.downsample)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
