import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            # Thêm Convolutional Layer với 32 kernel, kích thước kernel 3*3
            # dùng hàm relu làm hàm kích hoạt cho lớp đầu tiên
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            # Thêm Convolutional Layer
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            # Thêm Max pooling Layer
            nn.MaxPool2d(2),
        )
        size = (img_size - 4) // 2  # 128 -> 126 -> 124 -> 62
        self.classifier = nn.Sequential(
            # Flatten Layer chuyển từ tensor sang vector
            nn.Flatten(),
            # Thêm Fully Connected Layer với 128 nodes và dùng hàm relu
            nn.Linear(32 * size * size, 128), nn.ReLU(),
            nn.Dropout(0.5),
            # Output Layer có 3 node, softmax đã nằm sẵn trong CrossEntropyLoss
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_bn(c_in, c_out, k, stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    # 4 nhánh song song rồi ghép lại theo kênh
    def __init__(self, c_in, c1, c3, c5, cp):
        super().__init__()
        self.b1 = conv_bn(c_in, c1, 1)
        self.b2 = nn.Sequential(conv_bn(c_in, c3[0], 1), conv_bn(c3[0], c3[1], 3))
        self.b3 = nn.Sequential(conv_bn(c_in, c5[0], 1), conv_bn(c5[0], c5[1], 5))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1), conv_bn(c_in, cp, 1))
        self.out_channels = c1 + c3[1] + c5[1] + cp

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class ResidualBlock(nn.Module):
    # output = F(x) + x
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
        )
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return torch.relu(self.body(x) + self.shortcut(x))


class ComplexCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(1, 32, 7, stride=2),    # 224 -> 112
            nn.MaxPool2d(3, 2, padding=1),  # -> 56
            conv_bn(32, 64, 1),
            conv_bn(64, 96, 3),
            nn.MaxPool2d(3, 2, padding=1),  # -> 28
        )
        inc1 = InceptionBlock(96, 32, (48, 64), (8, 16), 16)                  # 128 kênh
        inc2 = InceptionBlock(inc1.out_channels, 64, (64, 96), (16, 32), 32)  # 224 kênh
        self.inception = nn.Sequential(inc1, inc2, nn.MaxPool2d(3, 2, padding=1))  # -> 14
        self.residual = nn.Sequential(
            ResidualBlock(inc2.out_channels, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 384, stride=2),  # -> 7
            ResidualBlock(384, 384),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(384, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.residual(self.inception(self.stem(x))))


def resnet50_transfer(num_classes=3):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def mobilenet_transfer(num_classes=3):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def unfreeze_last_stages(m):
    # giai đoạn fine-tune: mở khóa các tầng cuối
    if isinstance(m, models.ResNet):
        layers = [m.layer4]
    else:
        layers = m.features[-4:]
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = True


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
