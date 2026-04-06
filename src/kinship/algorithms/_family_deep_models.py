from __future__ import annotations

import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as f


class SmallFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
        )
        n_features = functools.reduce(operator.mul, list(self.features(torch.rand(1, *(6, 64, 64))).shape))
        self.classifier = nn.Sequential(nn.Linear(n_features, 640), nn.ReLU(), nn.Linear(640, 1))

    def forward(self, parent_image, children_image):
        x = torch.cat((parent_image, children_image), 1)
        x = self.features(x.float())
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SmallSiameseFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
        )
        n_features = functools.reduce(operator.mul, list(self.features(torch.rand(1, *(3, 64, 64))).shape))
        self.classifier = nn.Sequential(nn.Linear(n_features, 640), nn.ReLU(), nn.Linear(640, 1))

    def forward(self, parent_image, children_image):
        parent_features = self.features(parent_image.float())
        children_features = self.features(children_image.float())
        distance = torch.abs(parent_features - children_features)
        x = torch.flatten(distance, 1)
        x = f.normalize(x, dim=0, p=2)
        x = self.classifier(x)
        return x, torch.flatten(parent_features, 1), torch.flatten(children_features, 1)


class VGGFaceEncoder(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)

    def encode(self, x):
        x = f.relu(self.conv_1_1(x))
        x = f.relu(self.conv_1_2(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv_2_1(x))
        x = f.relu(self.conv_2_2(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv_3_1(x))
        x = f.relu(self.conv_3_2(x))
        x = f.relu(self.conv_3_3(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv_4_1(x))
        x = f.relu(self.conv_4_2(x))
        x = f.relu(self.conv_4_3(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv_5_1(x))
        x = f.relu(self.conv_5_2(x))
        x = f.relu(self.conv_5_3(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc6(x))
        x = f.dropout(x, 0.5, self.training)
        return self.fc7(x)

    def load_weights(self, path: str) -> None:
        import torchfile

        model = torchfile.load(path)
        counter = 1
        block = 1
        for layer in model.modules:
            if layer.weight is None:
                continue
            if block <= 5:
                try:
                    self_layer = getattr(self, f"conv_{block}_{counter}")
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                except Exception:
                    continue
            else:
                try:
                    self_layer = getattr(self, f"fc{block}")
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                except Exception:
                    continue


class VGGFaceSiamese(nn.Module):
    def __init__(self, vgg_weights_path: str | None = None):
        super().__init__()
        self.encoder = VGGFaceEncoder(input_channels=3)
        self.classifier = nn.Linear(4096, 1)
        if vgg_weights_path:
            self.encoder.load_weights(vgg_weights_path)

    def forward(self, parent_image, children_image):
        parent_features = self.encoder.encode(parent_image.float())
        children_features = self.encoder.encode(children_image.float())
        distance = torch.abs(parent_features - children_features)
        x = self.classifier(distance)
        return x, torch.flatten(parent_features, 1), torch.flatten(children_features, 1)


class VGGFaceMutiChannel(nn.Module):
    def __init__(self, vgg_weights_path: str | None = None):
        super().__init__()
        self.encoder = VGGFaceEncoder(input_channels=6)
        self.classifier = nn.Linear(4096, 1)
        if vgg_weights_path:
            self.encoder.load_weights(vgg_weights_path)

    def forward(self, parent_image, children_image):
        input_tensor = torch.cat((parent_image, children_image), 1)
        x = self.encoder.encode(input_tensor.float())
        x = self.classifier(f.relu(x))
        flattened = torch.flatten(input_tensor, 1)
        return x, flattened, flattened


class KinFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.backbone = InceptionResnetV1(pretrained="vggface2")
        self.classifier = nn.Sequential(
            nn.Linear(512, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, parent_image, children_image):
        parent_features = self.backbone(parent_image)
        children_features = self.backbone(children_image)
        distance = torch.abs(parent_features - children_features)
        x = self.classifier(torch.flatten(distance, 1))
        return x, torch.flatten(parent_features, 1), torch.flatten(children_features, 1)
