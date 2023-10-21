import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(0, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(0, 1)),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class EEGResNetCustomRaw(torch.nn.Module):
    def __init__(self, input_dim, block=ResidualBlock, layers=(3, 4, 6, 3), num_classes=1):
        super(EEGResNetCustomRaw, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.layer0 = self._make_layer(block, 64, layers[0], kernel_size=(1, 3), stride=(1, 1))
        self.layer1 = self._make_layer(block, 128, layers[1], kernel_size=(1, 3), stride=(1, 2))
        self.layer2 = self._make_layer(block, 256, layers[2], kernel_size=(1, 3), stride=(1, 2))
        self.layer3 = self._make_layer(block, 512, layers[3], kernel_size=(1, 3), stride=(1, 2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, kernel_size, stride=1):
        if isinstance(stride, int):
            stride = (stride, stride)

        downsample = None
        if any([s != 1 for s in stride]) or self.inplanes != planes:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size))

        return torch.nn.Sequential(*layers)

    def forward_features(self, x, layer='layer3'):
        # x.shape = (B, 1, H, W)

        activation = {}

        def get_activation(name):
            def hook(m, i, o):
                activation[name] = o.detach()

            return hook

        self.__getattr__(layer).register_forward_hook(get_activation(layer))
        _ = self(x)

        features = activation[layer]  # (B, 512, H // 2, W // 32)
        return features

    def forward_embeddings(self, x):
        # x.shape = (B, 1, H, W)
        features = self.forward_features(x)  # (B, 512, H // 2, W // 32)
        embeddings = self.avgpool(features)[..., 0, 0]  # (B, 512)
        return embeddings

    def get_embedding_dim(self):
        return self.layer3[2].conv2[0].out_channels

    def forward(self, x):
        x = self.conv1(x)
        # print(f'x_conv1 = {x.shape}')
        x = self.maxpool(x)
        # print(f'x_maxpool = {x.shape}')
        x = self.layer0(x)
        # print(f'x_layer0 = {x.shape}')
        x = self.layer1(x)
        # print(f'x_layer1 = {x.shape}')
        x = self.layer2(x)
        # print(f'x_layer2 = {x.shape}')
        x = self.layer3(x)
        # print(f'x_layer3 = {x.shape}')

        x = self.avgpool(x)
        # print(f'x_avgpool = {x.shape}')
        x = x.view(x.size(0), -1)
        # print(f'x_view = {x.shape}')
        x = self.fc(x)
        # print(f'x_fc = {x.shape}')

        return x


if __name__ == '__main__':
    model = EEGResNetCustomRaw(
        input_dim=1,
        block=ResidualBlock,
        layers=[3, 4, 6, 3],
    )

    batch_size, channels, height, width = 2, 1, 25, 1280
    input_tensor = torch.rand((batch_size, channels, height, width))
    print(f'input_tensor = {input_tensor.shape}')

    output_features = model.forward_features(input_tensor)
    print(f'output_features = {output_features.shape}')

    output_embeddings = model.forward_embeddings(input_tensor)
    print(f'output_embeddings = {output_embeddings.shape}')

    cnn_embedding_dim = model.get_embedding_dim()
    print(f'cnn_embedding_dim = {cnn_embedding_dim}')

    output_tensor = model(input_tensor)
    print(f'output_tensor = {output_tensor.shape}')
