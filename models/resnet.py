import torch
import torchvision


class EEGResNet18Spectrum(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.pretrained = pretrained
        self.model = torchvision.models.resnet18(pretrained=self.pretrained)

        conv1_pretrained_weight = self.model.conv1.weight
        self.model.conv1 = torch.nn.Conv2d(25, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.pretrained:
            conv1_weight = torch.cat([torch.mean(conv1_pretrained_weight, dim=1, keepdim=True) for _ in range(25)], dim=1)
            self.model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.get_embedding_dim(), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward_features(self, x, layer='layer4'):
        # x.shape = (B, 25, H, W)

        activation = {}

        def get_activation(name):
            def hook(m, i, o):
                activation[name] = o.detach()

            return hook

        self.model.__getattr__(layer).register_forward_hook(get_activation(layer))
        _ = self.model(x)

        features = activation[layer]  # (B, 512, H // 32, W // 32)
        return features

    def forward_embeddings(self, x):
        # x.shape = (B, 25, H, W)
        features = self.forward_features(x)  # (B, 512, H // 32, W // 32)
        embeddings = self.model.avgpool(features)[..., 0, 0]  # (B, 512)
        return embeddings

    def get_embedding_dim(self):
        return self.model.layer4[1].conv2.out_channels

    def forward(self, x):
        return self.model(x)


class EEGResNet18Raw(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.pretrained = pretrained
        self.model = torchvision.models.resnet18(pretrained=self.pretrained)

        conv1_pretrained_weight = self.model.conv1.weight
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.pretrained:
            conv1_weight = torch.mean(conv1_pretrained_weight, dim=1, keepdim=True)
            self.model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.get_embedding_dim(), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward_features(self, x, layer='layer4'):
        # x.shape = (B, 1, H, W)

        activation = {}

        def get_activation(name):
            def hook(m, i, o):
                activation[name] = o.detach()

            return hook

        self.model.__getattr__(layer).register_forward_hook(get_activation(layer))
        _ = self.model(x)

        features = activation[layer]  # (B, 512, H // 32, W // 32)
        return features

    def forward_embeddings(self, x):
        # x.shape = (B, 25, H, W)
        features = self.forward_features(x)  # (B, 512, H // 32, W // 32)
        embeddings = self.model.avgpool(features)[..., 0, 0]  # (B, 512)
        return embeddings

    def get_embedding_dim(self):
        return self.model.layer4[1].conv2.out_channels

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = EEGResNet18Raw(pretrained=False)
    print(model)
