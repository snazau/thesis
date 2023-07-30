import torch
import torchvision


class EEGResNet18Spectrum(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.pretrained = pretrained
        model = torchvision.models.resnet18(pretrained=self.pretrained)

        conv1_pretrained_weight = model.conv1.weight
        model.conv1 = torch.nn.Conv2d(25, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.pretrained:
            conv1_weight = torch.cat([torch.mean(conv1_pretrained_weight, dim=1, keepdim=True) for _ in range(25)], dim=1)
            model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


class EEGResNet18Raw(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.pretrained = pretrained
        model = torchvision.models.resnet18(pretrained=self.pretrained)

        conv1_pretrained_weight = model.conv1.weight
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.pretrained:
            conv1_weight = torch.mean(conv1_pretrained_weight, dim=1, keepdim=True)
            model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
