import torch
import timm


class EEGEfficientNetB0Spectrum(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.pretrained = pretrained
        model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=self.pretrained)

        conv_stem_pretrained_weight = model.conv_stem.weight
        model.conv_stem = torch.nn.Conv2d(25, 64, kernel_size=3, stride=2, padding=1, bias=False)
        if self.pretrained:
            conv_stem_weight = torch.cat([torch.mean(conv_stem_pretrained_weight, dim=1, keepdim=True) for _ in range(25)], dim=1)
            model.conv_stem.weight = torch.nn.parameter.Parameter(conv_stem_weight, requires_grad=True)

        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
