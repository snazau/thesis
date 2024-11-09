import torch
import timm

import models.gradcam


class EEGEfficientNetB0Spectrum(torch.nn.Module):
    def __init__(self, pretrained=False, initial_bn=False):
        super().__init__()

        self.initial_bn = initial_bn
        if self.initial_bn:
            self.bn = torch.nn.BatchNorm2d(25)

        self.pretrained = pretrained
        self.model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=self.pretrained)

        conv_stem_pretrained_weight = self.model.conv_stem.weight
        self.model.conv_stem = torch.nn.Conv2d(25, self.model.conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        if self.pretrained:
            conv_stem_weight = torch.cat([torch.mean(conv_stem_pretrained_weight, dim=1, keepdim=True) for _ in range(25)], dim=1)
            self.model.conv_stem.weight = torch.nn.parameter.Parameter(conv_stem_weight, requires_grad=True)

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.get_embedding_dim(), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward_features(self, x):
        # x.shape = (B, 25, H, W)
        if self.initial_bn:
            x = self.bn(x)
        return self.model.forward_features(x)  # (B, 1280, H // 32, W // 32)

    def forward_embeddings(self, x):
        # x.shape = (B, 25, H, W)
        if self.initial_bn:
            x = self.bn(x)
        features = self.forward_features(x)  # (B, 1280, H // 32, W // 32)
        embeddings = self.model.global_pool(features)  # (B, 1280)
        return embeddings

    def get_embedding_dim(self):
        return self.model.conv_head.out_channels

    def forward(self, x):
        if self.initial_bn:
            x = self.bn(x)
        return self.model(x)


class EEGEfficientNetB0Raw(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.pretrained = pretrained
        self.model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=self.pretrained)

        conv_stem_pretrained_weight = self.model.conv_stem.weight
        self.model.conv_stem = torch.nn.Conv2d(1, self.model.conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        if self.pretrained:
            conv_stem_weight = torch.mean(conv_stem_pretrained_weight, dim=1, keepdim=True)
            self.model.conv_stem.weight = torch.nn.parameter.Parameter(conv_stem_weight, requires_grad=True)

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.get_embedding_dim(), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward_features(self, x):
        # x.shape = (B, 1, H, W)
        return self.model.forward_features(x)  # (B, 1280, H // 32, W // 32)

    def forward_embeddings(self, x):
        # x.shape = (B, 1, H, W)
        features = self.forward_features(x)  # (B, 1280, H // 32, W // 32)
        embeddings = self.model.global_pool(features)  # (B, 1280)
        return embeddings

    def get_embedding_dim(self):
        return self.model.conv_head.out_channels

    def forward(self, x):
        return self.model(x)

    def interpret(self, x, layer='model.conv_head'):
        # x.shape = (B, 1, C, T)

        gcam = models.gradcam.GradCAM(self, candidate_layers=None)
        gcam.forward(x)

        pred_class_id = torch.tensor(0)
        gcam.backward(ids=pred_class_id)

        heatmap = gcam.generate(target_layer=layer)

        return heatmap


if __name__ == '__main__':
    model = EEGEfficientNetB0Spectrum(pretrained=True, initial_bn=True)

    input = torch.randn((4, 25, 391, 1280), dtype=torch.float32)

    from thop import profile
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input,), verbose=0)
        print(f'GMACs = {macs / 10 ** 9:.4f}')
        print(f'params(M) = {params / 10 ** 6:.4f}')
    exit()

    import models.resnet
    model = models.resnet.EEGResNet18Raw(pretrained=False)
    print(model)
