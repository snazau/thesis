import torch
import torchvision

import models.gradcam
import models.cameras


class EEGResNet18Spectrum(torch.nn.Module):
    def __init__(self, pretrained=False, initial_bn=False):
        super().__init__()

        self.initial_bn = initial_bn
        if self.initial_bn:
            self.bn = torch.nn.BatchNorm2d(25)

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
        _ = self.forward(x)

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
        if self.initial_bn:
            x = self.bn(x)
        return self.model(x)

    def interpret(self, x, layer='model.layer4.1.conv2', return_fm=False):
        # x.shape = (B, 25, H, W)

        gcam = models.gradcam.GradCAM(self, candidate_layers=None)
        gcam.forward(x)

        pred_class_id = torch.tensor(0)
        gcam.backward(ids=pred_class_id)

        heatmap, fmaps, grads = gcam.generate(target_layer=layer)
        # heatmap.shape = (B, 1, H, W)
        # fmaps.shape = (B, 512, H / 32, W / 32)
        # grads.shape = (B, 512, H / 32, W / 32)

        if return_fm:
            return heatmap, fmaps, grads
        else:
            return heatmap

    def cameras(self, x, layer='model.layer4.1.conv2'):
        batch_size, channels, freq_dim, time_dim = x.shape[:4]

        input_resolutions = list()
        curr_res = [freq_dim, time_dim]
        for i in range(4):
            input_resolutions.append(tuple(curr_res))
            curr_res[0] = int(curr_res[0] + 0.1 * freq_dim)
            curr_res[1] = int(curr_res[1] + 0.1 * time_dim)

        cameras = models.cameras.CAMERAS(
            model=self,
            targetLayerName=layer,
            inputResolutions=input_resolutions
        )
        heatmap = cameras.run(x, classOfInterest=0)  # (B, 1, F, T)

        return heatmap

    def channel_importance(self, x):
        # x.shape = (B, 25, H, W)

        self.zero_grad()
        # x.requires_grad = True
        x.requires_grad_()

        preds = self.forward(x)  # (B, 1)

        pred_class_id = torch.tensor(0)
        grad = torch.zeros_like(preds)
        grad[:, pred_class_id] = 1
        preds.backward(gradient=grad, retain_graph=True)
        grad_x = x.grad  # (B, 25, H, W)

        # importance = torch.sum(grad_x, dim=(2, 3))  # (B, 25)

        return grad_x


class EEGResNet18Raw(torch.nn.Module):
    def __init__(self, pretrained=False, initial_bn=False):
        super().__init__()

        self.initial_bn = initial_bn
        if self.initial_bn:
            self.bn = torch.nn.BatchNorm2d(1)

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
        _ = self.forward(x)

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
        if self.initial_bn:
            x = self.bn(x)
        return self.model(x)


if __name__ == '__main__':
    # test spectrum input
    model = EEGResNet18Spectrum(pretrained=False, initial_bn=True)
    print(model)

    input = torch.randn((4, 25, 391, 1280), dtype=torch.float32)
    heatmap = model.interpret(input)
    print('heatmap', heatmap.shape)

    heatmap_cameras = model.cameras(input)
    print('heatmap_cameras', heatmap_cameras.shape)
    exit()

    channel_importance = model.channel_importance(input)
    print('channel_importance', channel_importance.shape)
    exit()

    from thop import profile
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input, ), verbose=0)
        print(f'GMACs = {macs / 10 ** 9:.4f}')
        print(f'params(M) = {params / 10 ** 6:.4f}')
    exit()

    # test raw input
    model = EEGResNet18Raw(pretrained=False, initial_bn=True)
    print(model)

    input = torch.randn((4, 1, 25, 1280), dtype=torch.float32)

    from thop import profile
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input,), verbose=0)
        print(f'GMACs = {macs / 10 ** 9}')
        print(f'params(M) = {params / 10 ** 6}')
