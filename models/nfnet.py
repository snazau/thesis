import torch
import timm

import models.gradcam


class EEGNFNetF0Spectrum(torch.nn.Module):
    def __init__(self, pretrained=False, initial_bn=False):
        super().__init__()

        self.initial_bn = initial_bn
        if self.initial_bn:
            self.bn = torch.nn.BatchNorm2d(25)

        self.pretrained = pretrained
        self.model = timm.create_model('dm_nfnet_f0.dm_in1k', pretrained=self.pretrained, in_chans=25, num_classes=1)

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
        return self.model.final_conv.out_channels

    def forward(self, x):
        if self.initial_bn:
            x = self.bn(x)
        return self.model(x)


if __name__ == '__main__':
    # test spectrum input
    model = EEGNFNetF0Spectrum(pretrained=True, initial_bn=True)
    print(model)

    input = torch.randn((4, 25, 391, 1280), dtype=torch.float32)
    # heatmap = model.interpret(input)
    # print('heatmap', heatmap.shape)

    from thop import profile
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input, ), verbose=0)
        print(f'GMACs = {macs / 10 ** 9:.4f}')
        print(f'params(M) = {params / 10 ** 6:.4f}')
