from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits, device=self.device)
        one_hot.scatter_(0, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.model.zero_grad()
        self.logits = self.model(image).reshape(-1)
        return self.logits

    def backward(self, ids):
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=False)

    def generate(self, target_layer):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" # noqa E501
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers. # noqa E501
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                # print(name)
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

        # def save_fmaps(key):
        #     def forward_hook(module, input, output):
        #         self.fmap_pool[key] = output.detach()
        #
        #     return forward_hook
        #
        # def save_grads(key):
        #     def backward_hook(module, grad_in, grad_out):
        #         self.grad_pool[key] = grad_out[0].detach()
        #
        #     return backward_hook
        #
        # # If any candidates are not specified, the hook is registered to all the layers.
        # for name, module in self.model.named_modules():
        #     if self.candidate_layers is None or name in self.candidate_layers:
        #         self.handlers.append(module.register_forward_hook(save_fmaps(name)))
        #         self.handlers.append(module.register_backward_hook(save_grads(name)))

    def clear_memory(self):
        self.fmap_pool.clear()
        self.grad_pool.clear()
        self.remove_hook()
        self.logits = self.logits.detach()
        del self.device, self.model, self.handlers, self.logits
        del self.fmap_pool, self.grad_pool, self.candidate_layers, self.image_shape

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        if gcam.sum() != 0:  # to prevent NaN in ouput for gcam == zeros case
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam, fmaps, grads


if __name__ == '__main__':
    import PIL.Image
    import matplotlib.pyplot as plt
    image = PIL.Image.open("cat.jpg")
    plt.imshow(image)

    norm_data = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        norm_data
    ])
    display = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224))])
    tensor = preprocess(image).unsqueeze(0)
    print(tensor.shape)

    # model = torchvision.models.resnet18(pretrained=True).eval()
    #
    # gcam = GradCAM(model, candidate_layers=None)
    #
    # outputs = gcam.forward(tensor)
    # outputs = outputs.reshape(-1)
    # probs = torch.sigmoid(outputs)
    # pred_class_id = torch.tensor(torch.argmax(probs))
    # pred_ohe = torch.zeros(len(probs))
    # pred_ohe[pred_class_id] = 1
    #
    # gcam.backward(ids=pred_class_id)
    # heatmap = gcam.generate('layer4.0')
    # heatmap = heatmap[0, 0]
    # print(heatmap.shape)
    #
    # image_resized = torch.permute(tensor[0], dims=(1, 2, 0)).detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].imshow(image_resized)
    # axs[1].imshow(heatmap)
    # plt.show()
    # exit()

    import models.resnet
    model = models.resnet.EEGResNet18Spectrum(pretrained=True).eval()
    print(model)

    gcam = GradCAM(model, candidate_layers=None)

    tensor = torch.rand((1, 25, 400, 1280), dtype=torch.float32)
    outputs = gcam.forward(tensor)
    outputs = outputs.reshape(-1)
    probs = torch.sigmoid(outputs)
    pred_class_id = torch.tensor(0)

    gcam.backward(ids=pred_class_id)
    heatmap = gcam.generate('model.layer4.0')
    heatmap = heatmap[0, 0]
    print(heatmap.shape)

    image_resized = tensor[0, 0].detach().cpu().numpy()
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(image_resized)
    axs[1].imshow(heatmap)
    plt.show()
