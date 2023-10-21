from types import MappingProxyType

import torch
from models.efficientnet import EEGEfficientNetB0Raw
from models.resnet import EEGResNet18Raw
from models.resnet_custom import EEGResNetCustomRaw


class CRNN(torch.nn.Module):
    def __init__(
            self,
            cnn_backbone='EEGEfficientNetB0Raw',
            cnn_backbone_kwargs=MappingProxyType({}),
            cnn_backbone_pretrained_path=None,
            rnn_hidden_size=128,
            rnn_layers_num=1,
    ):
        super().__init__()

        self.cnn = globals()[cnn_backbone](**cnn_backbone_kwargs)
        # self.cnn = EEGEfficientNetB0Raw(pretrained)
        # self.cnn = EEGResNet18Raw(pretrained)
        self.embedding_dim = self.cnn.get_embedding_dim()
        if cnn_backbone_pretrained_path is not None:
            checkpoint = torch.load(cnn_backbone_pretrained_path)
            state_dict = checkpoint['model']['state_dict']
            # state_dict = {f'cnn.{key}': value for key, value in state_dict.items()}
            self.cnn.load_state_dict(state_dict)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers_num = rnn_layers_num
        self.rnn = torch.nn.LSTM(
            self.embedding_dim,
            self.rnn_hidden_size,
            num_layers=self.rnn_layers_num,
            batch_first=True,
            # bidirectional=self.bidirectional,
        )

        self.classifier = torch.nn.Linear(self.rnn_hidden_size, 1)

    def forward(self, x, h_prev=None, c_prev=None):
        # x.shape = (B, S, C, H, W)

        batch_size, seq_len, _, height, width = x.shape[:5]

        if h_prev is None:
            h_prev = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float32).to(x.device)

        if c_prev is None:
            c_prev = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float32).to(x.device)

        embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim, device=x.device)  # (B, S, E)

        # extract embeddings
        for time_step in range(seq_len):
            embedding = self.cnn.forward_embeddings(x[:, time_step])
            embeddings[:, time_step] = embedding

        rnn_out, (h_curr, c_curr) = self.rnn(embeddings, (h_prev, c_prev))
        out_scores = self.classifier(rnn_out)

        if self.training:
            return out_scores
        else:
            return out_scores, h_curr, c_curr


if __name__ == '__main__':
    checkpoint_path = r'D:\Study\asp\thesis\implementation\experiments\20230925_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_16excluded\checkpoints\best.pth.tar'
    checkpoint = torch.load(checkpoint_path)

    batch_size, seq_len, channels, height, width = 2, 6, 1, 224, 224
    input_tensor = torch.rand((batch_size, seq_len, channels, height, width))
    print(f'input_tensor = {input_tensor.shape}')

    model = CRNN(
        cnn_backbone='EEGResNetCustomRaw',
        cnn_backbone_kwargs={'input_dim': 1},
        cnn_backbone_pretrained_path='D:\\Study\\asp\\thesis\\implementation\\experiments\\20231005_EEGResNetCustomRaw_MixUp_TimeSeriesAug_raw_16excluded\\checkpoints\\best.pth.tar'
    )
    model.train()

    cnn_input_tensor = input_tensor[:, 0]
    cnn_output_features = model.cnn.forward_features(cnn_input_tensor)
    print(f'cnn_output_features = {cnn_output_features.shape}')

    cnn_output_embeddings = model.cnn.forward_embeddings(cnn_input_tensor)
    print(f'cnn_output_embeddings = {cnn_output_embeddings.shape}')

    cnn_embedding_dim = model.cnn.get_embedding_dim()
    print(f'cnn_embedding_dim = {cnn_embedding_dim}')

    print(f'input_tensor = {input_tensor.shape}')
    crnn_output = model(input_tensor)
    print(f'crnn_output = {crnn_output.shape}')
    print()

    # Inference imitation
    import utils.neural.training
    utils.neural.training.set_seed(seed=8, deterministic=True)

    model.eval()

    batch_size, seq_len, channels, height, width = 1, 10, 1, 25, 1280
    input_tensor = torch.rand((batch_size, seq_len, channels, height, width))

    # one-by-one inference
    preds_all_one_by_one = list()
    h_prev = torch.zeros((model.rnn.num_layers, batch_size, model.rnn.hidden_size), dtype=torch.float32)
    c_prev = torch.zeros((model.rnn.num_layers, batch_size, model.rnn.hidden_size), dtype=torch.float32)
    for time_idx in range(seq_len):
        with torch.no_grad():
            preds_curr, h_curr, c_curr = model(input_tensor[:, time_idx:time_idx + 1], h_prev, c_prev)
            preds_all_one_by_one.append(preds_curr.squeeze())
        h_prev, c_prev = h_curr, c_curr
        print(time_idx, preds_curr.shape, preds_curr.mean(), h_curr.mean(), c_curr.mean())

    h_one_by_one, c_one_by_one = h_curr, c_curr
    preds_all_one_by_one = torch.stack(preds_all_one_by_one).flatten()
    print('preds_all_one_by_one', preds_all_one_by_one)

    # batched inference
    preds_all_batched = list()
    h_prev = torch.zeros((model.rnn.num_layers, batch_size, model.rnn.hidden_size), dtype=torch.float32)
    c_prev = torch.zeros((model.rnn.num_layers, batch_size, model.rnn.hidden_size), dtype=torch.float32)

    with torch.no_grad():
        preds_curr, h_curr, c_curr = model(input_tensor[:, 0:5], h_prev, c_prev)
        preds_all_batched.append(preds_curr.squeeze())
    h_prev, c_prev = h_curr, c_curr

    with torch.no_grad():
        preds_curr, h_curr, c_curr = model(input_tensor[:, 5:], h_prev, c_prev)
        preds_all_batched.append(preds_curr.squeeze())

    preds_all_batched = torch.stack(preds_all_batched).flatten()
    print('preds_all_batched', preds_all_batched)

    print('diff', torch.mean(torch.abs(preds_all_one_by_one - preds_all_batched)))
