import torch
import torch.nn.functional


class BCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.criterion(batch['outputs'], batch['target'].float().unsqueeze(1))


class BCELossWithTimeToClosestSeizure(torch.nn.Module):
    def __init__(self, max_time, min_weight):
        super().__init__()
        assert max_time > 0
        assert 0 <= min_weight <= 1

        self.max_time = max_time
        self.min_weight = min_weight

    def forward(self, batch):
        outputs = batch['outputs']
        target = batch['target'].float().unsqueeze(1)
        time_to_closest_seizure = batch['time_to_closest_seizure'].to(outputs.device).float()

        batch_size = outputs.shape[0]
        batch_weights = torch.ones((batch_size, ), dtype=torch.float32, device=outputs.device)

        mask = (time_to_closest_seizure < self.max_time) * (time_to_closest_seizure > 0)
        batch_weights[mask] = ((1 - self.min_weight) * time_to_closest_seizure[mask] / self.max_time) + self.min_weight
        batch_weights = batch_weights.unsqueeze(1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, target, weight=batch_weights)
        return loss
