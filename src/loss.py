"""Custom loss for long tail problem.

- Author: Junghoon Kim
- Email: placidus36@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = [1] * classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        self.cls = len(pred[0])
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            for i, target_num in enumerate(target):
                # 0번 째 클래스면 본인보다 위에꺼에 약간 확률주기
                if target_num == 0:
                    true_dist[i][target_num+1] = self.smoothing
                # 마지막 클래스면 본인보다 아래꺼에 약간 확률주기
                elif target_num == self.cls-1:
                    true_dist[i][target_num-1] = self.smoothing
                # 그 외의 경우는 본인 위 아래로 약간 확률주기
                else:
                    true_dist[i][target_num+1] = self.smoothing/2
                    true_dist[i][target_num-1] = self.smoothing/2
            #true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))/torch.sum(self.weight)

class CustomCriterion:
    """Custom Criterion."""

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax"):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            self.no_of_classes = len(samples_per_cls)
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "logit_adjustment_loss":
            tau = 1.0
            self.logit_adj_val = (
                torch.tensor(tau * np.log(self.frequency_per_cls))
                .float()
                .to(self.device)
            )
            self.logit_adj_val = (
                self.logit_adj_val.half() if fp16 else self.logit_adj_val.float()
            )
            self.logit_adj_val = self.logit_adj_val.to(device)
            self.criterion = self.logit_adjustment_loss
        elif loss_type == "label_smoothing":
            self.criterion = LabelSmoothingLoss(classes=self.no_of_classes)

    def __call__(self, logits, labels):
        """Call criterion."""
        return self.criterion(logits, labels)

    def logit_adjustment_loss(self, logits, labels):
        """Logit adjustment loss."""
        logits_adjusted = logits + self.logit_adj_val.repeat(labels.shape[0], 1)
        loss = F.cross_entropy(input=logits_adjusted, target=labels)
        return loss
