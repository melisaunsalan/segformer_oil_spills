import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha  # Tensor of shape [C] or None

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        # targets: [B, H, W]

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce_loss)  # pt = softmax prob of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()
