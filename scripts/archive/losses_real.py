import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    """
    Encourages the SR output to preserve spatial edges/details.
    """

    def forward(self, pred, target):

        # Horizontal gradients
        pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Vertical gradients
        pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = torch.mean(
            torch.abs(pred_x - target_x)
        )

        loss_y = torch.mean(
            torch.abs(pred_y - target_y)
        )

        return (loss_x + loss_y) / 2.0


class SRLoss(nn.Module):
    """
    Combined pixel + gradient loss.

    L = 0.8 * L1 + 0.2 * Gradient
    """

    def __init__(
        self,
        l1_weight=0.8,
        gradient_weight=0.2,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight

        self.l1 = nn.L1Loss()
        self.gradient = GradientLoss()

    def forward(self, pred, target):

        pixel_loss = self.l1(pred, target)

        gradient_loss = self.gradient(
            pred,
            target
        )

        total = (
            self.l1_weight * pixel_loss
            + self.gradient_weight * gradient_loss
        )

        return total, pixel_loss, gradient_loss