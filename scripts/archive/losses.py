import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """
    Encourages the predicted SR image to preserve
    spatial edges and local structure of the HR target.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # Horizontal gradients
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Vertical gradients
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y


class SRLoss(nn.Module):
    """
    Combined reconstruction + edge-aware loss.

    Total loss:

        0.8 * L1 + 0.2 * GradientLoss
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        gradient_weight: float = 0.2,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight

        self.l1 = nn.L1Loss()
        self.gradient = GradientLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        l1_loss = self.l1(prediction, target)

        gradient_loss = self.gradient(
            prediction,
            target,
        )

        total = (
            self.l1_weight * l1_loss
            + self.gradient_weight * gradient_loss
        )

        return total


if __name__ == "__main__":

    prediction = torch.rand(
        2, 3, 256, 256
    )

    target = torch.rand(
        2, 3, 256, 256
    )

    criterion = SRLoss()

    loss = criterion(
        prediction,
        target,
    )

    print("Test loss:", loss.item())
    print("✅ Loss function test passed!")