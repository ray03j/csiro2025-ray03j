import torch
from torch import nn
import torch.nn.functional as F


def get_loss(cfg):
    return MyLoss(cfg)

class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg

        # 基本は SmoothL1（元コードと同じ）
        self.criterion = nn.SmoothL1Loss(beta=5.0, reduction="mean")

        # 将来の拡張用（今は使わないが cfg で制御できる）
        self.use_weights = getattr(cfg.loss, "use_weights", False)
        self.weights = getattr(cfg.loss, "weights", None)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (tuple of Tensor): (total, gdm, green, clover, dead)
                各 Tensor の形状 = (batch,)
            y_true (Tensor): (batch, 5)
                各列 = [Green, Dead, Clover, GDM, Total]
        Returns:
            dict:
                {
                    "loss": total_loss,
                    "loss_green": ...,
                    "loss_dead": ...,
                    "loss_clover": ...,
                    "loss_gdm": ...,
                    "loss_total": ...,
                }
        """
        return_dict = {}
        total, gdm, green, clover, dead = y_pred

        # 個別損失を計算
        l_green  = self.criterion(green.squeeze(),  y_true[:,0])
        l_dead   = self.criterion(dead.squeeze(),   y_true[:,1])
        l_clover = self.criterion(clover.squeeze(), y_true[:,2])
        l_gdm    = self.criterion(gdm.squeeze(),    y_true[:,3])
        l_total  = self.criterion(total.squeeze(),  y_true[:,4])

        # 辞書に格納
        return_dict["loss_green"]  = l_green
        return_dict["loss_dead"]   = l_dead
        return_dict["loss_clover"] = l_clover
        return_dict["loss_gdm"]    = l_gdm
        return_dict["loss_total"]  = l_total

        # 損失をまとめる
        losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])

        if self.use_weights and self.weights is not None:
            w = torch.as_tensor(self.weights, device=losses.device, dtype=losses.dtype)
            w = w / w.sum()
            total_loss = (losses * w).sum()
        else:
            total_loss = losses.mean()

        return_dict["loss"] = total_loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
