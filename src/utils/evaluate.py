import numpy as np
import torch
from torchmetrics import Metric

from data.transforms import to_log

def snr(true, y, e=1e-9):
    return 20 * np.log10(np.linalg.norm(true.flatten()) / (np.linalg.norm(true.flatten() - y.flatten()) + e))

def snr_t(true, y, e=1e-9):
    true_flat = true.flatten()
    y_flat = y.flatten()
    numerator = torch.norm(true_flat)
    denominator = torch.norm(true_flat - y_flat) + e
    return 20 * torch.log10(numerator / denominator)

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

class SNR_Metric(Metric):
    def __init__(self, log=False):
        super().__init__()
        self.add_state("SNR", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.log_im = log

    def update(self, output, target, a_expo=None):
        if self.log_im:
            assert a_expo is not None
            output = to_log(output, a_expo)
            target = to_log(target, a_expo)
        self.SNR += snr_t(output, target)
        self.total += output.shape[0]

    def compute(self):
        return self.SNR.float() / self.total