import torch
import random

class SpecAugmentGPU:
    def __init__(
        self,
        freq_mask_param=15,
        time_mask_param=35,
        num_freq_masks=2,
        num_time_masks=2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, specs):
        """
        specs: (B, T, F) tensor on GPU
        """
        B, T, F = specs.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,), device=specs.device)
            f0 = torch.randint(0, max(1, F - f.item()), (1,), device=specs.device)
            specs[:, :, f0 : f0 + f] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,), device=specs.device)
            t0 = torch.randint(0, max(1, T - t.item()), (1,), device=specs.device)
            specs[:, t0 : t0 + t, :] = 0

        return specs