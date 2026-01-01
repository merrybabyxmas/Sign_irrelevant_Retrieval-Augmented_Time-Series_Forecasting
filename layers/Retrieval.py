import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        similarity_type='cosine',  # 'cosine', 'pearson', or 'phase_aware'
        phase_multiplier=4,  # k value for phase-aware similarity
        neg_sign_weight=1.0,  # attenuation factor for negative cosine similarity
        shift_range=0,  # maximum temporal shift for shift-invariant similarity
        mixture_alpha=0.0,  # mixture weight: α·cos(θ) + (1-α)·cos(kθ)
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key

        # Phase-aware similarity configuration
        self.similarity_type = similarity_type
        self.phase_multiplier = phase_multiplier
        self.neg_sign_weight = neg_sign_weight
        self.shift_range = shift_range
        self.mixture_alpha = mixture_alpha

        # Validate similarity type
        assert similarity_type in ['cosine', 'pearson', 'phase_aware'], \
            f"similarity_type must be 'cosine', 'pearson', or 'phase_aware', got {similarity_type}"

        # Validate neg_sign_weight
        assert 0.0 < neg_sign_weight <= 1.0, \
            f"neg_sign_weight must be in (0, 1], got {neg_sign_weight}"

        # Validate shift_range
        assert shift_range >= 0, \
            f"shift_range must be non-negative, got {shift_range}"

        # Validate mixture_alpha
        assert 0.0 <= mixture_alpha <= 1.0, \
            f"mixture_alpha must be in [0, 1], got {mixture_alpha}"
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)

        return mg, offset

    def temporal_shift(self, x, delta):
        """
        Shift tensor along the time (feature) dimension by delta positions.

        Args:
            x: Tensor of shape (..., features)
            delta: Number of positions to shift (positive = right, negative = left)

        Returns:
            Shifted tensor with same shape as input, padded with zeros
        """
        if delta == 0:
            return x

        if delta > 0:
            # Shift right: remove first delta elements, pad zeros at end
            return torch.cat([x[..., delta:], torch.zeros_like(x[..., :delta])], dim=-1)
        else:
            # Shift left: remove last |delta| elements, pad zeros at start
            return torch.cat([torch.zeros_like(x[..., :-delta]), x[..., :delta]], dim=-1)

    def compute_base_similarity(self, bx_norm, ax_norm):
        """
        Compute base similarity between normalized query and key patches.

        This encapsulates the similarity logic (cosine, pearson, or phase-aware with sign attenuation)
        to enable reuse in shift-invariant similarity computation.

        Args:
            bx_norm: Normalized query patches, shape (G, B, features)
            ax_norm: Normalized key patches, shape (G, T, features)

        Returns:
            Similarity tensor, shape (G, B, T)
        """
        if self.similarity_type == 'pearson' or self.similarity_type == 'cosine':
            # Original implementation: Pearson correlation / cosine similarity
            # Both are equivalent after centering (mean removal)
            return torch.bmm(bx_norm, ax_norm.transpose(-1, -2))

        elif self.similarity_type == 'phase_aware':
            # Phase-aware cosine similarity for time-series retrieval
            eps = 1e-8

            # Compute standard cosine similarity: cos(θ)
            cos_sim = torch.bmm(bx_norm, ax_norm.transpose(-1, -2))

            # Compute angle θ from cosine similarity
            # Clamp to avoid numerical issues with acos
            cos_sim_clamped = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
            theta = torch.acos(cos_sim_clamped)

            # Apply phase-aware transformation: ρ = cos(k * θ)
            phase_sim = torch.cos(self.phase_multiplier * theta)

            # Apply mixture softening: α·cos(θ) + (1-α)·cos(kθ)
            if self.mixture_alpha > 0:
                phase_sim = self.mixture_alpha * torch.cos(theta) + (1.0 - self.mixture_alpha) * phase_sim

            # Apply sign-dependent attenuation
            # When cos(θ) < 0 (anti-correlated), attenuate the similarity by neg_sign_weight
            sign_weight = torch.where(
                cos_sim >= 0,
                torch.ones_like(cos_sim),
                self.neg_sign_weight * torch.ones_like(cos_sim)
            )

            return phase_sim * sign_weight

    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            # Normalize once for all shifts (efficiency)
            bx_norm = F.normalize(bx, dim=2)
            ax_norm = F.normalize(ax, dim=2)

            if self.shift_range == 0:
                # No shift invariance: use standard similarity computation
                cur_sim = self.compute_base_similarity(bx_norm, ax_norm)

            else:
                # Shift-invariant similarity: compute similarity for all shifts and take max
                # This allows the model to handle small temporal misalignments
                sims_at_shifts = []

                for delta in range(-self.shift_range, self.shift_range + 1):
                    # Shift the key patches along the time dimension
                    ax_shifted = self.temporal_shift(ax_norm, delta)

                    # Compute similarity at this shift
                    sim_at_delta = self.compute_base_similarity(bx_norm, ax_shifted)
                    sims_at_shifts.append(sim_at_delta)

                # Stack all similarities and take max across shifts
                # Shape: (num_shifts, G, B, T) -> max over dim=0 -> (G, B, T)
                sims_stacked = torch.stack(sims_at_shifts, dim=0)
                cur_sim = torch.max(sims_stacked, dim=0).values

            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)

        return sim
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals