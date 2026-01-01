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
        phase_multipliers=[4],  # List of k values for phase-aware similarity
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
        # Handle backward compatibility if single int passed or list
        if isinstance(phase_multipliers, int):
             self.phase_multipliers = [phase_multipliers]
        else:
             self.phase_multipliers = phase_multipliers

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

        # Print retrieval configuration for verification
        print(f"Retrieval Configuration:")
        print(f"  Similarity Type: {self.similarity_type}")
        print(f"  Phase Multipliers: {self.phase_multipliers}")
        print(f"  Top-m per multiplier: {self.topm}")
        print(f"  Total Candidates per Query: {len(self.phase_multipliers) * self.topm}")
        
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

    def compute_base_similarity(self, bx_norm, ax_norm, multipliers=None):
        """
        Compute base similarity between normalized query and key patches.

        This encapsulates the similarity logic (cosine, pearson, or phase-aware with sign attenuation)
        to enable reuse in shift-invariant similarity computation.

        Args:
            bx_norm: Normalized query patches, shape (G, B, features)
            ax_norm: Normalized key patches, shape (G, T, features)
            multipliers: Optional list of multipliers to compute. If None, uses first element of self.phase_multipliers

        Returns:
            Similarity tensor, shape (G, B, T) if multipliers is single value or None
            List of Similarity tensors if multipliers is a list
        """
        # Determine if we should return a list or single tensor
        return_list = isinstance(multipliers, list)

        if multipliers is None:
             multipliers = [self.phase_multipliers[0]]
             return_list = False
        elif not isinstance(multipliers, list):
             multipliers = [multipliers]

        if self.similarity_type == 'pearson' or self.similarity_type == 'cosine':
            # Original implementation: Pearson correlation / cosine similarity
            # Both are equivalent after centering (mean removal)
            sim = torch.bmm(bx_norm, ax_norm.transpose(-1, -2))
            if return_list:
                # If caller expects a list (e.g. iterating over multipliers but similarity type is cosine),
                # we should probably return the same sim for each multiplier?
                # Or just return list with one element?
                # The caller logic in periodic_batch_corr will likely zip results with multipliers.
                # If similarity type is not phase_aware, multipliers don't matter.
                # But to keep list structure consistent:
                return [sim] * len(multipliers)
            return sim

        elif self.similarity_type == 'phase_aware':
            # Phase-aware cosine similarity for time-series retrieval
            eps = 1e-8

            # Compute standard cosine similarity: cos(θ)
            cos_sim = torch.bmm(bx_norm, ax_norm.transpose(-1, -2))

            # Compute angle θ from cosine similarity
            # Clamp to avoid numerical issues with acos
            cos_sim_clamped = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
            theta = torch.acos(cos_sim_clamped)

            # Apply sign-dependent attenuation
            # When cos(θ) < 0 (anti-correlated), attenuate the similarity by neg_sign_weight
            sign_weight = torch.where(
                cos_sim >= 0,
                torch.ones_like(cos_sim),
                self.neg_sign_weight * torch.ones_like(cos_sim)
            )

            results = []
            for k in multipliers:
                # Apply phase-aware transformation: ρ = cos(k * θ)
                phase_sim = torch.cos(k * theta)

                # Apply mixture softening: α·cos(θ) + (1-α)·cos(kθ)
                if self.mixture_alpha > 0:
                    phase_sim = self.mixture_alpha * torch.cos(theta) + (1.0 - self.mixture_alpha) * phase_sim

                results.append(phase_sim * sign_weight)

            if not return_list:
                return results[0]
            return results

    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        # Initialize lists to hold results for each multiplier
        # If not phase_aware, we effectively have 1 multiplier.
        # But to unify logic, we iterate over self.phase_multipliers if phase_aware

        target_multipliers = self.phase_multipliers if self.similarity_type == 'phase_aware' else [1]

        sims_per_multiplier = [[] for _ in target_multipliers]

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
                cur_sims = self.compute_base_similarity(bx_norm, ax_norm, multipliers=target_multipliers)
                # Ensure cur_sims is a list
                if not isinstance(cur_sims, list):
                    cur_sims = [cur_sims]

                for idx, val in enumerate(cur_sims):
                    sims_per_multiplier[idx].append(val)

            else:
                # Shift-invariant similarity: compute similarity for all shifts and take max
                # This allows the model to handle small temporal misalignments

                # We need to collect sims for all multipliers across all shifts
                # sims_across_shifts[shift_idx][multiplier_idx]
                sims_across_shifts = []

                for delta in range(-self.shift_range, self.shift_range + 1):
                    # Shift the key patches along the time dimension
                    ax_shifted = self.temporal_shift(ax_norm, delta)

                    # Compute similarity at this shift for ALL multipliers
                    sim_at_delta = self.compute_base_similarity(bx_norm, ax_shifted, multipliers=target_multipliers)

                    if not isinstance(sim_at_delta, list):
                        sim_at_delta = [sim_at_delta]

                    sims_across_shifts.append(sim_at_delta)

                # Now we need to transpose: [shift][mult] -> [mult][shift]
                # Then stack shifts and max
                num_multipliers = len(target_multipliers)
                for m_idx in range(num_multipliers):
                    sims_for_this_multiplier = [sims_across_shifts[s][m_idx] for s in range(len(sims_across_shifts))]

                    # Stack all similarities and take max across shifts
                    # Shape: (num_shifts, G, B, T) -> max over dim=0 -> (G, B, T)
                    sims_stacked = torch.stack(sims_for_this_multiplier, dim=0)
                    cur_sim = torch.max(sims_stacked, dim=0).values

                    sims_per_multiplier[m_idx].append(cur_sim)

        # Concatenate chunks for each multiplier
        # final_sims is a list of tensors (G, B, T), one for each multiplier
        final_sims = [torch.cat(sim_list, dim=2) for sim_list in sims_per_multiplier]

        return final_sims
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        # Get list of similarity matrices (one per multiplier)
        sim_list = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # List of (G, B, T)

        all_top_indices = []
        all_top_scores = []

        # Prepare sliding index mask if training (common for all multipliers)
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
        else:
            self_mask = None
            
        for sim in sim_list:
            if train:
                 sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

            sim_reshaped = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T

            # Top m
            top_res = torch.topk(sim_reshaped, self.topm, dim=1) # indices and values

            all_top_indices.append(top_res.indices)
            all_top_scores.append(top_res.values)
        
        # Concatenate across multipliers
        # indices: (G*B, L*m)
        final_indices = torch.cat(all_top_indices, dim=1)
        # scores: (G*B, L*m)
        final_scores = torch.cat(all_top_scores, dim=1)
        
        # Reshape back to G, B
        final_indices = final_indices.reshape(self.n_period, bsz, -1) # G, B, L*m
        final_scores = final_scores.reshape(self.n_period, bsz, -1) # G, B, L*m

        # DEBUG
        # print(f"DEBUG: final_indices={final_indices}")
        # print(f"DEBUG: final_scores={final_scores}")

        # Calculate probabilities
        # Softmax over the L*m dimension
        # Move to CPU for safety as y_data_all is likely on CPU
        ranking_prob = F.softmax(final_scores / self.temperature, dim=2).cpu() # G, B, L*m

        # Move indices to CPU for gathering from y_data_all (which is on CPU)
        final_indices = final_indices.cpu()

        # We need to gather values from y_data_all
        # y_data_all: (G, T, P*C)
        y_data_all = self.y_data_all_mg.flatten(start_dim=2)

        # Gather logic
        # We need to expand indices to gather from y_data_all
        # indices is (G, B, L*m)
        # y_data_all is (G, T, P*C)

        # We need to treat G as batch dimension too?
        # gather expects index to have same number of dimensions as input
        # We want to gather along dim 1 (T) of y_data_all

        # Reshape for gather
        # Flatten G and B? No, y_data_all has G but not B.
        # We can expand y_data_all to (G, B, T, P*C) but that is huge.

        # Instead, let's process per Group G? Or use advanced indexing?

        # y_data_all: (G, T, D) where D = P*C
        # final_indices: (G, B, K) where K = L*m

        # We want output: (G, B, K, D)

        # Let's iterate over G? G is small (e.g. 3).
        gathered_values_list = []
        for g in range(self.n_period):
            y_g = y_data_all[g] # (T, D)
            idx_g = final_indices[g] # (B, K)

            # y_g[idx_g] should work -> (B, K, D)
            # because y_g is 2D and idx_g is 2D with integers < T.
            # idx_g is on CPU, y_g is on CPU (default for y_data_all_mg)
            gathered_vals = y_g[idx_g.long()]
            gathered_values_list.append(gathered_vals)
            
        gathered_values = torch.stack(gathered_values_list, dim=0) # (G, B, K, D)

        # Weighted aggregation
        # ranking_prob: (G, B, K)
        # gathered_values: (G, B, K, D)

        # Expand ranking_prob to (G, B, K, 1)
        weights = ranking_prob.unsqueeze(-1)
        
        # Sum over K
        pred_from_retrieval = (weights * gathered_values).sum(dim=2) # (G, B, D)
        
        # Reshape to (G, B, P, C)
        pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, channels)
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
