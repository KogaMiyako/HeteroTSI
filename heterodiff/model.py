import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from info_nce import InfoNCE
from .diffusion import denoising_network


class LightSAEDataEmbedding(nn.Module):
    """
    LightSAE data-dependent channel embedding.
    Computes channel statistics from observable data, then applies
    SAE decomposition: shared base + low-rank auxiliary with gating.

    Channel embedding:
      e_i = stats_i @ W_sh + stats_i @ (sum_k g_{i,k} * L_k @ R)
    where stats_i is computed from X_Tilde and cond_mask.
    """

    def __init__(self, num_channels, stats_dim, embed_dim, rank=4, pool_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim
        self.rank = rank
        self.pool_size = pool_size

        self.W_sh = nn.Parameter(torch.Tensor(stats_dim, embed_dim))
        self.L_pool = nn.Parameter(torch.Tensor(pool_size, stats_dim, rank))
        self.R = nn.Parameter(torch.Tensor(rank, embed_dim))
        self.gate_logits = nn.Parameter(torch.Tensor(num_channels, pool_size))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_sh)
        nn.init.xavier_uniform_(self.L_pool)
        nn.init.xavier_uniform_(self.R)
        nn.init.zeros_(self.gate_logits)

    def forward(self, channel_stats):
        """
        Args:
            channel_stats: (B, K, stats_dim) computed from observable data
        Returns:
            embeddings: (B, K, embed_dim), entropy: scalar
        """
        g = F.softmax(
            self.gate_logits - self.gate_logits.max(dim=-1, keepdim=True).values,
            dim=-1,
        )  # (K, pool_size), numerically stable softmax
        entropy = -(g * torch.log(g + 1e-8)).sum(dim=-1).mean()

        shared = torch.matmul(channel_stats, self.W_sh)  # (B, K, embed_dim)

        L_combined = torch.einsum("nk,ksr->nsr", g, self.L_pool)  # (K, stats_dim, rank)
        W_aux = torch.matmul(L_combined, self.R)  # (K, stats_dim, embed_dim)
        auxiliary = torch.einsum(
            "bks,ksd->bkd", channel_stats, W_aux
        )  # (B, K, embed_dim)

        embeddings = shared + auxiliary
        return embeddings, entropy


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, seq_len: int, channels: int):
        super().__init__()
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1

        self.freq_weight = nn.Parameter(torch.ones(self.freq_len))

        self.proj = nn.Sequential(
            nn.Linear(self.freq_len, channels), nn.ReLU(), nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, L = x.shape

        x_fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(x_fft)

        weighted_amp = amplitude * self.freq_weight.unsqueeze(0).unsqueeze(0)

        freq_hidden = self.proj(weighted_amp)

        return freq_hidden


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x, observed_mask=None):
        if observed_mask is None:
            observed_mask = torch.ones_like(x)
        if observed_mask.shape != x.shape:
            raise ValueError("observed_mask must have the same shape as x")

        observed_mask = observed_mask.to(dtype=x.dtype)
        x_observed = x * observed_mask
        x_perm = x_observed.permute(0, 2, 1)
        mask_perm = observed_mask.permute(0, 2, 1)
        pad = (self.kernel_size - 1) // 2

        x_front = x_perm[:, 0:1, :].repeat(1, pad, 1)
        x_end = x_perm[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([x_front, x_perm, x_end], dim=1)

        mask_front = mask_perm[:, 0:1, :].repeat(1, pad, 1)
        mask_end = mask_perm[:, -1:, :].repeat(1, pad, 1)
        mask_padded = torch.cat([mask_front, mask_perm, mask_end], dim=1)

        moving_sum = self.moving_avg(x_padded.permute(0, 2, 1))
        moving_count = self.moving_avg(mask_padded.permute(0, 2, 1))
        moving_mean = moving_sum / moving_count.clamp_min(1e-8)

        observed_count = observed_mask.sum(dim=-1, keepdim=True)
        channel_mean = x_observed.sum(dim=-1, keepdim=True) / observed_count.clamp_min(1)
        moving_mean = torch.where(
            moving_count > 0,
            moving_mean,
            channel_mean.expand_as(moving_mean),
        )
        res = (x - moving_mean) * observed_mask
        return res, moving_mean


class HeterogeneityMaskGenerator(nn.Module):
    def __init__(
        self,
        channels: int,
        seq_len: int,
        het_mask_mode: str = "gumbel",
        top_k_ratio: float = 0.3,
        bias_strength: float = 5.0,
        het_gate_init: float = -2.0,
        neg_bias_strength: float = None,
    ):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.het_mask_mode = het_mask_mode
        self.top_k = max(2, int(channels * top_k_ratio))
        self.bias_strength = bias_strength
        self.neg_bias_strength = (
            neg_bias_strength if neg_bias_strength is not None else bias_strength
        )

        self.decomp = series_decomp(kernel_size=25)

        self.linears = nn.Linear(seq_len, 1, bias=False)
        self.lineart = nn.Linear(seq_len, 1, bias=False)

        self.weight_s = nn.Parameter(torch.tensor([0.45]))
        self.weight_t = nn.Parameter(torch.tensor([0.45]))

        if het_mask_mode == "topk_sparse":
            self.het_gate = nn.Parameter(torch.tensor([float(het_gate_init)]))

    @staticmethod
    def _masked_temporal_projection(
        differences: torch.Tensor,
        overlap: torch.Tensor,
        projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = projection(differences * overlap).squeeze(-1)
        absolute_weights = projection.weight.abs().view(1, 1, 1, -1)
        observed_weight = (absolute_weights * overlap).sum(dim=-1)
        full_weight = absolute_weights.sum(dim=-1)
        valid = overlap.sum(dim=-1) > 0
        scale = full_weight / observed_weight.clamp_min(1e-8)
        projected = torch.where(valid, projected * scale, torch.zeros_like(projected))
        return projected, valid

    def calculate_raw_distance(
        self,
        x: torch.Tensor,
        observed_mask: torch.Tensor,
        return_validity: bool = False,
    ):
        B, K, L = x.shape
        if observed_mask.shape != x.shape:
            raise ValueError("observed_mask must have the same shape as x")
        observed_mask = observed_mask.to(dtype=x.dtype)
        sea, trend = self.decomp(x, observed_mask)
        overlap = observed_mask.unsqueeze(2) * observed_mask.unsqueeze(1)

        sea_unsqueezed_i = sea.unsqueeze(2)
        sea_unsqueezed_j = sea.unsqueeze(1)
        diffs, valid_seasonal = self._masked_temporal_projection(
            torch.abs(sea_unsqueezed_i - sea_unsqueezed_j),
            overlap,
            self.linears,
        )

        trend_unsqueezed_i = trend.unsqueeze(2)
        trend_unsqueezed_j = trend.unsqueeze(1)
        difft, valid_trend = self._masked_temporal_projection(
            torch.abs(trend_unsqueezed_i - trend_unsqueezed_j),
            overlap,
            self.lineart,
        )

        raw_distance = self.weight_t * difft + self.weight_s * diffs
        valid_pairs = valid_seasonal & valid_trend
        if return_validity:
            return raw_distance, valid_pairs
        return raw_distance




    def _topk_sparse_bias(
        self, x: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        B, K, L = x.shape
        raw_dist, valid_pairs = self.calculate_raw_distance(
            x, observed_mask, return_validity=True
        )
        abs_dist = torch.abs(raw_dist)
        valid_float = valid_pairs.to(dtype=abs_dist.dtype)
        valid_count = valid_float.sum(dim=0)
        pair_valid = valid_count > 0
        dist_mean = (abs_dist * valid_float).sum(dim=0) / valid_count.clamp_min(1)

        k = min(self.top_k, K - 1)
        dist_masked = dist_mean.masked_fill(~pair_valid, float("inf"))
        dist_masked.fill_diagonal_(float("inf"))
        topk_values, topk_idx = dist_masked.topk(k, dim=-1, largest=False)

        topk_mask = torch.zeros(K, K, device=x.device)
        topk_mask.scatter_(1, topk_idx, torch.isfinite(topk_values).to(x.dtype))
        topk_mask.fill_diagonal_(1.0)

        off_diag = ~torch.eye(K, dtype=torch.bool, device=x.device)
        valid_off_diag = pair_valid & off_diag
        similarity = torch.zeros_like(dist_mean)
        if valid_off_diag.any():
            valid_dist = dist_mean[valid_off_diag]
            d_min = valid_dist.min()
            d_max = valid_dist.max()
            norm_dist = (dist_mean - d_min) / (d_max - d_min + 1e-6)
            similarity = torch.where(pair_valid, 1.0 - norm_dist, similarity)

        scale = self.bias_strength
        neg_scale = self.neg_bias_strength
        bias = torch.full((K, K), -neg_scale, device=x.device)
        bias[topk_mask.bool()] = scale * similarity[topk_mask.bool()]
        bias.fill_diagonal_(0.0)

        gate = torch.sigmoid(self.het_gate)
        return (gate * bias).unsqueeze(0).unsqueeze(0)

    def forward(
        self, x: torch.Tensor, observed_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._topk_sparse_bias(x, observed_mask)


class MTSCI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        self.use_lightsae_embed = config["model"].get("use_lightsae_embed", False)
        if self.use_lightsae_embed:
            self.lightsae_rank = config["model"].get("lightsae_rank", 4)
            self.lightsae_pool_size = config["model"].get("lightsae_pool_size", 3)
            self.lightsae_stats_dim = 3  # [obs_ratio, mean, std]
            self.lightsae_channel_embed = LightSAEDataEmbedding(
                num_channels=self.target_dim,
                stats_dim=self.lightsae_stats_dim,
                embed_dim=self.emb_feature_dim,
                rank=self.lightsae_rank,
                pool_size=self.lightsae_pool_size,
            )
            lightsae_mode = config["model"].get("lightsae_mode", "fixed")
            lightsae_alpha = config["model"].get("lightsae_alpha", 0.1)
            if lightsae_mode == "learnable":
                self.lightsae_alpha = nn.Parameter(torch.tensor(lightsae_alpha))
            else:
                self.lightsae_alpha = lightsae_alpha

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1
        self.diffmodel = denoising_network(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

        self.calc_intra_cons_loss = InfoNCE()

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def compute_channel_stats(self, X_Tilde, cond_mask):
        """Compute per-channel statistics from observable data.
        Args:
            X_Tilde: (B, K, L) input data
            cond_mask: (B, K, L) mask of observed values
        Returns:
            channel_stats: (B, K, 3) = [obs_ratio, mean, std]
        """
        X_obs = X_Tilde * cond_mask
        obs_count = cond_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        obs_ratio = cond_mask.mean(dim=-1, keepdim=True)
        channel_mean = X_obs.sum(dim=-1, keepdim=True) / obs_count
        diff_sq = (X_obs - channel_mean * cond_mask) ** 2
        channel_var = diff_sq.sum(dim=-1, keepdim=True) / obs_count
        channel_std = channel_var.sqrt().clamp(min=1e-8)
        return torch.cat([obs_ratio, channel_mean, channel_std], dim=-1)

    def get_side_info(self, observed_tp, cond_mask, X_Tilde=None) -> torch.Tensor:
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        if self.use_lightsae_embed and X_Tilde is not None:
            channel_stats = self.compute_channel_stats(X_Tilde, cond_mask)
            lightsae_embed, self._lightsae_entropy = self.lightsae_channel_embed(
                channel_stats
            )
            identity_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )
            alpha = self.lightsae_alpha
            if isinstance(alpha, nn.Parameter):
                alpha = torch.clamp(alpha, min=0.0)
            feature_embed = identity_embed.unsqueeze(0) + alpha * lightsae_embed
            feature_embed = feature_embed.unsqueeze(1).expand(-1, L, -1, -1)
        else:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
            self._lightsae_entropy = torch.tensor(0.0, device=self.device)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)

        return side_info



    def set_input_to_diffmodel(
        self,
        X_original,
        noisy_data,
        cond_mask,
        reverse_cond_mask,
    ):

        def get_noisy_total_input(X_original, X_noisy, mask) -> torch.Tensor:
            return (mask * X_original).unsqueeze(1) + ((1 - mask) * X_noisy).unsqueeze(
                1
            )

        total_input = get_noisy_total_input(X_original, noisy_data, cond_mask)
        reverse_total_input = get_noisy_total_input(
            X_original, noisy_data, reverse_cond_mask
        )
        return (
            total_input,
            reverse_total_input,
            X_original.unsqueeze(1),
            noisy_data.unsqueeze(1),
        )


    def forward(self, batch, is_train=1):
        if is_train:
            (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                X_pred,
                X_pred_mask,
            ) = self.process_data(batch, is_train)
        else:
            (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask) = (
                self.process_data(batch, is_train)
            )
            X_pred, X_pred_mask = None, None

        cond_mask = X_mask
        reverse_cond_mask = X_Tilde_mask * indicating_mask
        negative_mask = torch.zeros_like(X_Tilde_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, X_Tilde)
        reverse_side_info = self.get_side_info(observed_tp, reverse_cond_mask, X_Tilde)
        anchor_cond_info = self.get_side_info(observed_tp, X_Tilde_mask, X_Tilde)
        negative_cond_info = self.get_side_info(observed_tp, negative_mask, X_Tilde)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            X_Tilde=X_Tilde,
            cond_mask=cond_mask,
            X_Tilde_mask=X_Tilde_mask,
            indicating_mask=indicating_mask,
            X_pred=X_pred,
            side_info=side_info,
            reverse_side_info=reverse_side_info,
            observed_cond_info=anchor_cond_info,
            noisy_cond_info=negative_cond_info,
            is_train=is_train,
        )



class MTSCI(MTSCI_base):
    def __init__(self, config, device, target_dim=36, seq_len=24):
        super(MTSCI, self).__init__(target_dim, config, device)
        self.seq_len = seq_len

    def process_data(self, batch, istrain=1):
        if istrain == 1:
            (
                X_tensor,
                mask_tensor,
                indicating_mask_tensor,
                X_Tilde_tensor,
                X_Tilde_mask_tensor,
                pred_tensor,
                pred_mask_tensor,
            ) = batch
        else:
            (
                X_tensor,
                mask_tensor,
                X_Tilde_tensor,
                X_Tilde_mask_tensor,
                indicating_mask_tensor,
            ) = batch

        X_Tilde = X_Tilde_tensor.to(self.device).float()
        X_mask = mask_tensor.to(self.device).float()
        indicating_mask = indicating_mask_tensor.to(self.device).float()
        X_Tilde_mask = X_Tilde_mask_tensor.to(self.device).float()

        batch_size = X_Tilde.shape[0]
        observed_tp = (
            torch.from_numpy(
                np.tile(np.arange(self.seq_len), batch_size).reshape(
                    batch_size, self.seq_len
                )
            )
            .to(self.device)
            .float()
        )

        X_Tilde = X_Tilde.permute(0, 2, 1)
        X_Tilde_mask = X_Tilde_mask.permute(0, 2, 1)
        X_mask = X_mask.permute(0, 2, 1)
        indicating_mask = indicating_mask.permute(0, 2, 1)

        if istrain == 0:
            return (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask)

        else:
            pred = pred_tensor.to(self.device).float()
            pred_mask = pred_mask_tensor.to(self.device).float()
            pred = pred.permute(0, 2, 1)
            pred_mask = pred_mask.permute(0, 2, 1)

            return (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                pred,
                pred_mask,
            )


class MTSCI_Freq(MTSCI):
    def __init__(self, config, device, target_dim=36, seq_len=24):
        super(MTSCI_Freq, self).__init__(config, device, target_dim, seq_len)
        self.seq_len = seq_len
        if config["model"].get("het_mask_mode") != "topk_sparse":
            raise ValueError("het_mask_mode must be topk_sparse")

        channels = config["diffusion"]["channels"]
        self.freq_extractor = FrequencyFeatureExtractor(
            seq_len=seq_len, channels=channels
        )
        self.use_het_mask = config["model"].get("use_het_mask", True)
        # Control attention bias independently of affinity-based auxiliary losses.
        self.use_agca_bias = config["model"].get("use_agca_bias", True)
        self.het_mask_mode = config["model"].get("het_mask_mode", "gumbel")
        if self.use_het_mask:
            self.heterog_mask_gen = HeterogeneityMaskGenerator(
                channels=target_dim,
                seq_len=seq_len,
                het_mask_mode=self.het_mask_mode,
                top_k_ratio=config["model"].get("top_k_ratio", 0.3),
                bias_strength=config["model"].get("bias_strength", 5.0),
                het_gate_init=config["model"].get("het_gate_init", -2.0),
                neg_bias_strength=config["model"].get("neg_bias_strength", None),
            )
        self.eta = config["train"].get("eta", 0.0)

    def calc_loss_valid(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        X_pred,
        side_info,
        reverse_side_info,
        observed_cond_info,
        noisy_cond_info,
        is_train,
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            outputs = self.calc_loss(
                X_Tilde,
                cond_mask,
                X_Tilde_mask,
                indicating_mask,
                X_pred,
                side_info,
                reverse_side_info,
                observed_cond_info,
                noisy_cond_info,
                is_train=is_train,
                set_t=t,
            )
            loss_sum += outputs[0].detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        X_pred,
        side_info,
        reverse_side_info,
        observed_cond_info,
        noisy_cond_info,
        is_train=1,
        set_t=-1,
    ):
        B, K, L = X_Tilde.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(X_Tilde)
        noisy_data = (current_alpha**0.5) * X_Tilde + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        reverse_cond_mask = X_Tilde_mask * indicating_mask

        (
            total_input,
            reverse_total_input,
            all_observed_input,
            all_noisy_input,
        ) = self.set_input_to_diffmodel(
            X_original=X_Tilde,
            noisy_data=noisy_data,
            cond_mask=cond_mask,
            reverse_cond_mask=reverse_cond_mask,
        )

        if self.use_het_mask and self.use_agca_bias:
            heterog_mask = self.heterog_mask_gen(X_Tilde, cond_mask)
        else:
            heterog_mask = None

        (
            forward_pred_noise,
            forward_hidden,
            reverse_hidden,
            negative_hidden,
        ) = self.diffmodel(
            total_input,
            side_info,
            reverse_total_input,
            reverse_side_info,
            all_noisy_input,
            noisy_cond_info,
            X_pred,
            t,
            heterog_mask=heterog_mask,
        )

        target_mask = X_Tilde_mask - cond_mask
        residual = (noise - forward_pred_noise) * target_mask
        num_eval = target_mask.sum()
        loss_noise = (residual**2).sum() / (num_eval if num_eval > 0 else 1)

        predict_hidden = forward_hidden.reshape(B, -1, K, L) * (
            1 - cond_mask.unsqueeze(1)
        ) + reverse_hidden.reshape(B, -1, K, L) * (1 - reverse_cond_mask.unsqueeze(1))

        # L_cic: Channel Interpolation Consistency Loss
        # Uses TopkSparse similarity to weight-interpolate observed data
        # and measures interpolation error at missing positions.
        # Gradients flow only through similarity weights (not data).
        if self.use_het_mask and self.eta > 0:
            raw_dist, valid_pairs = self.heterog_mask_gen.calculate_raw_distance(
                X_Tilde, cond_mask, return_validity=True
            )
            similarity = torch.exp(-torch.abs(raw_dist)) * valid_pairs.to(
                dtype=X_Tilde.dtype
            )
            off_diag = 1 - torch.eye(K, device=X_Tilde.device).unsqueeze(0)
            weights = similarity * off_diag
            weight_sum = weights.sum(dim=-1, keepdim=True)
            weights = weights / weight_sum.clamp_min(1e-8)
            has_neighbor = weight_sum > 0

            X_obs = X_Tilde * cond_mask
            obs_count = cond_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            X_mean = X_obs.sum(dim=-1, keepdim=True) / obs_count
            X_centered = (X_Tilde - X_mean) * cond_mask
            X_var = (X_centered**2).sum(dim=-1, keepdim=True) / obs_count
            X_std = X_var.clamp(min=1e-8).sqrt()
            X_norm = X_centered / X_std

            X_interp_norm = torch.bmm(weights, X_norm)
            X_interp = X_interp_norm * X_std + X_mean

            cic_target_mask = target_mask * has_neighbor.to(dtype=target_mask.dtype)
            num_eval = cic_target_mask.sum()
            loss_cic = (((X_interp - X_Tilde.detach()) * cic_target_mask) ** 2).sum() / (
                num_eval if num_eval > 0 else 1
            )
        else:
            loss_cic = torch.tensor(0.0, device=self.device)

        # Sample-level contrastive loss.
        loss_cons_sample = self.calc_intra_cons_loss(
            negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1)
        )

        loss_cons = loss_cons_sample

        torch.cuda.empty_cache()

        return loss_noise, loss_cons, loss_cic, self._lightsae_entropy

    def evaluate(self, batch, n_samples):
        (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask) = (
            self.process_data(batch, istrain=0)
        )

        with torch.no_grad():
            cond_mask = X_mask
            eval_mask = X_Tilde_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, X_Tilde)

            if self.use_het_mask and self.use_agca_bias:
                attention_bias = self.heterog_mask_gen(X_Tilde, cond_mask)
            else:
                attention_bias = None

            samples = self._impute_with_attention_bias(
                X_Tilde, cond_mask, side_info, n_samples, attention_bias
            )

        return samples, X_Tilde, eval_mask, X_Tilde_mask, observed_tp

    def _impute_with_attention_bias(
        self, X_Tilde, cond_mask, side_info, n_samples, attention_bias=None
    ):
        B, K, L = X_Tilde.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(X_Tilde)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * X_Tilde).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = cond_obs + noisy_target

                predicted = self.diffmodel.impute(
                    diff_input,
                    side_info,
                    torch.tensor([t]).to(X_Tilde.device),
                    heterog_mask=attention_bias,
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1 - self.alpha[t - 1]) / (1 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples
