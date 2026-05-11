"""
Improved FullSubNet model for single-channel speech enhancement.

This implementation incorporates a refined finer-to-coarser processing strategy,
where the lower subband section undergoes more detailed processing, while the higher
subband section undergoes more generalized processing.

Reference: https://github.com/Audio-WestlakeU/FullSubNet

This version is adapted for project integration and follows the official
improved FullSubNet flow more closely:

1. waveform -> complex STFT
2. magnitude compression
3. remove last frequency bin before FB/SB processing
4. predict complex ratio mask (cRM)
5. pad mask back to full frequency dimension
6. apply mask to complex STFT
7. ISTFT -> enhanced waveform
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

EPSILON = np.finfo(np.float32).eps


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="LSTM",
        output_activate_function=None,
        dropout=0.0,
    ):
        super().__init__()

        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=False,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=False,
            )
        else:
            raise NotImplementedError(f"Unsupported sequence model: {sequence_model}")

        if int(output_size):
            hidden_dim = hidden_size * (2 if bidirectional else 1)
            self.fc_output_layer = nn.Linear(hidden_dim, output_size)
        else:
            self.fc_output_layer = None

        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Unsupported activation: {output_activate_function}"
                )
        else:
            self.activate_function = None

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, output_size, T]
        """
        assert x.dim() == 3, f"Expected [B, F, T], got {x.shape}"

        # [B, F, T] -> [T, B, F]
        x = x.permute(2, 0, 1).contiguous()
        o, _ = self.sequence_model(x)

        if self.fc_output_layer is not None:
            o = self.fc_output_layer(o)

        if self.activate_function is not None:
            o = self.activate_function(o)

        # [T, B, F] -> [B, F, T]
        return o.permute(1, 2, 0).contiguous()


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def offline_laplace_norm(input_tensor, return_mu=False):
        mu = torch.mean(
            input_tensor,
            dim=list(range(1, input_tensor.dim())),
            keepdim=True,
        )
        normed = input_tensor / (mu + EPSILON)
        if return_mu:
            return normed, mu
        return normed

    @staticmethod
    def cumulative_laplace_norm(input_tensor):
        """
        input_tensor: [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input_tensor.size()
        x = input_tensor.reshape(batch_size * num_channels, num_freqs, num_frames)

        # sum over frequency for each sample independently
        step_sum = torch.sum(x, dim=1)  # [B*C, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B*C, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, num_frames)

        entry_count = entry_count.expand_as(cumulative_sum)
        cumulative_mean = cumulative_sum / entry_count
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = x / (cumulative_mean + EPSILON)
        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input_tensor):
        mu = torch.mean(
            input_tensor,
            dim=list(range(1, input_tensor.dim())),
            keepdim=True,
        )
        std = torch.std(
            input_tensor,
            dim=list(range(1, input_tensor.dim())),
            keepdim=True,
        )
        return (input_tensor - mu) / (std + EPSILON)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            return self.offline_laplace_norm
        if norm_type == "cumulative_laplace_norm":
            return self.cumulative_laplace_norm
        if norm_type == "offline_gaussian_norm":
            return self.offline_gaussian_norm
        raise NotImplementedError(f"Unknown norm type: {norm_type}")


class SubBandSequenceWrapper(SequenceModel):
    """
    Processes each subband unit independently, then folds outputs back.
    """
    def __init__(
        self,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(
            dropout=dropout,
            **kwargs
        )

    def forward(self, subband_input):
        """
        Args:
            subband_input: [B, N, C, F_sub, T]
        Returns:
            [B, 2, N * center_freqs, T]
        """
        B, N, C, F_sub, T = subband_input.shape
        assert C == 1, "Only mono audio is supported."

        x = subband_input.reshape(B * N, F_sub, T)
        output = super().forward(x)  # [B*N, 2*center_freqs, T]

        output_size = output.shape[1]
        assert output_size % 2 == 0, "Expected output_size to be 2 * center_freqs"
        center_freqs = output_size // 2

        # [B*N, 2*center_freqs, T] -> [B, N, 2, center_freqs, T]
        output = output.reshape(B, N, 2, center_freqs, T)

        # [B, N, 2, center_freqs, T] -> [B, 2, N, center_freqs, T]
        output = output.permute(0, 2, 1, 3, 4).contiguous()

        # [B, 2, N, center_freqs, T] -> [B, 2, N*center_freqs, T]
        output = output.reshape(B, 2, N * center_freqs, T)
        return output


class SubbandModel(BaseModel):
    def __init__(
        self,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        sequence_model,
        hidden_size,
        activate_function=False,
        norm_type="offline_laplace_norm",
        dropout=0.0,
    ):
        super().__init__()

        sb_models = []
        for sb_cf, sb_nf, fb_cf, fb_nf in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
        ):
            sb_models.append(
                SubBandSequenceWrapper(
                    input_size=(sb_cf + sb_nf * 2) + (fb_cf + fb_nf * 2),
                    output_size=sb_cf * 2,
                    hidden_size=hidden_size,
                    num_layers=2,
                    sequence_model=sequence_model,
                    bidirectional=False,
                    output_activate_function=activate_function,
                    dropout=dropout
                )
            )

        self.sb_models = nn.ModuleList(sb_models)
        self.freq_cutoffs = freq_cutoffs
        self.sb_num_center_freqs = sb_num_center_freqs
        self.sb_num_neighbor_freqs = sb_num_neighbor_freqs
        self.fb_num_center_freqs = fb_num_center_freqs
        self.fb_num_neighbor_freqs = fb_num_neighbor_freqs
        self.norm = self.norm_wrapper(norm_type)

    def _freq_unfold(
        self,
        input_tensor,
        lower_cutoff_freq,
        upper_cutoff_freq,
        num_center_freqs,
        num_neighbor_freqs,
    ):
        """
        Args:
            input_tensor: [B, 1, F, T]
        Returns:
            [B, N, 1, F_sub, T]
        """
        B, C, num_freqs, T = input_tensor.shape
        assert C == 1, "Only mono audio is supported."

        freq_range = upper_cutoff_freq - lower_cutoff_freq
        if freq_range % num_center_freqs != 0:
            raise ValueError(
                f"Frequency range [{lower_cutoff_freq}, {upper_cutoff_freq}] "
                f"({freq_range} bins) must be divisible by num_center_freqs "
                f"({num_center_freqs})."
            )

        if lower_cutoff_freq == 0:
            valid_input = input_tensor[..., 0 : (upper_cutoff_freq + num_neighbor_freqs), :]
            valid_input = F.pad(valid_input, (0, 0, num_neighbor_freqs, 0), mode="reflect")
        elif upper_cutoff_freq == num_freqs:
            valid_input = input_tensor[..., lower_cutoff_freq - num_neighbor_freqs : num_freqs, :]
            valid_input = F.pad(valid_input, (0, 0, 0, num_neighbor_freqs), mode="reflect")
        else:
            valid_input = input_tensor[
                ...,
                lower_cutoff_freq - num_neighbor_freqs : upper_cutoff_freq + num_neighbor_freqs,
                :
            ]

        kernel_size = num_center_freqs + num_neighbor_freqs * 2

        unfolded = F.unfold(
            input=valid_input,
            kernel_size=(kernel_size, T),
            stride=(num_center_freqs, T),
        )  # [B, kernel_size, N]

        N = unfolded.shape[-1]

        output = unfolded.reshape(B, C, kernel_size, T, N)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_sub, T]
        return output

    def forward(self, noisy_input, fb_output):
        """
        Args:
            noisy_input: [B, 1, F, T]  where F is already trimmed (e.g. 256)
            fb_output:   [B, 1, F, T]
        Returns:
            [B, 2, F, T]
        """
        B, C, num_freqs, T = noisy_input.size()
        assert C == 1, "Only mono audio is supported."

        subband_outputs = []

        for idx, sb_model in enumerate(self.sb_models):
            if idx == 0:
                lower = 0
                upper = self.freq_cutoffs[0]
            elif idx == len(self.sb_models) - 1:
                lower = self.freq_cutoffs[-1]
                upper = num_freqs
            else:
                lower = self.freq_cutoffs[idx - 1]
                upper = self.freq_cutoffs[idx]

            noisy_sb = self._freq_unfold(
                noisy_input,
                lower,
                upper,
                self.sb_num_center_freqs[idx],
                self.sb_num_neighbor_freqs[idx],
            )

            fb_sb = self._freq_unfold(
                fb_output,
                lower,
                upper,
                self.fb_num_center_freqs[idx],
                self.fb_num_neighbor_freqs[idx],
            )

            sb_input = torch.cat([noisy_sb, fb_sb], dim=-2)  # concat on F_sub axis
            sb_input = self.norm(sb_input)

            sb_out = sb_model(sb_input)  # [B, 2, F_section, T]
            subband_outputs.append(sb_out)

        output = torch.cat(subband_outputs, dim=2)  # concat along frequency
        return output


class FullSubNet(BaseModel):
    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        fdrc=0.5,
        num_freqs=257,
        freq_cutoffs=(16, 96),
        sb_num_center_freqs=(1, 4, 8),
        sb_num_neighbor_freqs=(7, 7, 7),
        fb_num_center_freqs=(1, 4, 8),
        fb_num_neighbor_freqs=(7, 7, 7),
        fb_hidden_size=512,
        sb_hidden_size=384,
        sequence_model="LSTM",
        dropout=0.0,
        fb_output_activate_function=False,
        sb_output_activate_function=False,
        norm_type="offline_laplace_norm",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc
        self.num_freqs = num_freqs

        self.fb_model = SequenceModel(
            input_size=num_freqs - 1,
            output_size=num_freqs - 1,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            dropout=dropout
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=list(freq_cutoffs),
            sb_num_center_freqs=list(sb_num_center_freqs),
            sb_num_neighbor_freqs=list(sb_num_neighbor_freqs),
            fb_num_center_freqs=list(fb_num_center_freqs),
            fb_num_neighbor_freqs=list(fb_num_neighbor_freqs),
            hidden_size=sb_hidden_size,
            sequence_model=sequence_model,
            activate_function=sb_output_activate_function,
            norm_type=norm_type,
            dropout=dropout
        )

        self.norm = self.norm_wrapper(norm_type)

    def forward(self, y):
        """
        Args:
            y: [B, T] or [B, 1, T]
        Returns:
            enhanced_y: [B, 1, T]
        """
        ndim = y.dim()
        assert ndim in (2, 3), "Input must be [B, T] or [B, 1, T]"

        if ndim == 3:
            assert y.size(1) == 1, "Expected mono input of shape [B, 1, T]"
            y = y.squeeze(1)

        window = torch.hann_window(self.win_length, device=y.device)

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )  # [B, F, T]

        spec_real = torch.view_as_real(spec)  # [B, F, T, 2]
        mag = torch.abs(spec).unsqueeze(1)    # [B, 1, F, T]

        # Remove the last bin BEFORE both FB and SB processing
        mag_proc = mag ** self.fdrc
        mag_proc = mag_proc[..., :-1, :]      # [B, 1, F-1, T]

        # Fullband
        fb_in = rearrange(self.norm(mag_proc), "b c f t -> b (c f) t")
        fb_out = self.fb_model(fb_in)         # [B, F-1, T]
        fb_out = rearrange(fb_out, "b f t -> b 1 f t")  # [B, 1, F-1, T]

        # Subband
        crm = self.sb_model(mag_proc, fb_out)  # [B, 2, F-1, T]

        # Pad mask back to original frequency dimension
        crm = F.pad(crm, (0, 0, 0, 1), mode="constant", value=0.0)  # [B, 2, F, T]

        # Apply mask
        spec_real = rearrange(spec_real, "b f t c -> b c f t")  # [B, 2, F, T]
        enhanced_spec = crm * spec_real

        enhanced_complex = torch.complex(
            enhanced_spec[:, 0, ...],
            enhanced_spec[:, 1, ...],
        )

        enhanced_y = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=y.size(-1),
        )

        return enhanced_y.unsqueeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FullSubNet(
        n_fft=512,
        hop_length=128,
        win_length=512,
        num_freqs=257,
        freq_cutoffs=(16, 96),          # valid with trimmed F=256
        sb_num_center_freqs=(1, 4, 8),
        sb_num_neighbor_freqs=(7, 7, 7),
        fb_num_center_freqs=(1, 4, 8),
        fb_num_neighbor_freqs=(7, 7, 7),
    )

    x = torch.randn(2, 16000)

    with torch.no_grad():
        y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Trainable params:", f"{count_parameters(model):,}")