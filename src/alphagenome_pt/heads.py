# Provenance: Derived from AlphaGenome (Google LLC) Apache-2.0 code; translated to PyTorch and added MLM head. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
import abc
from collections.abc import Sequence
import dataclasses
import enum
import math

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# AlphaGenome
from .transformer import apply_rope
from .metadata import Metadata
from .schemas import Channels
from . import bundles
from . import schemas
from . import embeddings as embeddings_module
from . import losses


_SOFT_CLIP_VALUE = 10.0


class HeadType(enum.Enum):
    """Head types."""

    GENOME_TRACKS = 'genome_tracks'
    CONTACT_MAPS = 'contact_maps'
    SPLICE_SITES_CLASSIFICATION = 'splice_sites_classification'
    SPLICE_SITES_USAGE = 'splice_sites_usage'
    SPLICE_SITES_JUNCTION = 'splice_sites_junction'
    MASKED_LANGUAGE_MODELING = 'masked_language_modeling'


class HeadName(enum.Enum):
    """Output heads."""

    ATAC = 'atac'
    DNASE = 'dnase'
    PROCAP = 'procap'
    CAGE = 'cage'
    RNA_SEQ = 'rna_seq'
    CHIP_TF = 'chip_tf'
    CHIP_HISTONE = 'chip_histone'
    CONTACT_MAPS = 'contact_maps'
    SPLICE_SITES_CLASSIFICATION = 'splice_sites_classification'
    SPLICE_SITES_USAGE = 'splice_sites_usage'
    SPLICE_SITES_JUNCTION = 'splice_sites_junction'
    MASKED_LANGUAGE_MODELING = 'masked_language_modeling'

@dataclasses.dataclass
class HeadConfig:
    type: HeadType
    name: str
    num_organisms: int
    channels: Channels


@dataclasses.dataclass
class GenomeTracksHeadConfig(HeadConfig):
    max_seq_len: int
    num_tracks: int
    track_means: torch.Tensor   # [O, T]
    resolutions: Sequence[int]
    apply_squashing: bool
    bundle: str
    min_zero_multinomial_loss: bool = True


@dataclasses.dataclass
class ContactMapsHeadConfig(HeadConfig):
    num_tracks: int


@dataclasses.dataclass
class SpliceSitesClassificationHeadConfig(HeadConfig):
    num_tracks: int


@dataclasses.dataclass
class SpliceSitesUsageHeadConfig(HeadConfig):
    num_tracks: int


@dataclasses.dataclass
class SpliceSitesJunctionHeadConfig(HeadConfig):
    max_seq_len: int
    splice_site_channels: int
    num_tissues: int


@dataclasses.dataclass
class MaskedLanguageModelingHeadConfig(HeadConfig):
    out_vocab_size: int = 5


def create_heads(
    max_seq_len: int,
    channels: Channels,
    splice_site_channels: int,
    min_zero_multinomial_loss: bool,
    metadata: Metadata,
):
    heads = nn.ModuleDict()
    for head_name in HeadName:
        if head_name.value not in metadata.get_heads():
            continue
        head_config = get_head_config(
            head_name=head_name,
            max_seq_len=max_seq_len,
            channels=channels,
            splice_site_channels=splice_site_channels,
            min_zero_multinomial_loss=min_zero_multinomial_loss,
            metadata=metadata,
        )
        heads[head_name.value] = create_head(head_config)
    return heads


def create_head(
    config: HeadConfig,
) -> 'Head':
    match config.type:
        case HeadType.GENOME_TRACKS:
            assert isinstance(config, GenomeTracksHeadConfig)
            return GenomeTracksHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                max_seq_len=config.max_seq_len,
                num_tracks=config.num_tracks,
                track_means=config.track_means,
                resolutions=config.resolutions,
                apply_squashing=config.apply_squashing,
                bundle=config.bundle,
                min_zero_multinomial_loss=config.min_zero_multinomial_loss,
            )
        case HeadType.CONTACT_MAPS:
            return ContactMapsHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                num_tracks=config.num_tracks,
            )
        case HeadType.SPLICE_SITES_CLASSIFICATION:
            return SpliceSitesClassificationHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                num_tracks=config.num_tracks,
            )
        case HeadType.SPLICE_SITES_USAGE:
            return SpliceSitesUsageHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                num_tracks=config.num_tracks,
            )
        case HeadType.SPLICE_SITES_JUNCTION:
            return SpliceSitesJunctionHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                max_seq_len=config.max_seq_len,
                splice_site_channels=config.splice_site_channels,
                num_tissues=config.num_tissues,
            )
        case HeadType.MASKED_LANGUAGE_MODELING:
            return MaskedLanguageModelingHead(
                name=config.name,
                num_organisms=config.num_organisms,
                channels=config.channels,
                out_vocab_size=config.out_vocab_size,
            )
        case _:
            raise ValueError(f'Unknown head type: {config.type}')


def get_head_config(
        head_name: HeadName,
        max_seq_len: int,
        channels: Channels,
        splice_site_channels: int,
        min_zero_multinomial_loss: bool,
        metadata: Metadata,
    ) -> HeadConfig:
    """Returns a head for the given head name."""
    num_organisms = metadata.get_num_organisms()
    match head_name:
        case HeadName.ATAC:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]s
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.ATAC.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[1, 128],
                apply_squashing=False,
                bundle=bundles.BundleName.ATAC,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.DNASE:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.DNASE.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[1, 128],
                apply_squashing=False,
                bundle=bundles.BundleName.DNASE,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.PROCAP:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.PROCAP.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[1, 128],
                apply_squashing=False,
                bundle=bundles.BundleName.PROCAP,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.CAGE:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.CAGE.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[1, 128],
                apply_squashing=False,
                bundle=bundles.BundleName.CAGE,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.RNA_SEQ:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.RNA_SEQ.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[1, 128],
                apply_squashing=True,
                bundle=bundles.BundleName.RNA_SEQ,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.CHIP_TF:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.CHIP_TF.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[128],
                apply_squashing=False,
                bundle=bundles.BundleName.CHIP_TF,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.CHIP_HISTONE:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            track_means = metadata.get_means(head_name.value)               # [O, T]
            return GenomeTracksHeadConfig(
                type=HeadType.GENOME_TRACKS,
                name=HeadName.CHIP_HISTONE.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                num_tracks=num_tracks,
                track_means=track_means,
                resolutions=[128],
                apply_squashing=False,
                bundle=bundles.BundleName.CHIP_HISTONE,
                min_zero_multinomial_loss=min_zero_multinomial_loss,
            )
        case HeadName.CONTACT_MAPS:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            return ContactMapsHeadConfig(
                type=HeadType.CONTACT_MAPS,
                name=HeadName.CONTACT_MAPS.value,
                num_organisms=num_organisms,
                channels=channels,
                num_tracks=num_tracks,
            )
        case HeadName.SPLICE_SITES_CLASSIFICATION:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            return SpliceSitesClassificationHeadConfig(
                type=HeadType.SPLICE_SITES_CLASSIFICATION,
                name=HeadName.SPLICE_SITES_CLASSIFICATION.value,
                num_organisms=num_organisms,
                channels=channels,
                num_tracks=num_tracks,
            )
        case HeadName.SPLICE_SITES_USAGE:
            num_tracks = metadata.get_num_tracks(head_name.value)           # [1] (padded to be same across organisms)
            return SpliceSitesUsageHeadConfig(
                type=HeadType.SPLICE_SITES_USAGE,
                name=HeadName.SPLICE_SITES_USAGE.value,
                num_organisms=num_organisms,
                channels=channels,
                num_tracks=num_tracks,
            )
        case HeadName.SPLICE_SITES_JUNCTION:
            num_tissues = metadata.get_num_tissues(head_name.value)         # [1] (padded to be same across organisms)
            return SpliceSitesJunctionHeadConfig(
                type=HeadType.SPLICE_SITES_JUNCTION,
                name=HeadName.SPLICE_SITES_JUNCTION.value,
                num_organisms=num_organisms,
                channels=channels,
                max_seq_len=max_seq_len,
                splice_site_channels=splice_site_channels,
                num_tissues=num_tissues,
            )
        case HeadName.MASKED_LANGUAGE_MODELING:
            return MaskedLanguageModelingHeadConfig(
                type=HeadType.MASKED_LANGUAGE_MODELING,
                name=HeadName.MASKED_LANGUAGE_MODELING.value,
                num_organisms=num_organisms,
                channels=channels,
            )
        case _:
            raise ValueError(f'Unknown head name: {head_name}')


def _sum_pool(
    x: torch.Tensor,                        # [B, S, C]
    width: int,                             # W
) -> torch.Tensor:
    """Sum pooling over the sequence dimension."""
    B, S, C = x.shape                       # [B, S, C]
    dtype = torch.float32
    x = x.view(B, S // width, width, C)     # [B, S//W, W, C]
    return x.sum(dim=2, dtype=dtype)        # [B, S//W, C]


def get_param_for_index(
    params: torch.Tensor,                    # [O, ...]
    index: torch.Tensor,                     # [B]
) -> torch.Tensor:
    """
    Returns a parameter for a specific index. This is necessary for
    for multiorganism linear models where each organism owns a slice
    of the parameter tensor.

    Example:
        - params: [3, 4, 5] (3 organisms, 4 in_features, 5 out_features)
        - index: [2, 0, 1] (batch size 3)
        - returns: [3, 4, 5] (batch size 3, 4 in_features, 5 out_features)
    """
    return params[index]


class MultiOrganismLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_organisms: int,
    ):
        super().__init__()
        # Read
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_organisms = num_organisms

        # Parameters
        # NOTE: parameters are NOT shared across organisms
        w_shape = (num_organisms, in_channels, out_channels)
        b_shape = (num_organisms, out_channels)
        self.weight = nn.Parameter(torch.Tensor(*w_shape))
        self.bias = nn.Parameter(torch.Tensor(*b_shape))
        self.reset_parameters()

    def reset_parameters(self):
        # Match Haiku VarianceScaling(fan_in, truncated_normal)
        fan_in = self.in_channels
        target_std = math.sqrt(1.0 / fan_in)
        trunc_std_correction = 0.8796256610342398  # std of N(0,1) truncated to [-2, 2]
        raw_std = target_std / trunc_std_correction
        nn.init.trunc_normal_(self.weight, mean=0.0, std=raw_std, a=-2 * raw_std, b=2 * raw_std)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,                    # [B, *, D_in]
        organism_index: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        w = get_param_for_index(self.weight, organism_index).to(x.dtype)                            # [B, D_in, D_out]
        b = get_param_for_index(self.bias, organism_index).to(x.dtype)                              # [B, D_out]
        num_inner_dims = len(x.shape) - 2
        target_b_shape = (b.shape[0],) + (1,) * num_inner_dims + (b.shape[1],)
        x = torch.einsum('b...i, bij -> b...j', x, w.to(x.dtype)) + b.reshape(target_b_shape)       # [B, *, D_out]
        return x


def predictions_scaling(
    x: torch.Tensor,                        # [B, S, C]
    track_means: torch.Tensor,              # [B, C]
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
) -> torch.Tensor:
    """Scales predictions to experimental data scale."""
    x = torch.where(
        x > soft_clip_value,
        (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
        x,
    )                                                           # [B, S, C]
    if apply_squashing:
        x = x.pow(1.0 / 0.75)                                   # [B, S, C]
    x = x * (track_means[:, None] * resolution).to(x.dtype)     # [B, S, C]
    return x                                                    # [B, S, C]


def targets_scaling(
    x: torch.Tensor,                        # [B, S, C]
    track_means: torch.Tensor,              # [B, C]
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
) -> torch.Tensor:
    """Scales experimental targets to the model prediction space."""
    x = x / (track_means[:, None] * resolution).to(x.dtype)     # [B, S, C]

    if apply_squashing:
        x = x.pow(0.75)                                         # [B, S, C]

    x = torch.where(
        x > soft_clip_value,
        2.0 * torch.sqrt(x * soft_clip_value) - soft_clip_value,
        x,
    )                                                           # [B, S, C]
    return x


class Head(nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for a model head."""

    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
    ):
        """Initializes the Head class."""
        super().__init__()
        self._name = name
        self._num_organisms = num_organisms
        self._channels = channels
        if self._num_organisms == 0:
            raise ValueError('No metadata provided for any organism.')

    @property
    def name(self) -> str:
        return self._name
    
    @abc.abstractmethod
    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        **kwargs,
    ):
        """Returns the predictions for the head."""

    @abc.abstractmethod
    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ):
        """Returns the loss for the head."""


class GenomeTracksHead(Head):
    """A model head that predicts genome tracks at multiple resolutions.

    This module takes sequence embeddings at different resolutions and produces
    predictions for a specified number of tracks. It uses organism-specific
    linear layers and learnt scales to generate the predictions.
    """

    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        max_seq_len: int,
        num_tracks: int,
        track_means: torch.Tensor,          # [O, T]
        resolutions: Sequence[int],
        apply_squashing: bool,
        bundle: bundles.BundleName | None = None,
        min_zero_multinomial_loss: bool = True,
    ):
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._max_seq_len = max_seq_len
        self._num_tracks = num_tracks
        if isinstance(track_means, torch.Tensor):
            _track_means = track_means.detach().clone()
        else:
            _track_means = torch.as_tensor(track_means)
        self.register_buffer("_track_means", _track_means.to(torch.float32))    # [O, T]
        self._resolutions = sorted(resolutions)
        self._apply_squashing = apply_squashing
        self._bundle = bundle
        self._min_zero_multinomial_loss = min_zero_multinomial_loss
        self.multiorg_linear = nn.ModuleDict({
            str(r): MultiOrganismLinear(
                in_channels=channels.get_num_channels(r),       # dict: {1: C1, 128: C128, ...}
                out_channels=num_tracks,
                num_organisms=num_organisms,
            )
            for r in self._resolutions
        })
        self.residual_scales = nn.ParameterDict({
            str(r): nn.Parameter(
                torch.ones((self._num_organisms, self._num_tracks))
            )
            for r in self._resolutions
        })

    def unscale(
        self,
        x: torch.Tensor,                    # [B, S, C]
        organism_index: torch.Tensor,       # [B]
        resolution: int,
    ) -> torch.Tensor:
        """Unscales predictions to experimental data scale."""
        track_means = get_param_for_index(self._track_means, organism_index)
        return predictions_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self._apply_squashing,
        )
    
    def scale(
        self,
        x: torch.Tensor,                    # [B, S, C]
        organism_index: torch.Tensor,       # [B]
        resolution: int,
    ) -> torch.Tensor:
        """Scales targets to model predictions scale."""
        track_means = get_param_for_index(self._track_means, organism_index)
        return targets_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self._apply_squashing,
        )
    
    def _forward(
        self,
        embeddings: embeddings_module.Embeddings,   # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,               # [B]
        resolution: int,
    ) -> torch.Tensor:
        """Predicts genome tracks."""
        x = embeddings.get_sequence_embeddings(resolution)              # [B, S, C]
        x = self.multiorg_linear[str(resolution)](x, organism_index)    # [B, S, T]
        residual_scale = get_param_for_index(self.residual_scales[str(resolution)], organism_index)
        return F.softplus(x) * F.softplus(residual_scale[:, None, :])
    
    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B] 
        **kwargs
    ):
        predictions = {}
        for resolution in self._resolutions:
            scaled_predictions = self._forward(
                embeddings, organism_index, resolution
            )
            predictions[f'scaled_predictions_{resolution}bp'] = scaled_predictions
            predictions[f'predictions_{resolution}bp'] = self.unscale(
                scaled_predictions, organism_index, resolution
            )
        return predictions
    
    def _compute_loss(
        self,
        *,
        organism_index: torch.Tensor,       # [B]
        predictions: torch.Tensor,          # [B, S, C]
        targets: torch.Tensor,              # [B, S, C]
        targets_mask: torch.Tensor | None,  # [B, #S, C] | None
        resolution: int,
    ) -> dict[str, torch.Tensor]:
        """Computes the loss for the head at a given resolution."""
        assert predictions.shape == targets.shape, \
            f'Predictions shape {predictions.shape} does not match targets shape {targets.shape}.'
        scaled_targets = self.scale(targets, organism_index, resolution)
        all_losses = losses.multinomial_loss(
            y_pred=predictions,
            y_true=scaled_targets,
            mask=targets_mask,
            positional_weight=5.0,
            multinomial_resolution=predictions.shape[-2],
            min_zero=self._min_zero_multinomial_loss,
        )
        return all_losses
    
    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: schemas.DataBatch,
    ) -> dict[str, torch.Tensor]:
        """Returns the loss for the head."""
        if self._bundle is None:
            raise ValueError('Bundle is required for loss computation.')
        
        tracks, mask = batch.get_genome_tracks(self._bundle)
        
        if mask.shape[-2] != 1:
            raise ValueError(
                'We assume the mask to broadcast over the sequence length.'
            )
        
        bundle_resolution = self._bundle.get_resolution()
        loss_sum, scalars = 0.0, {}
        
        for resolution in self._resolutions:
            predictions_for_resolution = predictions[
                f'scaled_predictions_{resolution}bp'
            ]
            if resolution == bundle_resolution:
                targets = tracks
            else:
                targets = _sum_pool(tracks, resolution)
            
            all_losses = self._compute_loss(
                organism_index=batch.get_organism_index(),
                predictions=predictions_for_resolution,
                targets=targets,
                targets_mask=mask,
                resolution=resolution,
            )
            for k, v in all_losses.items():
                scalars[f'{k}_{resolution}bp'] = v
            loss_sum += all_losses['loss']
        
        scalars['loss'] = loss_sum
        return scalars


class ContactMapsHead(Head):
    """A model head that predicts contact maps from pairwise embeddings."""
    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        num_tracks: int,
    ):
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._num_tracks = num_tracks
        self.multiorg_linear = MultiOrganismLinear(
            in_channels=channels.channels_pair,
            out_channels=num_tracks,
            num_organisms=num_organisms,
        )

    def _forward(
        self,
        pair_embeddings: torch.Tensor,      # [B, S, S, C]
        organism_index: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """Predicts contact maps from pairwise embeddings."""
        return self.multiorg_linear(pair_embeddings, organism_index)
   
    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predicts contact maps from embeddings."""
        return {
            'predictions': self._forward(
                embeddings.embeddings_pair, organism_index
            )
        }
    
    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: schemas.DataBatch,
    ) -> dict[str, torch.Tensor]:
        """Returns the loss for the head."""
        if (targets := batch.contact_maps) is None:
            raise ValueError('contact_maps target not in batch.')
        
        B, S, _, T = targets.shape
        device = batch.get_organism_index().device

        if (targets_mask := batch.contact_maps_mask) is None:
            targets_mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=device)

        contact_predictions = predictions['predictions']
        assert contact_predictions.shape == targets.shape, \
            'Predictions shape does not match targets shape.'
        
        # Mask out NaN targets (which happens when balancing a missing slice).
        targets_mask = torch.where(
            torch.isnan(targets), False, targets_mask
        )
        targets = torch.where(torch.isnan(targets), 0.0, targets)
        loss = losses.mse(contact_predictions, targets, targets_mask)
        return {'loss': loss}


class SpliceSitesClassificationHead(Head):
    """A model head that predicts splice site classification."""
    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        num_tracks: int,
    ):
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._num_tracks = num_tracks
        self.multiorg_linear = MultiOrganismLinear(
            in_channels=channels.get_num_channels(1),
            out_channels=num_tracks,
            num_organisms=num_organisms,
        )

    def _forward_logits(
        self,
        x: torch.Tensor,                    # [B, S, C]
        organism_index: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """Splice site classification."""
        return self.multiorg_linear(x, organism_index)

    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predicts splice site classification from embeddings."""
        embeddings_1bp = embeddings.get_sequence_embeddings(1)
        logits = self._forward_logits(embeddings_1bp, organism_index)
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        return {'logits': logits, 'predictions': probs}

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: schemas.DataBatch,
        ) -> dict[str, torch.Tensor]:
            """Returns the loss for the head."""
            if (splice_sites := batch.splice_sites) is None:
                raise ValueError('splice_sites target not in batch.')
            logits = predictions['logits']
            assert splice_sites.shape == logits.shape, \
                    'Predictions shape does not match targets shape.'
            
            classification_mask = torch.any(splice_sites.bool(), dim=-1, keepdim=True)
            loss = losses.cross_entropy_loss_from_logits(
                y_pred_logits=logits,
                # Label smoothing with FP32 machine precision (~1e-7) for 5 classes.
                y_true=(1.0 - 1e-7) * splice_sites.to(torch.float32)
                + 1e-7 / self._num_tracks,
                mask=classification_mask,
                axis=-1,
            )
            return {'loss': loss}


class SpliceSitesUsageHead(Head):
    """A model head that predicts splice site usage."""

    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        num_tracks: int,
    ):
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._num_tracks = num_tracks
        self.multiorg_linear = MultiOrganismLinear(
            in_channels=channels.get_num_channels(1),
            out_channels=num_tracks,
            num_organisms=num_organisms,
        )

    def _forward_logits(
        self,
        x: torch.Tensor,                    # [B, S, C]
        organism_index: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """Splice site usage."""
        return self.multiorg_linear(x, organism_index)
    
    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predicts splice site usage from embeddings."""
        embeddings_1bp = embeddings.get_sequence_embeddings(1)              # [B, S, C]
        logits = self._forward_logits(embeddings_1bp, organism_index)       # [B, S, T]
        splice_site_usage = torch.sigmoid(logits.to(torch.float32)).to(torch.float16)
        return {'logits': logits, 'predictions': splice_site_usage}

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: schemas.DataBatch,
    ) -> dict[str, torch.Tensor]:
        """Returns the loss for the head."""
        if (splice_site_usage := batch.splice_site_usage) is None:
            raise ValueError('splice_site_usage target not in batch.')
        logits = predictions['logits']
        assert splice_site_usage.shape == logits.shape, \
            'Predictions shape does not match targets shape.'
        
        mask = batch.splice_site_usage_mask
        loss = losses.binary_crossentropy_from_logits(
            y_pred=logits,
            y_true=torch.clamp(splice_site_usage.to(torch.float32), 1e-7, 1.0 - 1e-7),
            mask=mask,
        )
        return {'loss': loss}


class SpliceSitesJunctionHead(Head):
    """A model head that predicts splice site junctions."""
    
    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        max_seq_len: int,
        splice_site_channels: int,
        num_tissues: int,
    ):
        """Initializes the SpliceSitesJunctionHead module."""
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._num_tissues = num_tissues
        self._num_tracks = 2 * self._num_tissues
        self._max_position_encoding_distance = max_seq_len
        self.in_channels = channels.get_num_channels(1)
        self._splice_site_channels = splice_site_channels
        self.multiorg_linear = MultiOrganismLinear(
            in_channels=self.in_channels,
            out_channels=self._splice_site_channels,
            num_organisms=num_organisms,
        )
        shape = (self._num_organisms, 2, self._num_tissues, self._splice_site_channels)
        self.pos_acceptor_logits_embeddings = nn.Parameter(torch.zeros(shape))
        self.pos_donor_logits_embeddings = nn.Parameter(torch.zeros(shape))
        self.neg_acceptor_logits_embeddings = nn.Parameter(torch.zeros(shape))
        self.neg_donor_logits_embeddings = nn.Parameter(torch.zeros(shape))
    
    def _get_track_mask(self, tissue_mask) -> torch.Tensor:                     # tissue_mask: [B, #A, #D, T]
        return torch.cat([tissue_mask, tissue_mask], dim=-1).to(torch.bool)     # [B, #A, #D, 2*T]
    
    def _forward(
        self,
        x: torch.Tensor,                        # [B, S, C]
        splice_site_positions: torch.Tensor,    # [B, 4, P]
        organism_index: torch.Tensor,           # [B]
        tissue_mask: torch.Tensor,              # [B, #A, #D, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:     # both [B, D, A, 2*T]
        """Splice site junctions."""
        assert splice_site_positions.shape[1] == 4, \
            'splice_site_positions must have shape [B, 4, P] for 4 DNA base pairs.'
        pos_donor_idx = splice_site_positions[:, 0, :]      # [B, D]
        pos_accept_idx = splice_site_positions[:, 1, :]     # [B, A]
        neg_donor_idx = splice_site_positions[:, 2, :]      # [B, D]
        neg_accept_idx = splice_site_positions[:, 3, :]     # [B, A]

        def _index_embedding(embedding, indices):
            # embedding: [B, S, C], indices: [B, P] (may contain -1)
            B, S, C = embedding.shape
            safe = indices.clamp(min=0)                         # gather can’t take -1
            idx = safe.unsqueeze(-1).expand(-1, -1, C).long()   # [B, P, C]
            return embedding.gather(dim=1, index=idx)           # [B, P, C]

        def _apply_rope(x, indices, name: str):                                 # x: [B, S, C_splice] | indices: [B, P]
            x = _index_embedding(x, indices).to(torch.float32)                  # [B, P, C_splice]
            params = get_param_for_index(
                getattr(self, f"{name}_embeddings"), organism_index
            )                                                                   # [B, 2, T, C_splice]
            # scale and offset have shape [B, 1, T, C_splice]
            scale, offset = params[:, [0], :, :], params[:, [1], :, :]          # [B, 1, T, C_splice]
            x = scale * x[:, :, None, :] + offset                               # [B, P, T, C_splice]
            x = apply_rope(
                x, indices, max_position=self._max_position_encoding_distance   # [B, P, T, C_splice]
            )
            return x
        
        splice_site_logits = self.multiorg_linear(x, organism_index)            # [B, S, C_splice]

        pos_accept_logits = _apply_rope(splice_site_logits, pos_accept_idx, "pos_acceptor_logits")      # [B, A, T, C_splice]
        pos_donor_logits = _apply_rope(splice_site_logits, pos_donor_idx, "pos_donor_logits")           # [B, D, T, C_splice]
        neg_accept_logits = _apply_rope(splice_site_logits, neg_accept_idx, "neg_acceptor_logits")      # [B, A, T, C_splice]
        neg_donor_logits = _apply_rope(splice_site_logits, neg_donor_idx, "neg_donor_logits")           # [B, D, T, C_splice]

        pos_counts = F.softplus(
            torch.einsum(
                'bdtc, batc -> bdat',
                pos_donor_logits,
                pos_accept_logits,
            )
        )   # [B, D, T, C_splice] x [B, A, T, C_splice] -> [B, D, A, T]
        neg_counts = F.softplus(
            torch.einsum(
                'bdtc, batc -> bdat',
                neg_donor_logits,
                neg_accept_logits,
            )
        )   # [B, D, T, C_splice] x [B, A, T, C_splice] -> [B, D, A, T]

        # NOTE: The einsum for these masks is equivalent to an "and" operation
        pos_mask = torch.einsum(
            'bd,ba->bda', pos_donor_idx >= 0, pos_accept_idx >= 0
        ).to(torch.float32)                                                     # [B, D, A]
        neg_mask = torch.einsum(
            'bd,ba->bda', neg_donor_idx >= 0, neg_accept_idx >= 0
        ).to(torch.float32)                                                     # [B, D, A]
        track_mask = self._get_track_mask(tissue_mask)                          # [B, #A, #D, 2*T]
        pos_mask = (
            pos_mask[:, :, :, None]
            * track_mask[:, :, :, :self._num_tissues]
        )                                                                       # [B, D, A, 2*T]
        neg_mask = (
            neg_mask[:, :, :, None]
            * track_mask[:, :, :, self._num_tissues:]
        )                                                                       # [B, D, A, 2*T]
        
        splice_site_junction_mask = torch.cat([pos_mask, neg_mask], dim=-1)     # [B, D, A, 2*T]
        pred_counts = torch.cat([pos_counts, neg_counts], dim=-1)               # [B, D, A, 2*T]
        pred_counts = torch.where(
            splice_site_junction_mask.bool(), pred_counts, 0
        )
        return pred_counts, splice_site_junction_mask
    
    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        tissue_mask: torch.Tensor,                      # [B, #A, #D, T]
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predicts splice site junctions from embeddings."""
        if (splice_site_positions := kwargs.get('splice_site_positions')) is None:
            raise ValueError(
                'splice_site_positions is required for junctions predictions.'
            )
        embeddings_1bp = embeddings.get_sequence_embeddings(1)          # [B, S, C]
        splice_site_junction, splice_junction_mask = self._forward(     # both [B, D, A, 2*T]
           embeddings_1bp, splice_site_positions, 
           organism_index, tissue_mask
        )
        return {
           'predictions': splice_site_junction,                         # [B, D, A, 2*T]
           'splice_site_positions': splice_site_positions,              # [B, 4, P]
           'splice_junction_mask': splice_junction_mask                 # [B, D, A, 2*T]
        }
    
    def loss(
        self,
        predictions: torch.Tensor,              # [B, ...]
        batch: schemas.DataBatch,
    ) -> torch.Tensor:
        """Returns the loss for the head."""
        if (count_target := batch.splice_site_junction) is None:
            raise ValueError('splice_site_junction target not in batch.')

        pred_pair = predictions['predictions']
        pairs_mask = predictions['splice_junction_mask']
        mask = self._get_track_mask(batch.splice_site_junction_mask)
        pairs_mask = pairs_mask * mask
        # Junctions shape is [B, D, A, 2*T]

        def _scale_junction_counts(counts):     # [B, D, A, 2*T]
            return torch.where(
                counts > _SOFT_CLIP_VALUE,
                2.0 * torch.sqrt(counts * _SOFT_CLIP_VALUE) - _SOFT_CLIP_VALUE,
                counts,
            )
        
        pairs_mask = pairs_mask.to(torch.bool)
        accept_total_loss = losses.poisson_loss(
            y_true=_scale_junction_counts(
               (count_target.masked_fill(~pairs_mask, 0.0)).sum(dim=-2, dtype=torch.float32)
            ),
            y_pred=(pred_pair.masked_fill(~pairs_mask, 0.0)).sum(dim=-2, dtype=torch.float32),
            mask=(pairs_mask.any(dim=-2)),
        )
        donor_total_loss = losses.poisson_loss(
            y_true=_scale_junction_counts(
               (count_target.masked_fill(~pairs_mask, 0.0)).sum(dim=-3, dtype=torch.float32)
            ),
            y_pred=(pred_pair.masked_fill(~pairs_mask, 0.0)).sum(dim=-3, dtype=torch.float32),
            mask=(pairs_mask.any(dim=-3)),
        )

        # Ratios with cross entropy loss
        donor_ratios_loss = losses.cross_entropy_loss(
            y_true=count_target,
            y_pred=pred_pair,
            mask=pairs_mask,
            axis=-3,
        )
        acceptor_ratios_loss = losses.cross_entropy_loss(
            y_true=count_target,
            y_pred=pred_pair,
            mask=pairs_mask,
            axis=-2,
        )
        loss = (
           donor_ratios_loss
           + acceptor_ratios_loss
           + 0.2 *(accept_total_loss + donor_total_loss)
        )
        return {'loss': loss}


class MaskedLanguageModelingHead(Head):
    """
    A model head that predicts masked language modeling (MLM) for DNA sequences.
    
    NOTE: It would make sense to make the out_vocab_size=4 so that the model doesn't
          predict (N) ambiguous bases. However, in the scenario that the user allows 
          (N) bases to be masked but the model doesn't allow them for prediction, that
          would likely be destructive to training. In the scenario that the user doesn't
          allow (N) bases to be masked but the model allows them for prediction, the 
          model can just learn to not predict (N). So, we include (N) in the output 
          vocabulary for safety, even if it may not be used in practice.

          Changing the out_vocab_size is currently not a supported feature.
    """
    def __init__(
        self,
        *,
        name: str,
        num_organisms: int,
        channels: Channels,
        out_vocab_size: int = 5,  # 4 DNA bases (A,T,C,G) + ambiguous (N)
    ):
        super().__init__(
            name=name,
            num_organisms=num_organisms,
            channels=channels,
        )
        self._channels = channels.get_num_channels(1)
        self.linear = nn.Linear(self._channels, out_vocab_size)

    def forward(
        self,
        embeddings: embeddings_module.Embeddings,       # (1bp, 128bp, 2048pair)
        organism_index: torch.Tensor,                   # [B]
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predicts masked language modeling from embeddings."""
        embeddings_1bp = embeddings.get_sequence_embeddings(1)          # [B, S, C]
        logits = self.linear(embeddings_1bp)                            # [B, S, 5]
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        return {'logits': logits, 'predictions': probs}

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: schemas.DataBatch,
    ) -> dict[str, torch.Tensor]:
        """Returns the loss for the head."""
        if (labels := batch.mlm) is None:
            raise ValueError('masked_lm_labels target not in batch.')
        
        logits = predictions['logits']
        assert labels.shape == logits.shape[:-1], \
            'Predictions shape does not match targets shape.'
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')
        return {'loss': loss}
    
