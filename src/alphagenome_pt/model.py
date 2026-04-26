# Provenance: Derived from AlphaGenome (Google LLC) Apache-2.0 code; partially reimplemented from bioRxiv paper, partially from code, and translated to PyTorch. Rylie Weaver, 2026.
# SPDX-License-Identifier: Apache-2.0

"""Imports"""
# General
import json
from typing import Union
from pathlib import Path

# Torch
import torch
import torch.nn as nn
from einx import add
from einops import rearrange

# AlphaGenome
from .schemas import Channels, DataBatch
from .metadata import Metadata
from .encoder import SequenceEncoder
from .transformer import TransformerTower
from .decoder import SequenceDecoder
from .output import OutputEmbedder, OutputPairEmbedder
from .heads import create_heads, HeadName
from .embeddings import Embeddings
from .splicing import generate_splice_site_positions



class AlphaGenomeConfig():
    """
    NOTE: The max_seq_len and metadata (determining the number of organisms, heads, and tracks)
    are the only hyperparameters with defaults that differ from the published AlphaGenome model.

    NOTE: Some hyperparameters (channel_increment, qk_head_dim, v_head_dim, pair_channels, 
    num_splice_contexts, splice_site_channels) are defaulted to keep their proportions consistent 
    with the published model wrt either channels or sequence length. This way, user can simply 
    specify num_channels and max_seq_len if they want to scale the model size up or down.The only 
    caveat is pos_channels, whose default is kept at 64 because the induced complexity is not worth
    the cheap computational cost (it's very cheap a dynamic default would make more sense to be 
    based on max_seq_len, which could unintentionally cause users to not be able to load the 
    published AlphaGenome weights).
    """
    def __init__(
        self,
        # Required
        max_seq_len: int = 8192,
        num_channels: int = 768,
        channel_increment: int = None,
        transformer_layers: int = 9,
        first_conv_width: int = 15,
        block_width: int = 5,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        qk_head_dim: int = None,
        v_head_dim: int = None,
        pair_channels: int = None,
        pair_heads: int = 32,
        pos_channels: int = 64,
        transformer_mlp_ratio: int = 2,
        init_scale: float = 0.9,
        embedder_mlp_ratio: int = 2,
        dropout: float = 0.0,
        num_splice_sites: int = None,
        splice_site_channels: int = None,
        splice_site_threshold: float = 0.1,
        min_zero_multinomial_loss: bool = True,
        metadata: Union[Metadata, dict] = None,
        **kwargs  # Catches unexpected args if the config is changed in future versions
    ):
        self.model_name = "AlphaGenome"
        self.max_seq_len = max_seq_len
        self.num_channels = num_channels
        self.channel_increment = channel_increment if channel_increment is not None else num_channels // 6
        self.transformer_layers = transformer_layers
        self.first_conv_width = first_conv_width
        self.block_width = block_width
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim if qk_head_dim is not None else (num_channels // 12)*2       # (must be kept even for RoPE)
        self.v_head_dim = v_head_dim if v_head_dim is not None else (num_channels // 12)*2          # (must be kept even for RoPE)
        self.pair_channels = pair_channels if pair_channels is not None else num_channels // 6
        self.pair_heads = pair_heads
        self.pos_channels = pos_channels
        self.transformer_mlp_ratio = transformer_mlp_ratio
        self.init_scale = init_scale
        self.embedder_mlp_ratio = embedder_mlp_ratio
        self.dropout = dropout
        if isinstance(metadata, dict):
            metadata = Metadata(metadata)
        self.metadata = metadata
        self.num_splice_sites = num_splice_sites if num_splice_sites is not None else self.max_seq_len // 2048
        self.splice_site_channels = splice_site_channels if splice_site_channels is not None else num_channels
        self.splice_site_threshold = splice_site_threshold
        self.min_zero_multinomial_loss = min_zero_multinomial_loss

    def to_dict(self) -> dict:
        return self.__dict__
    
    def save(self, cfg_path: Union[Path, str], metadata_path: Union[Path, str]):
        cfg_path = Path(cfg_path)
        metadata_path = Path(metadata_path)

        cfg = self.to_dict().copy()
        cfg.pop("metadata", None)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4)
        metadata_obj = self.metadata.metadata if isinstance(self.metadata, Metadata) else self.metadata
        torch.save(metadata_obj, metadata_path)
    
    @staticmethod
    def load(
        cfg_path: Union[Path, str], metadata_path: Union[Path, str]
    ) -> "AlphaGenomeConfig":
        cfg_path = Path(cfg_path)
        metadata_path = Path(metadata_path)
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)

        metadata_raw = torch.load(metadata_path, map_location="cpu")
        if isinstance(metadata_raw, Metadata):
            metadata_obj = metadata_raw
        elif isinstance(metadata_raw, dict):
            metadata_obj = Metadata(metadata_raw)
        else:
            raise TypeError(f"Unsupported metadata type in {metadata_path}: {type(metadata_raw)}")
        cfg_dict["metadata"] = metadata_obj

        return AlphaGenomeConfig(**cfg_dict)



class AlphaGenome(nn.Module):
    def __init__(self, cfg: AlphaGenomeConfig):
        super().__init__()

        # Read inputs
        self.cfg = cfg
        self.num_channels = cfg.num_channels                                                                    # C
        self.channel_increment = cfg.channel_increment                                                          # I
        self.transformer_layers = cfg.transformer_layers
        self.max_seq_len = cfg.max_seq_len                                                                  # S
        self.first_conv_width = cfg.first_conv_width
        self.block_width = cfg.block_width
        self.num_q_heads = cfg.num_q_heads                                                                      # H_q
        self.num_kv_heads = cfg.num_kv_heads                                                                    # H_kv
        self.qk_head_dim = cfg.qk_head_dim                                                                      # D_qk
        self.v_head_dim = cfg.v_head_dim                                                                        # D_v
        self.pair_channels = cfg.pair_channels                                                                  # F
        self.pair_heads = cfg.pair_heads                                                                        # H_p
        self.pos_channels = cfg.pos_channels                                                                    # C_p
        self.transformer_mlp_ratio = cfg.transformer_mlp_ratio                                                  # M
        self.embedder_mlp_ratio = cfg.embedder_mlp_ratio                                                        # M'
        self.init_scale = cfg.init_scale
        self.dropout = cfg.dropout
        self.num_splice_sites = cfg.num_splice_sites
        self.splice_site_channels = cfg.splice_site_channels
        self.splice_site_threshold = cfg.splice_site_threshold
        self.min_zero_multinomial_loss = cfg.min_zero_multinomial_loss

        # Read the metadata (make class if in dict form)
        if isinstance(cfg.metadata, dict):
            self.metadata = Metadata(cfg.metadata) 
        else:
            self.metadata = cfg.metadata
        self.num_organisms = self.metadata.get_num_organisms()

        # Strictly define the downsampling args to retain consistent bp-resolution
        self.encoder_downsample_width = 2
        self.stages = 7
        self.pair_downsample_width = 16

        # Check inputs
        self.check_seq_len(self.max_seq_len)
        assert (self.qk_head_dim % 2) == 0, "qk_head_dim must be even for RoPE."
        assert (self.v_head_dim % 2) == 0, "v_head_dim must be even for RoPE."
        assert (self.num_q_heads % self.num_kv_heads) == 0, "num_q_heads must be divisible by num_kv_heads."

        # Define sizes throughout the model
        self.channel_sizes = [4] + [self.num_channels + self.channel_increment * i for i in range(self.stages)]     # [4, C, C+I, ..., C+(S-1)I]
        self.transformer_max_seq_len = self.max_seq_len // (self.encoder_downsample_width ** self.stages)             # S'

        # Modules
        self.encoder = SequenceEncoder(
            channel_sizes=self.channel_sizes,
            first_conv_width=self.first_conv_width,
            encoder_downsample_width=self.encoder_downsample_width,
            block_width=self.block_width
        )
        self.org_embedder = nn.Embedding(self.num_organisms, self.channel_sizes[-1])
        self.transformer = TransformerTower(
            num_channels=self.channel_sizes[-1],
            transformer_seq_len=self.transformer_max_seq_len,
            num_blocks=self.transformer_layers,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            pair_downsample_width=self.pair_downsample_width,
            pair_channels=self.pair_channels,
            pair_heads=self.pair_heads,
            pos_channels=self.pos_channels,
            mlp_ratio=self.transformer_mlp_ratio,
            dropout=self.dropout
        )
        self.decoder = SequenceDecoder(
            channel_sizes=self.channel_sizes,
            block_width=self.block_width,
            init_scale=self.init_scale
        )
        self.embedder_t = OutputEmbedder(
            self.channel_sizes[-1],
            self.num_organisms,
            mlp_ratio=self.embedder_mlp_ratio,
        )
        self.embedder_x = OutputEmbedder(
            self.channel_sizes[1],
            self.num_organisms,
            mlp_ratio=self.embedder_mlp_ratio,
            skip_channels=self.channel_sizes[-1]*self.embedder_mlp_ratio
        )
        self.embedder_pair = OutputPairEmbedder(
            pair_channels=self.pair_channels,
            num_organisms=self.num_organisms,
        )
        self.channels = Channels(
            channels_1bp=self.channel_sizes[1]*self.embedder_mlp_ratio,
            channels_128bp=self.channel_sizes[-1]*self.embedder_mlp_ratio,
            channels_pair=self.pair_channels,
        )
        self._heads = create_heads(
            max_seq_len=self.max_seq_len,
            channels=self.channels,
            splice_site_channels=self.splice_site_channels,
            min_zero_multinomial_loss=self.min_zero_multinomial_loss,
            metadata=self.metadata,
        )

    def check_seq_len(self, seq_len: int):
        min_seq_len = (self.encoder_downsample_width**self.stages) * self.pair_downsample_width
        if seq_len < min_seq_len:
            raise ValueError(
                "Input sequence length must be at least "
                "(encoder_downsample_width^(stages) * pair_downsample_width)."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length must be <= max_seq_len={self.max_seq_len}, got {seq_len}."
            )
        if seq_len % min_seq_len != 0:
            raise ValueError(
                "Input sequence length must be divisible by "
                "(encoder_downsample_width^(stages) * pair_downsample_width)."
            )
    
    @property
    def _num_organisms(self) -> int:
        return self.metadata.get_num_organisms()

    def predict_junctions(
        self,
        embeddings: Embeddings,                     # Embeddings object containing 1bp, 128bp, and pair embeddings
        splice_site_positions: torch.Tensor,        # [B, 4, K]
        organism_index: torch.Tensor,               # [B]
        tissue_mask: torch.Tensor                   # [B, #A, #D, T]
    ) -> torch.Tensor:
        """Predicts splice site junctions from embeddings and splice site positions.

        Splice site positions has order: [donor_pos_idx, accept_pos_idx, donor_neg_idx, accept_neg_idx]
        """
        junction_head = self._heads[HeadName.SPLICE_SITES_JUNCTION.value]
        if junction_head is None:
            raise ValueError('Junction head is not supported by this model.')
        return junction_head(
            embeddings=embeddings,
            organism_index=organism_index,
            splice_site_positions=splice_site_positions,
            tissue_mask=tissue_mask,
        )

    def forward(self, batch: DataBatch, organism_index: torch.Tensor | None = None, return_embeddings: bool = True):
        # Unpack batch
        x = batch.dna_sequence                                                          # [B, S, V]
        B, S, _ = x.shape
        if organism_index is None:
            organism_index = torch.zeros(B, dtype=torch.long, device=x.device)          # [B]

        # Check inputs
        self.check_seq_len(S)
        
        # Encode sequence with CNN encoder
        x = rearrange(x, 'b v s -> b s v')                                              # [B, S, 4]
        t, intermediates = self.encoder(x)                                              # [B, 4, S] --> x: [B, C', S'] | intermediates: [B, C_i, S // 2^i] for U-Net skip connections

        # Add organism embedding
        if self._num_organisms >= 1:
            org_embed = self.org_embedder(organism_index)
            t = add('b c s, b c -> b c s', t, org_embed)                                # [B, C', S'] + [B, C', 1] --> [B, C', S']

        # Transformer tower
        t = rearrange(t, 'b c s -> b s c')                                              # [B, S', C']
        t, pair_activations = self.transformer(t)                                       # x: [B, S', C'] | pair_activations: [B, P, P, F]
        t = rearrange(t, 'b s c -> b c s')                                              # [B, C', S']

        # Decode sequence with CNN decoder
        x = self.decoder(t, intermediates)                                              # x: [B, C', S'] | intermediates: [B, C_i, S // 2^i] --> [B, C, S]

        # Output embedders
        t = rearrange(t, 'b c s -> b s c')                                              # [B, S', C']
        x = rearrange(x, 'b c s -> b s c')                                              # [B, S, C]
        embeddings_t = self.embedder_t(t, organism_index)                               # [B, S', C'*M']
        embeddings_x = self.embedder_x(x, organism_index, skip_x=embeddings_t)          # [B, S, C'*M']
        embeddings_pair = self.embedder_pair(pair_activations, organism_index)          # [B, P, P, F]

        # Collect embeddings
        embeddings = Embeddings(
            embeddings_1bp=embeddings_x,
            embeddings_128bp=embeddings_t,
            embeddings_pair=embeddings_pair,
        )
        predictions = {}

        for head_name, head_fn in self._heads.items():
            if not self.metadata.metadata['heads'][head_name].get("enabled", True):
                # If the head is not enabled, skip it
                continue
            if head_name == HeadName.SPLICE_SITES_JUNCTION.value:
                # This head is handled separately (see below).
                continue
            predictions[head_name] = head_fn(
                embeddings,
                organism_index,
            )

        # Handle the splice junction head separately. It requires splice site
        # positions as input, which are derived from the splice site
        # classification predictions.
        junction_head = HeadName.SPLICE_SITES_JUNCTION.value
        if (
            junction_head in self._heads and 
            self.metadata.metadata['heads'][junction_head].get("enabled", True)
        ):
            if (
                HeadName.SPLICE_SITES_CLASSIFICATION.value not in self._heads or
                not self.metadata.metadata['heads'][HeadName.SPLICE_SITES_CLASSIFICATION.value].get("enabled", True)
            ):
                raise ValueError(
                    'SPLICE_SITES_CLASSIFICATION head is required for junctions'
                    ' predictions.'
                )
            splice_sites_probabilities = predictions[
                HeadName.SPLICE_SITES_CLASSIFICATION.value
            ]['predictions']
            splice_site_positions = generate_splice_site_positions(
                splice_sites_probabilities,
                alt=None,
                splice_sites=None,
                k=self.num_splice_sites,
                pad_to_length=self.num_splice_sites,
                threshold=self.splice_site_threshold,
            )  # [B, 4, K]
            predictions[junction_head] = self.predict_junctions(
                embeddings, splice_site_positions, 
                organism_index, tissue_mask=batch.splice_site_junction_mask
            )
        if return_embeddings:
            return predictions, embeddings
        else:
            return predictions

    def loss(self, batch: DataBatch):
        predictions, _ = self(batch, batch.get_organism_index())
        total_loss, all_scalars = 0.0, {}
        for head_name, head_fn in self._heads.items():
            if not self.metadata.metadata['heads'][head_name].get("enabled", True):
                # If the head is not enabled, skip it
                continue
            scalars = head_fn.loss(predictions[head_name], batch)
            all_scalars.update(
                {f'{head_name}_{k}': v for k, v in scalars.items()}
            )
            total_loss += scalars['loss']
        return total_loss, all_scalars, predictions
