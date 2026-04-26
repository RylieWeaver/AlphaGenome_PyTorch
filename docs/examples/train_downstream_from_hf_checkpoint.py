# General
import random
from math import sqrt
from pathlib import Path

# Torch
import torch
import torch.nn.functional as F

# AlphaGenome
from alphagenome_pt import (
    AlphaGenome,
    DataBatch,
    Metadata,
    SequenceEncoder,
    load_alphagenome_checkpoint,
    official_alphagenome_config,
)
from utils import resolve_device, move_to, make_means


"""
Below is an example of how to fine-tune the downstream AlphaGenome heads from
the converted official checkpoint hosted on Hugging Face.

This is intentionally parallel to train_downstream_from_pretrained_mlm.py. The
only conceptual difference is that pretrained weights come from Hugging Face
rather than from locally running the MLM pretraining section.

NOTE: The official checkpoint requires the official trunk shape, but we skip
checkpoint heads here so that custom downstream heads can be used. Runtime
sequence length can be shorter than the configured max length, but it must be a
multiple of 2048.
"""


if __name__ == "__main__":
    # Define the device
    device = resolve_device()

    # Define some size variables for synthetic data/metadata generation
    organisms = ['human', 'mouse']
    # Track numbers
    T_rna = [200, 150]
    T_cage = [100, 0]
    T_atac = [0, 150]
    T_ssc = [5, 5]
    T_ssu = [10, 10]
    T_ssj = [25, 25]

    # Define metadata
    metadata = {
        "organisms": organisms,
        "heads": {
            "rna_seq": {
                "num_tracks": T_rna,
                "means": make_means(T_rna),
            },
            "cage": {
                "num_tracks": T_cage,
                "means": make_means(T_cage),
            },
            "atac": {
                "num_tracks": T_atac,
                "means": make_means(T_atac),
            },
            'splice_sites_classification': {
                "num_tracks": T_ssc,
            },
            'splice_sites_usage': {
                "num_tracks": T_ssu,
            },
            'splice_sites_junction': {
                "num_tissues": T_ssj,
            },
        },
    }
    metadata = Metadata(metadata=metadata)
    metadata.make_all_masks()

    # Define the checkpoint-compatible official model with custom downstream metadata
    model_cfg = official_alphagenome_config(metadata=metadata)
    model = AlphaGenome(model_cfg)

    # Load pretrained weights from Hugging Face
    load_result = load_alphagenome_checkpoint(
        model,
        Path("alphagenome_converted_state_dict.pt"),
        heads=False,
    )
    print(f"Missing keys: {load_result.missing_keys}")
    print(f"Unexpected keys: {load_result.unexpected_keys}")
    model = model.to(device)
    # print(model)
    print("Finished loading HF checkpoint! Now starting fine-tuning on downstream tasks...")



    ### FINE-TUNE ON DOWNSTREAM TASKS #####

    # Turn on all downstream heads. This metadata does not include MLM
    for head_name in model.metadata.get_heads():
        model.metadata.metadata["heads"][head_name]["enabled"] = True

    # Runtime sequence length can be shorter than model.max_seq_len, but must
    # remain a multiple of 2048 for the pair/contact path
    S = 4096
    model.num_splice_sites = S // 2048  # Generally shorter sequence length should correspond to fewer splice sites and it's computationally expensive
    k = model.num_splice_sites
    num_classes = max(T_ssc)

    ### Synthetic DNA ###
    seq = "".join(random.choices("ACGT", k=S))
    seq_encoder = SequenceEncoder()
    dna_sequence = seq_encoder.encode([seq]).to(device)      # [B, S, V]
    organism_index = torch.tensor([0], device=device)        # Human
    B = organism_index.shape[0]

    # Define the synthetic data
    """
    NOTE: Here, we get the masks directly from the metadata, which are constant across the 
    dataset (corresponding to padded tracks). However, if you want to have variable masks 
    across samples in your dataset, you can simply generate them accordingly and pass them 
    to the DataBatch in the same way.
    """
    heads = model.metadata.metadata["heads"]

    ## RNA-seq
    rna_seq = torch.poisson(
        heads["rna_seq"]["means"][organism_index].unsqueeze(1).repeat(1, S, 1)
    ).to(device)                                                                                        # [B, S, T_rna]
    rna_seq_mask = heads["rna_seq"]["track_mask"][organism_index].unsqueeze(1).to(device)               # [B, 1, T_rna]
    ## CAGE
    cage = torch.poisson(
        heads["cage"]["means"][organism_index].unsqueeze(1).repeat(1, S, 1)
    ).to(device)                                                                                        # [B, S, T_cage]
    cage_mask = heads["cage"]["track_mask"][organism_index].unsqueeze(1).to(device)                     # [B, 1, T_cage]
    ## ATAC
    atac = torch.poisson(
        heads["atac"]["means"][organism_index].unsqueeze(1).repeat(1, S, 1)
    ).to(device)                                                                                        # [B, S, T_atac]
    atac_mask = heads["atac"]["track_mask"][organism_index].unsqueeze(1).to(device)                     # [B, 1, T_atac]
    ## Splice Sites
    ### Classification
    splice_sites = torch.randint(0, num_classes, (B, S, 1), device=device)                              # [B, S, 1]
    splice_sites = F.one_hot(
        splice_sites.squeeze(-1), num_classes=num_classes
    ).view(B, S, -1).to(torch.float32)                                                                  # [B, S, num_classes]
    ### Usage
    splice_site_usage = torch.rand(B, S, max(T_ssu), device=device)                                     # [B, S, T_ssu]
    splice_site_usage_mask = (
        heads["splice_sites_usage"]["track_mask"][organism_index].unsqueeze(1).to(device)               # [B, 1, T_ssu]
    )
    ### Junction
    splice_site_junction = torch.poisson(
        sqrt(k) * torch.ones((B, k, k, 2 * max(T_ssj)), dtype=torch.float32, device=device)
    ).to(torch.float32)                                                                                 # [B, k, k, 2*T_ssj]
    splice_site_junction_mask = (
        heads["splice_sites_junction"]["tissue_mask"][organism_index]
        .view(B, 1, 1, max(T_ssj))
        .to(device)
    )                                                                                                   # [B, 1, 1, T_ssj]


    # Batch together the data
    data = DataBatch(
        dna_sequence=dna_sequence,
        organism_index=organism_index,
        rna_seq=rna_seq,
        rna_seq_mask=rna_seq_mask,
        cage=cage,
        cage_mask=cage_mask,
        atac=atac,
        atac_mask=atac_mask,
        splice_sites=splice_sites,
        splice_site_usage=splice_site_usage,
        splice_site_usage_mask=splice_site_usage_mask,
        splice_site_junction=splice_site_junction,
        splice_site_junction_mask=splice_site_junction_mask,
    ).to(device)
    # print(data)


    # Add some backprop to test that the model is fully differentiable
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    for step in range(1000):
        optimizer.zero_grad()
        data = move_to(data, device=device)
        total_loss, all_scalars, predictions = model.loss(data)
        print(f"Step {step}: Loss: {total_loss.item()}")
        total_loss.backward()
        optimizer.step()

    # test = model(data)
    # print(test)
    print("SUCCESS!!!")
