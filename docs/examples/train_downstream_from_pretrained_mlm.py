# General
import random
from math import sqrt

# Torch
import torch
import torch.nn.functional as F

# AlphaGenome
from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig
from alphagenome_pytorch.metadata import Metadata
from alphagenome_pytorch.schemas import DataBatch
from alphagenome_pytorch.sequence_encoder import SequenceEncoder
from utils import resolve_device, move_to, bert_mlm, make_means


"""
Below is an example of how to pretrain the AlphaGenome model on masked language modeling,
then fine-tune it on downstream tasks (RNA-seq, CAGE, ATAC, and splice site prediction) 
using synthetic data. This serves as a template for training on real data.

NOTE: The downstream tasks' loss may be higher than you expect. This is because of a "positional 
loss" term (formulated as Cross-Entropy) in the genome track heads (RNA-seq, CAGE, ATAC) that 
has nonzero loss even for perfect predictions (see losses.py -> multinomial_loss()). A 
'loss_positional_zero_floor' scalar is included in the returned scalars from multinomial_loss() 
to help give an idea of the scale of the positional loss component when the predictions are perfect, 
and future releases may expose an argument to use the zero floor loss term in the total loss. 
However, for now, we stick to the original AlphaGenome formulation of the loss.
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
        'organisms': organisms,
        'heads': {
            'rna_seq': {
                'num_tracks': T_rna,                # [1]
                'means': make_means(T_rna),         # [O, max(T_rna)]
            },
            'cage': {
                'num_tracks': T_cage,               # [1]
                'means': make_means(T_cage),        # [O, max(T_cage)]
            },
            'atac': {
                'num_tracks': T_atac,               # [1]
                'means': make_means(T_atac),        # [O, max(T_atac)]
            },
            'splice_sites_classification': {
                'num_tracks': T_ssc,                # [1]
            },
            'splice_sites_usage': {
                'num_tracks': T_ssu,                # [1]
            },
            'splice_sites_junction': {
                'num_tissues': T_ssj,               # [1]
            },
            'masked_language_modeling': {
            },
        },
    }
    metadata = Metadata(metadata=metadata)
    metadata.make_all_masks()

    # Define the model
    model_cfg = AlphaGenomeConfig(
        input_seq_len=2048,
        num_channels=96,
        metadata=metadata
    )
    model = AlphaGenome(model_cfg).to(device)
    # print(model)

    # Get synthetic data parameters from model/metadata
    S = model.input_seq_len
    k = model.num_splice_sites
    num_classes = max(T_ssc)

    ### Synthetic DNA ###
    seq1 = "".join(random.choices("ACGT", k=S))
    seq2 = "".join(random.choices("ACGT", k=S))
    seq_encoder = SequenceEncoder()
    dna_sequence = seq_encoder.encode([seq1, seq2])     # [B, S, V]
    organism_index = torch.tensor([0, 1])               # First sequence is human, second is mouse
    B = organism_index.shape[0]



    ##### PART 1: MLM PRETRAINING #####
    print("Starting MLM pretraining...")

    # Mask
    masked_seqs, labels = bert_mlm([seq1, seq2])
    # Encode
    masked_dna_sequence = seq_encoder.encode(masked_seqs).to(device)    # [B, S, 4]
    mlm_labels = seq_encoder.labels_encode(labels).to(device)           # [B, S] (class indices for ACGT, with masked tokens as 4)

    # Batch together the data
    data = DataBatch(
        dna_sequence=masked_dna_sequence,
        organism_index=organism_index,
        mlm=mlm_labels,
    ).to(device)
    # print(data)

    # Turn on only the MLM head
    for head_name in model.metadata.get_heads():
        if head_name == "masked_language_modeling":
            model.metadata.metadata['heads'][head_name]['enabled'] = True
        else:
            model.metadata.metadata['heads'][head_name]['enabled'] = False

    # Do some training
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(1000):
        optimizer.zero_grad()
        data = move_to(data, device=device)
        total_loss, all_scalars, predictions = model.loss(data)
        mlm_preds = predictions['masked_language_modeling']['predictions']  # [B, S, 5]
        mlm_true = data.mlm  # [B, S]
        accuracy = (mlm_preds.argmax(dim=-1) == mlm_true).float().mean()
        print(f"Step {step}: Loss: {total_loss.item()}, Accuracy: {accuracy.item()}")
        total_loss.backward()
        optimizer.step()

    # Save the model
    model.cfg.save(cfg_path="AG_mlm_pretrained_config.json", metadata_path="AG_mlm_pretrained_metadata.json")
    torch.save(model.state_dict(), "AG_mlm_pretrained.pth")
    print("Finished MLM pretraining and saved the model! Now starting fine-tuning on downstream tasks...")



    ### PART 2: FINE-TUNE ON DOWNSTREAM TASKS (RNA-seq, CAGE, ATAC, Splice Site Prediction) #####

    # We don't need to save/load the model here, but do so for completeness in the example
    model_cfg = AlphaGenomeConfig.load(cfg_path="AG_mlm_pretrained_config.json", metadata_path="AG_mlm_pretrained_metadata.json")
    model = AlphaGenome(model_cfg).to(device)
    model.load_state_dict(torch.load("AG_mlm_pretrained.pth"))

    # Turn on all heads for downstream tasks and turn off the MLM head
    for head_name in model.metadata.get_heads():
        if head_name == "masked_language_modeling":
            model.metadata.metadata['heads'][head_name]['enabled'] = False
        else:
            model.metadata.metadata['heads'][head_name]['enabled'] = True

    # Define the synthetic data
    """
    NOTE: Here, we get the masks directly from the metadata, which are constant across the 
    dataset (corresponding to padded tracks). However, if you want to have variable masks 
    across samples in your dataset, you can simply generate them accordingly and pass them 
    to the DataBatch in the same way.
    """
    heads = model.metadata.metadata['heads']

    ## RNA-seq
    rna_seq = torch.poisson(
        heads['rna_seq']['means'][organism_index].unsqueeze(1).repeat(1, S, 1)
    )                                                                                   # [B, S, T_rna]
    rna_seq_mask = heads['rna_seq']['track_mask'][organism_index].unsqueeze(1)          # [B, 1, T_rna]
    ## CAGE
    cage = torch.poisson(
        heads['cage']['means'][organism_index].unsqueeze(1).repeat(1, S, 1)
    )                                                                                   # [B, S, T_cage]
    cage_mask = heads['cage']['track_mask'][organism_index].unsqueeze(1)                # [B, 1, T_cage]
    ## ATAC
    atac = torch.poisson(
        heads['atac']['means'][organism_index].unsqueeze(1).repeat(1, S, 1)
    )                                                                                   # [B, S, T_atac]
    atac_mask = heads['atac']['track_mask'][organism_index].unsqueeze(1)                # [B, 1, T_atac]
    ## Splice Sites
    ### Classification
    splice_sites = torch.randint(0, num_classes, (2, S, 1)).to(device)                  # [B, S, 1]
    splice_sites = F.one_hot(
        splice_sites.squeeze(-1), num_classes=num_classes
    ).view(2, S, -1).to(torch.float32)                                                  # [B, S, num_classes]
    ### Usage
    splice_site_usage = torch.rand(2, S, max(T_ssu)).to(device)                         # [B, S, T_ssu]
    splice_site_usage_mask = (
        heads['splice_sites_usage']['track_mask'][organism_index].unsqueeze(1)          # [B, 1, T_ssu]
    )
    ### Junction
    splice_site_junction = torch.poisson(
        sqrt(k) * torch.ones((2, k, k, 2*max(T_ssj)), dtype=torch.int32)
    ).to(torch.float32).to(device)                                                      # [B, k, k, 2*T_ssj]  (2*T for two strands)
    splice_site_junction_mask = (
        heads['splice_sites_junction']['tissue_mask'][organism_index]
        .view(B, 1, 1, max(T_ssj))                                                      # [B, 1, 1, T_ssj] (will be duplicated for strands in the model)
    )


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
