# General
import random

# Torch
import torch

# AlphaGenome
from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig
from alphagenome_pytorch.schemas import DataBatch
from alphagenome_pytorch.sequence_encoder import SequenceEncoder
from utils import resolve_device, move_to, bert_mlm



if __name__ == "__main__":
    # Define the device
    device = resolve_device()

    # Define some size variables for synthetic data/metadata generation
    organisms = ['human', 'mouse']

    # Define metadata
    metadata = {
        'organisms': organisms,
        'heads': {
            'masked_language_modeling': {},
        },
    }

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

    # Define the synthetic data
    ## DNA
    seq1 = "".join(random.choices("ACGT", k=S))
    seq2 = "".join(random.choices("ACGT", k=S))
    organism_index = torch.tensor([0, 1], device=device)  # First sequence is human, second is mouse
    ## Mask
    masked_seqs, labels = bert_mlm([seq1, seq2])
    ## Encode
    seq_encoder = SequenceEncoder()
    masked_dna_sequence = seq_encoder.encode(masked_seqs).to(device)  # [B, S, 4]
    mlm_labels = seq_encoder.labels_encode(labels).to(device)   # [B, S] (class indices for ACGT, with masked tokens as 4)
    
    # Batch together the data
    data = DataBatch(
        dna_sequence=masked_dna_sequence,
        organism_index=organism_index,
        mlm=mlm_labels,
    ).to(device)
    # print(data)

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

    # Save the model (lets you use train_downstream_from_pretrained.py)
    torch.save(model.state_dict(), "AG_mlm_pretrained.pth") 
    
    # test = model(data)
    # print(test)
    print("SUCCESS!!!")
