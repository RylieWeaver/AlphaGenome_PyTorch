# External
import torch

# Internal
from alphagenome_pt import (
    HeadName,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)
from utils import resolve_device


"""
Train the MLM head on synthetic data.
"""


if __name__ == "__main__":
    device = resolve_device()

    metadata = synthetic_metadata(
        heads=(HeadName.MASKED_LANGUAGE_MODELING,),
        num_organisms=2,
    )
    model = small_alphagenome(
        metadata,
        num_channels=96,
    ).to(device)
    batch = synthetic_batch(metadata, seq_len=model.max_seq_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # NOTE: not using example_train() here to show the accuracy
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        total_loss, _, predictions = model.loss(batch)
        accuracy = (
            predictions[HeadName.MASKED_LANGUAGE_MODELING.value]["predictions"]
            .argmax(dim=-1)
            .eq(batch.mlm)
            .float()
            .mean()
        )
        print(f"Step {step}: loss={total_loss.item():.6g}, accuracy={accuracy.item():.4f}")
        total_loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "AG_mlm_pretrained.pt")
    print("SUCCESS")
