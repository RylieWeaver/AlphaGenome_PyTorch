from __future__ import annotations

# External
import numpy as np
import torch

# Internal
from alphagenome_pt import get_dtype_policy
from .batches import equivalence_num_splice_sites, make_jax_target_batch
from .precision import (
    jax_mixed_precision_policy,
    use_dtype_policy,
    use_jax_compute_uptype_policy,
)
from .utils import jax_device, use_jax_junction_padding_mask


##### CONSTANTS #####
ALL_SHARED_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
    "contact_maps",
    "splice_sites_classification",
    "splice_sites_usage",
    "splice_sites_junction",
)
MODEL_VERSION = "all_folds"


### HELPERS ###
def shared_head_outputs(predictions):
    return {
        f"heads/{head}/{name}": value
        for head, head_outputs in predictions.items()
        if head in ALL_SHARED_HEADS
        for name, value in head_outputs.items()
    }


### PYTORCH ###
def load_pytorch_model(device):
    from alphagenome_pt import AlphaGenome, deepmind_config, load_deepmind_state

    model = AlphaGenome(deepmind_config())
    load_deepmind_state(model, fold=MODEL_VERSION)
    model.to(device)
    model.eval()
    return model


def run_pytorch(
    model,
    batch,
    device,
    *,
    policy_name,
    include_loss=True,
):
    pt_dtype_policy = get_dtype_policy(policy_name)
    with use_dtype_policy(model, pt_dtype_policy):
        batch = batch.to(torch.device(device))
        with torch.inference_mode():
            if include_loss:
                output = model.loss(
                    batch,
                    return_predictions=True,
                    return_embeddings=True,
                )
                predictions = output.predictions
                embeddings = output.embeddings
            else:
                predictions, embeddings = model(
                    batch, mode="predict", return_embeddings=True
                )

        assert predictions is not None
        assert embeddings is not None
        captured = {
            "embeddings/1bp": embeddings.embeddings_1bp,
            "embeddings/128bp": embeddings.embeddings_128bp,
            "embeddings/pair": embeddings.embeddings_pair,
        }
        if include_loss:
            captured["loss/total"] = output.total
            captured.update({
                f"loss/{name}": value
                for name, value in output.tree.head_loss_totals().items()
            })
            captured.update({
                f"loss/{'/'.join(path)}": leaf.value
                for path, leaf in output.tree.iter_leaves()
                if len(path) == 3 and path[1].endswith("bp")
            })
        captured.update(shared_head_outputs(predictions))
        # Keep framework dtypes intact for the equivalence report. Numerical
        # conversion happens only after record_and_assert_close observes them.
        captured = {
            name: value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else value
            for name, value in captured.items()
        }

        del predictions, embeddings, batch
        if include_loss:
            del output
        return captured


### JAX ###
def load_jax_model(device):
    from alphagenome.models import dna_model as public_dna_model
    from alphagenome_research.model.dna_model import (
        OrganismSettings,
        create_from_huggingface,
    )

    device = jax_device(device)
    settings = {
        public_dna_model.Organism.HOMO_SAPIENS: OrganismSettings(),
        public_dna_model.Organism.MUS_MUSCULUS: OrganismSettings(),
    }
    return create_from_huggingface(
        model_version=MODEL_VERSION,
        organism_settings=settings,
        device=device,
    )


def _jax_embeddings(
    model,
    jax_dtype_policy,
    dna_sequence,
    organism_index,
):
    """Run the official trunk separately when loss() does not return embeddings."""
    import haiku as hk
    from alphagenome_research.model import model as research_model

    @hk.transform_with_state
    def embeddings(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(research_model.AlphaGenome, jax_dtype_policy):
            return research_model.AlphaGenome(
                model._metadata,
                num_splice_sites=equivalence_num_splice_sites(
                    dna_sequence.shape[1]
                ),
            ).forward_trunk(dna_sequence, organism_index)

    return embeddings.apply(
        model._params, model._state, None, dna_sequence, organism_index
    )


def run_jax(
    model,
    batch,
    device,
    *,
    policy_name,
    include_loss=True,
):
    import haiku as hk
    import jax
    from alphagenome_research.model import heads as research_heads
    from alphagenome_research.model import model as research_model

    pt_dtype_policy = get_dtype_policy(policy_name)
    jax_dtype_policy = jax_mixed_precision_policy(pt_dtype_policy)
    num_splice_sites = equivalence_num_splice_sites(
        batch.dna_sequence_one_hot.shape[1]
    )
    batch = make_jax_target_batch(batch)
    batch = jax.device_put(batch, jax_device(device))
    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(research_model.AlphaGenome, jax_dtype_policy):
            return research_model.AlphaGenome(
                model._metadata,
                num_splice_sites=num_splice_sites,
            )(dna_sequence, organism_index)

    @hk.transform_with_state
    def loss(batch):
        with hk.mixed_precision.push_policy(research_model.AlphaGenome, jax_dtype_policy):
            return research_model.AlphaGenome(
                model._metadata,
                num_splice_sites=num_splice_sites,
            ).loss(batch)

    with (
        use_jax_compute_uptype_policy(pt_dtype_policy),
        use_jax_junction_padding_mask(),
    ):
        if include_loss:
            (total, scalars, predictions), _state = loss.apply(
                model._params, model._state, None, batch
            )
            embeddings, _state = _jax_embeddings(
                model,
                jax_dtype_policy,
                batch.dna_sequence,
                batch.organism_index,
            )
        else:
            (predictions, embeddings), _state = forward.apply(
                model._params,
                model._state,
                None,
                batch.dna_sequence,
                batch.organism_index,
            )

    junction = predictions["splice_sites_junction"]

    captured = {
        "embeddings/1bp": embeddings.embeddings_1bp,
        "embeddings/128bp": embeddings.embeddings_128bp,
        "embeddings/pair": embeddings.embeddings_pair,
        "splice_site_positions": junction["splice_site_positions"],
    }
    if include_loss:
        captured["loss/total"] = total
        captured.update({
            f"loss/{head}": (
                research_heads.get_head_config(
                    research_heads.HeadName(head)
                ).loss_weight
                * scalars[f"{head}_loss"]
            )
            for head in ALL_SHARED_HEADS
        })
        for head in ALL_SHARED_HEADS:
            prefix = f"{head}_loss_total_"
            for name, total_count in scalars.items():
                if not name.startswith(prefix):
                    continue
                resolution = name.removeprefix(prefix)
                resolution_loss = scalars[f"{head}_loss_{resolution}"]
                captured[f"loss/{head}/{resolution}/total_count"] = total_count
                captured[f"loss/{head}/{resolution}/positional"] = (
                    resolution_loss - total_count
                )
    captured.update(shared_head_outputs(predictions))
    return jax.tree.map(np.asarray, captured)
