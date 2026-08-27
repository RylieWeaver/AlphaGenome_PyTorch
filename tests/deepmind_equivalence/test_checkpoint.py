"""Selected-module and full-model parity using the published checkpoint."""

# External
from dataclasses import dataclass
from typing import Any
import numpy as np
import pytest
import torch

# Internal
from alphagenome_pt import AlphaGenome, DataBatch, Embeddings, deepmind_metadata
from .batches import load_dna_chunks, make_pytorch_batch
from .capture import (
    run_jax,
    run_pytorch,
    load_jax_model,
    load_pytorch_model,
)
from .precision import (
    equivalence_criteria,
    jax_dtype,
    jax_mixed_precision_policy,
    use_dtype_policy,
    use_jax_compute_uptype_policy,
)
from .utils import (
    jax_device,
    linspace_values,
    normal_values,
    one_hot_dna_values,
    use_jax_junction_padding_mask,
)


# NOTE: Used to generate two unique random inputs for testing purposes
SECONDARY_INPUT_SEED = 1


### SHAPES ###
MODULE_BATCH_SIZE = 1
ORGANISM_BATCH_SIZE = 2
SEQUENCE_LENGTH = 32
ENCODER_SEQUENCE_LENGTH = 256
OUTPUT_1BP_SEQUENCE_LENGTH = 256

DNA_SHAPE = (MODULE_BATCH_SIZE, SEQUENCE_LENGTH, 4)
ENCODER_DNA_SHAPE = (MODULE_BATCH_SIZE, ENCODER_SEQUENCE_LENGTH, 4)
DOWN_RESOLUTION_INPUT_SHAPE = (MODULE_BATCH_SIZE, SEQUENCE_LENGTH, 768)
UP_RESOLUTION_INPUT_SHAPE = (MODULE_BATCH_SIZE, SEQUENCE_LENGTH // 2, 1536)
UP_RESOLUTION_SKIP_SHAPE = (MODULE_BATCH_SIZE, SEQUENCE_LENGTH, 1536)
TRANSFORMER_SHAPE = (MODULE_BATCH_SIZE, SEQUENCE_LENGTH, 1536)
PAIR_SHAPE = (
    MODULE_BATCH_SIZE, SEQUENCE_LENGTH // 16, SEQUENCE_LENGTH // 16, 128
)
OUTPUT_128BP_INPUT_SHAPE = (ORGANISM_BATCH_SIZE, SEQUENCE_LENGTH, 1536)
OUTPUT_1BP_INPUT_SHAPE = (
    ORGANISM_BATCH_SIZE, OUTPUT_1BP_SEQUENCE_LENGTH, 768
)
OUTPUT_1BP_SKIP_SHAPE = (
    ORGANISM_BATCH_SIZE, OUTPUT_1BP_SEQUENCE_LENGTH // 128, 3072
)
OUTPUT_PAIR_INPUT_SHAPE = (
    ORGANISM_BATCH_SIZE,
    SEQUENCE_LENGTH,
    SEQUENCE_LENGTH,
    128,
)
RNA_SEQ_1BP_EMBEDDING_SHAPE = (
    ORGANISM_BATCH_SIZE, OUTPUT_1BP_SEQUENCE_LENGTH, 1536
)
RNA_SEQ_128BP_EMBEDDING_SHAPE = (
    ORGANISM_BATCH_SIZE, SEQUENCE_LENGTH,  3072
)
JUNCTION_INPUT_SHAPE = (ORGANISM_BATCH_SIZE, SEQUENCE_LENGTH, 1536)
JUNCTION_NUM_SITES = 4
# NOTE: Counts are expected to be non-negative and can be large
JUNCTION_COUNT_TARGET_RANGE = (0.0, 1000.0)


### CHECKPOINT HELPERS ###
@pytest.fixture(scope="module")
def checkpoint_backend_settings(pytestconfig):
    """Apply matched JAX and PyTorch settings during checkpoint tests."""
    # NOTE: These args are relevant to GPU execution so only included in this file.
    if not pytestconfig.getoption("--run-equivalence"):
        yield
        return

    import jax

    original_settings = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )

    # Request full float32 precision for ordinary JAX matrix multiplications
    # and convolutions that do not specify an explicit precision.
    with jax.default_matmul_precision("float32"):
        # Disable TensorFloat-32 so FP32 matrix multiplications and convolutions use
        # full float32 precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Force deterministic cuDNN algorithms and disable runtime benchmarking so
        # repeated parity runs use the same kernels.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            yield
        finally:
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
            ) = original_settings


@dataclass
class CheckpointModels:
    jax: Any
    pytorch: AlphaGenome
    device: str
    jax_device: Any


@pytest.fixture(scope="module")
def checkpoint_models(
    pytestconfig, checkpoint_backend_settings
) -> CheckpointModels:
    """Load one JAX model and one PyTorch model for all checkpoint tests."""
    if not pytestconfig.getoption("--run-equivalence"):
        pytest.skip("pass --run-equivalence to compare checkpoint modules")

    pytest.importorskip("jax")
    pytest.importorskip("haiku")
    pytest.importorskip("alphagenome_research")

    device = pytestconfig.getoption("--checkpoint-equivalence-device")
    jax_model = load_jax_model(device)
    pytorch_model = load_pytorch_model(device)
    resolved_jax_device = jax_device(device)

    return CheckpointModels(
        jax=jax_model,
        pytorch=pytorch_model,
        device=device,
        jax_device=resolved_jax_device,
    )


def _run_jax(
    models: CheckpointModels, pt_dtype_policy, scopes, module, *inputs
):
    """Run a checkpoint module with the selected JAX compute policy."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    jax_dtype_policy = jax_mixed_precision_policy(pt_dtype_policy)

    class CheckpointModule(hk.Module):
        def __call__(self, *values):
            if not scopes:
                return module(*values)
            if len(scopes) == 1:
                with hk.name_scope(scopes[0]):
                    return module(*values)
            if len(scopes) == 2:
                with hk.name_scope(scopes[0]):
                    with hk.name_scope(scopes[1]):
                        return module(*values)
            raise ValueError(f"Only two nested Haiku scopes are supported: {scopes}")

    def forward(*values):
        with hk.mixed_precision.push_policy(CheckpointModule, jax_dtype_policy):
            return CheckpointModule(name="alphagenome")(*values)

    transformed = hk.transform_with_state(forward)
    jax_inputs = []
    for value in inputs:
        dtype = (
            jax_dtype(pt_dtype_policy.compute_dtype)
            if np.issubdtype(value.dtype, np.floating)
            else None
        )
        jax_inputs.append(
            jax.device_put(
                jnp.asarray(value, dtype=dtype), models.jax_device
            )
        )
    with (
        use_jax_compute_uptype_policy(pt_dtype_policy),
        use_jax_junction_padding_mask(),
    ):
        output, _ = transformed.apply(
            models.jax._params,
            models.jax._state,
            None,
            *jax_inputs,
        )
    return output


def _run_pytorch(
    models: CheckpointModels,
    pt_dtype_policy,
    module,
    *inputs,
    channels_first=False,
):
    """Run a checkpoint module with the selected PyTorch policy."""
    import torch

    values = [torch.as_tensor(x, device=models.device) for x in inputs]
    values = [
        value.to(pt_dtype_policy.compute_dtype) if value.is_floating_point() else value
        for value in values
    ]
    if channels_first:
        values = [x.permute(0, 2, 1) for x in values]
    module.eval()
    with use_dtype_policy(models.pytorch, pt_dtype_policy):
        with torch.inference_mode():
            return pt_dtype_policy.cast_output(module(*values))


### ENCODER/DECODER ###
def test_dna_embedder(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import convolutions

    values = one_hot_dna_values(DNA_SHAPE[0], DNA_SHAPE[1])
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("sequence_encoder",),
        lambda x: convolutions.DnaEmbedder()(x, is_training=False),
        values,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        checkpoint_models.pytorch.sequence_encoder.downres_blocks["bin_size_1"],
        values,
        channels_first=True,
    )
    torch_output = torch_output.permute(0, 2, 1)
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_down_resolution_block(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import convolutions

    values = normal_values(DOWN_RESOLUTION_INPUT_SHAPE)
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("sequence_encoder",),
        lambda x: convolutions.DownResBlock("downres_block_0")(
            x, is_training=False
        ),
        values,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        checkpoint_models.pytorch.sequence_encoder.downres_blocks["bin_size_2"],
        values,
        channels_first=True,
    )
    torch_output = torch_output.permute(0, 2, 1)
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_up_resolution_block(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import convolutions

    block = checkpoint_models.pytorch.sequence_decoder.upres_blocks["bin_size_64"]
    values = normal_values(UP_RESOLUTION_INPUT_SHAPE)
    skip = normal_values(
        UP_RESOLUTION_SKIP_SHAPE, seed=SECONDARY_INPUT_SEED
    )
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("sequence_decoder",),
        lambda x, s: convolutions.UpResBlock()(x, s, is_training=False),
        values,
        skip,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        block,
        values,
        skip,
        channels_first=True,
    )
    torch_output = torch_output.permute(0, 2, 1)
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_encoder_by_layer(
    checkpoint_models, pt_dtype_policy, record_and_assert_close
):
    from alphagenome_research.model import model as research_model

    values = one_hot_dna_values(ENCODER_DNA_SHAPE[0], ENCODER_DNA_SHAPE[1])
    bin_sizes = (1, 2, 4, 8, 16, 32, 64, 128)
    jax_trunk, jax_intermediates = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        (),
        lambda x: research_model.SequenceEncoder()(
            x, is_training=False
        ),
        values,
    )
    torch_values = torch.as_tensor(
        values,
        dtype=pt_dtype_policy.compute_dtype,
        device=checkpoint_models.device,
    ).permute(0, 2, 1)
    with use_dtype_policy(checkpoint_models.pytorch, pt_dtype_policy):
        with torch.inference_mode():
            torch_trunk, torch_intermediates = (
                checkpoint_models.pytorch.sequence_encoder(torch_values)
            )
    jax_representations = {
        **jax_intermediates,
        "bin_size_128": jax_trunk,
    }
    torch_representations = {
        name: pt_dtype_policy.cast_output(
            intermediate["embeddings"]
        ).permute(0, 2, 1)
        for name, intermediate in torch_intermediates.items()
    }
    torch_representations["bin_size_128"] = (
        pt_dtype_policy.cast_output(torch_trunk).permute(0, 2, 1)
    )

    representation_names = tuple(f"bin_size_{size}" for size in bin_sizes)
    expected_names = set(representation_names)
    assert set(torch_representations) == expected_names
    assert set(jax_representations) == expected_names

    failures = []
    for name in representation_names:
        try:
            record_and_assert_close(
                torch_representations[name],
                jax_representations[name],
                name=f"encoder/{name}",
                dtype_policy=pt_dtype_policy.name,
                **equivalence_criteria(
                    pt_dtype_policy.name, "architecture"
                ),
            )
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        pytest.fail(
            "Encoder-stage differences exceeded thresholds:\n"
            + "\n".join(failures)
        )

### TRANSFORMER ###
def test_pair_update_block(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import attention

    sequence = normal_values(TRANSFORMER_SHAPE)
    pair = normal_values(PAIR_SHAPE, seed=SECONDARY_INPUT_SEED)
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("transformer_tower",),
        lambda x, p: attention.PairUpdateBlock()(x, p),
        sequence,
        pair,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        checkpoint_models.pytorch.transformer_tower.blocks[0].pair_update,
        sequence,
        pair,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "architecture"),
    )


def test_transformer_block(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import attention

    sequence = normal_values(TRANSFORMER_SHAPE)
    pair = normal_values(PAIR_SHAPE, seed=SECONDARY_INPUT_SEED)

    # NOTE: Transformer block is not exposed as a single module in JAX
    def jax_block(x, pair_x):
        pair_x = attention.PairUpdateBlock()(x, pair_x)
        bias = attention.AttentionBiasBlock()(pair_x, is_training=False)
        x = x + attention.MHABlock()(x, bias, is_training=False)
        x = x + attention.MLPBlock()(x, is_training=False)
        return x, pair_x

    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("transformer_tower",),
        jax_block,
        sequence,
        pair,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        checkpoint_models.pytorch.transformer_tower.blocks[0],
        sequence,
        pair,
    )
    record_and_assert_close(
        torch_output[0],
        jax_output[0],
        name="sequence",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "architecture"),
    )
    record_and_assert_close(
        torch_output[1],
        jax_output[1],
        name="pair",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "architecture"),
    )


### OUTPUT EMBEDDINGS ###
def test_output_128bp_embedder(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import embeddings

    values = normal_values(OUTPUT_128BP_INPUT_SHAPE)
    organisms = np.arange(ORGANISM_BATCH_SIZE, dtype=np.int64)
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        (),
        lambda x, org: embeddings.OutputEmbedder(2)(
            x, org, is_training=False
        ),
        values,
        organisms,
    )
    torch_output = _run_pytorch(
        checkpoint_models, pt_dtype_policy, checkpoint_models.pytorch.output_t, values, organisms
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_output_1bp_embedder(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import embeddings

    output_x = checkpoint_models.pytorch.output_x
    values = normal_values(OUTPUT_1BP_INPUT_SHAPE)
    trunk_skip = normal_values(
        OUTPUT_1BP_SKIP_SHAPE, seed=SECONDARY_INPUT_SEED
    )
    organisms = np.arange(ORGANISM_BATCH_SIZE, dtype=np.int64)
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        (),
        lambda x, org, skip: embeddings.OutputEmbedder(
            2, name="output_embedder_1"
        )(x, org, is_training=False, skip_x=skip),
        values,
        organisms,
        trunk_skip,
    )
    torch_output = _run_pytorch(
        checkpoint_models,
        pt_dtype_policy,
        output_x,
        values,
        organisms,
        trunk_skip,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_output_pair_embedder(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import embeddings

    values = normal_values(OUTPUT_PAIR_INPUT_SHAPE)
    organisms = np.arange(ORGANISM_BATCH_SIZE, dtype=np.int64)
    jax_output = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        (),
        lambda x, org: embeddings.OutputPair(2)(x, org),
        values,
        organisms,
    )
    torch_output = _run_pytorch(
        checkpoint_models, pt_dtype_policy, checkpoint_models.pytorch.output_pair, values, organisms
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### PREDICTION HEADS ###
def test_rna_seq_head(checkpoint_models, pt_dtype_policy, record_and_assert_close):
    from alphagenome_research.model import embeddings as research_embeddings
    from alphagenome_research.model import heads as research_heads

    embeddings_1bp = normal_values(RNA_SEQ_1BP_EMBEDDING_SHAPE)
    embeddings_128bp = normal_values(
        RNA_SEQ_128BP_EMBEDDING_SHAPE, seed=SECONDARY_INPUT_SEED
    )
    organisms = np.arange(ORGANISM_BATCH_SIZE, dtype=np.int32)

    def jax_head(x_1bp, x_128bp, organism_index):
        config = research_heads.get_head_config(research_heads.HeadName.RNA_SEQ)
        head = research_heads.create_head(
            config,
            checkpoint_models.jax._metadata,
            num_organisms=ORGANISM_BATCH_SIZE,
        )
        return head(
            research_embeddings.Embeddings(
                embeddings_1bp=x_1bp,
                embeddings_128bp=x_128bp,
            ),
            organism_index,
        )

    jax_predictions = _run_jax(
        checkpoint_models,
        pt_dtype_policy,
        ("head",),
        jax_head,
        embeddings_1bp,
        embeddings_128bp,
        organisms,
    )

    torch_1bp = torch.as_tensor(
        embeddings_1bp,
        dtype=pt_dtype_policy.compute_dtype,
        device=checkpoint_models.device,
    )
    torch_128bp = torch.as_tensor(
        embeddings_128bp,
        dtype=pt_dtype_policy.compute_dtype,
        device=checkpoint_models.device,
    )
    torch_organisms = torch.as_tensor(
        organisms, device=checkpoint_models.device
    )
    with use_dtype_policy(checkpoint_models.pytorch, pt_dtype_policy):
        with torch.inference_mode():
            torch_predictions = pt_dtype_policy.cast_output(
                checkpoint_models.pytorch._heads["rna_seq"](
                    Embeddings(
                        embeddings_1bp=torch_1bp,
                        embeddings_128bp=torch_128bp,
                    ),
                    torch_organisms,
                )
            )

    track_mask = (
        checkpoint_models.pytorch.metadata.get_multiorg_track_mask(
            "rna_seq", torch_organisms
        )
        .cpu()
        .numpy()
    )
    for name in (
        "scaled_predictions_1bp",
        "predictions_1bp",
        "scaled_predictions_128bp",
        "predictions_128bp",
    ):
        torch_value = torch_predictions[name].detach().cpu()
        jax_value = jax_predictions[name]
        mask = np.broadcast_to(
            track_mask[:, None, :], jax_value.shape
        ).copy()
        record_and_assert_close(
            torch_value[torch.from_numpy(mask)],
            jax_value[mask],
            name=name,
            dtype_policy=pt_dtype_policy.name,
            **equivalence_criteria(pt_dtype_policy.name, "module"),
        )


def test_splice_junction_head_with_fixed_positions(
    checkpoint_models, pt_dtype_policy, record_and_assert_close
):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import heads as research_heads
    from alphagenome_research.model import model as research_model
    from alphagenome_research.model import schemas as research_schemas

    values = normal_values(JUNCTION_INPUT_SHAPE)
    organisms = np.arange(ORGANISM_BATCH_SIZE, dtype=np.int32)
    positions = np.array(
        [
            [[1, 5, 9, -1], [2, 7, 11, -1], [3, 8, 12, -1], [4, 10, 14, -1]],
            [[1, 6, 10, -1], [2, 7, 13, -1], [3, 9, 15, -1], [4, 11, 16, -1]],
        ],
        dtype=np.int32,
    )
    num_tracks = (
        checkpoint_models.pytorch._heads["splice_sites_junction"]._num_tracks
    )
    targets = linspace_values(
        (
            ORGANISM_BATCH_SIZE,
            JUNCTION_NUM_SITES,
            JUNCTION_NUM_SITES,
            num_tracks,
        ),
        *JUNCTION_COUNT_TARGET_RANGE,
    )
    jax_dtype_policy = jax_mixed_precision_policy(pt_dtype_policy)

    @hk.transform_with_state
    def jax_forward(x, site_positions, organism_index, target_values):
        with hk.mixed_precision.push_policy(
            research_model.AlphaGenome, jax_dtype_policy
        ):
            model = research_model.AlphaGenome(
                checkpoint_models.jax._metadata
            )
            predictions = model.predict_junctions(
                x, site_positions, organism_index
            )
            head = model._heads[
                research_heads.HeadName.SPLICE_SITES_JUNCTION
            ]
            loss = head.loss(
                predictions,
                research_schemas.DataBatch(splice_junctions=target_values),
            )["loss"]
            return predictions, loss

    jax_values = tuple(
        jax.device_put(value, checkpoint_models.jax_device)
        for value in (
            jnp.asarray(
                values,
                dtype=jax_dtype(pt_dtype_policy.compute_dtype),
            ),
            jnp.asarray(positions),
            jnp.asarray(organisms),
            jnp.asarray(
                targets,
                dtype=jax_dtype(pt_dtype_policy.compute_dtype),
            ),
        )
    )
    with (
        use_jax_compute_uptype_policy(pt_dtype_policy),
        use_jax_junction_padding_mask(),
    ):
        (jax_predictions, jax_loss), _ = jax_forward.apply(
            checkpoint_models.jax._params,
            checkpoint_models.jax._state,
            None,
            *jax_values,
        )

    torch_values = torch.as_tensor(
        values,
        dtype=pt_dtype_policy.compute_dtype,
        device=checkpoint_models.device,
    )
    torch_positions = torch.as_tensor(
        positions, device=checkpoint_models.device
    )
    torch_organisms = torch.as_tensor(
        organisms, device=checkpoint_models.device
    )
    torch_targets = torch.as_tensor(
        targets,
        dtype=pt_dtype_policy.compute_dtype,
        device=checkpoint_models.device,
    )
    torch_head = checkpoint_models.pytorch._heads[
        "splice_sites_junction"
    ]
    with use_dtype_policy(checkpoint_models.pytorch, pt_dtype_policy):
        with torch.inference_mode():
            torch_predictions = checkpoint_models.pytorch.predict_junctions(
                Embeddings(embeddings_1bp=torch_values),
                torch_positions,
                torch_organisms,
            )
            torch_loss_tree = torch_head.loss(
                torch_predictions,
                DataBatch(splice_junctions=torch_targets),
            )
            torch_loss = sum(
                leaf.value
                for group in torch_loss_tree.values()
                for leaf in group.values()
            )

    record_and_assert_close(
        torch_predictions["splice_junction_mask"],
        jax_predictions["splice_junction_mask"],
        name="mask",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "exact"),
    )
    record_and_assert_close(
        pt_dtype_policy.cast_output(torch_predictions["predictions"]),
        jax_predictions["predictions"],
        name="predictions",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )
    jax_head_weight = research_heads.get_head_config(
        research_heads.HeadName.SPLICE_SITES_JUNCTION
    ).loss_weight
    record_and_assert_close(
        torch_loss,
        jax_head_weight * jax_loss,
        name="loss",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### FULL MODEL ###
def test_full_model(checkpoint_models, pt_dtype_policy, record_and_assert_close, pytestconfig):
    device = pytestconfig.getoption("--checkpoint-equivalence-device")
    sequence_length = pytestconfig.getoption("--equivalence-sequence-length")
    if sequence_length % 2048:
        pytest.fail("model equivalence sequence length must be divisible by 2048")
    if sequence_length >= 131072 and sequence_length % 131072:
        pytest.fail(
            "loss equivalence sequence length must be divisible by 131072"
        )
    include_loss = sequence_length >= 131072
    batch = make_pytorch_batch(
        deepmind_metadata(),
        sequence_length=sequence_length,
        seed=42,
    )
    dna_paths = pytestconfig.getoption("--equivalence-dna")
    if dna_paths:
        try:
            batch.dna_sequence_one_hot = load_dna_chunks(
                dna_paths,
                sequence_length,
            )
        except (OSError, ValueError) as error:
            pytest.fail(str(error), pytrace=False)
    jax_values = run_jax(
        checkpoint_models.jax,
        batch,
        device,
        policy_name=pt_dtype_policy.name,
        include_loss=include_loss,
    )
    np.testing.assert_array_equal(
        jax_values["splice_site_positions"],
        jax_values[
            "heads/splice_sites_junction/splice_site_positions"
        ],
    )
    batch.splice_site_positions = torch.from_numpy(
        jax_values.pop("splice_site_positions").copy()
    )
    pytorch_values = run_pytorch(
        checkpoint_models.pytorch,
        batch,
        device,
        policy_name=pt_dtype_policy.name,
        include_loss=include_loss,
    )

    assert pytorch_values.keys() == jax_values.keys()
    # Use the widest composition tier for full-model relative-norm checks.
    failures = []
    for name in jax_values:
        pytorch_value = pytorch_values[name]
        jax_value = jax_values[name]
        path = name.split("/")
        if (
            len(path) == 3
            and path[0] == "heads"
            and (
                path[2].startswith("predictions_")
                or path[2].startswith("scaled_predictions_")
            )
        ):
            # JAX represents padded track means as NaN, whereas PyTorch uses
            # finite fillers. Padded tracks are not model outputs, so compare
            # scaled and unscaled predictions only over valid organism tracks.
            track_mask = (
                checkpoint_models.pytorch.metadata.get_multiorg_track_mask(
                    path[1], batch.get_organism_index()
                )
                .cpu()
                .numpy()
            )
            track_mask = np.broadcast_to(
                track_mask[:, None, :], pytorch_value.shape
            ).copy()
            pytorch_value = pytorch_value[torch.from_numpy(track_mask)]
            jax_value = jax_value[track_mask]
        if name == "heads/splice_sites_junction/predictions":
            representation = "junction"
        elif (
            len(path) == 3
            and path[0] == "heads"
            and path[2].startswith("predictions_")
        ):
            representation = "descaled"
        else:
            representation = "default"
        criteria = equivalence_criteria(
            pt_dtype_policy.name,
            "full_model",
            representation=representation,
        )
        try:
            record_and_assert_close(
                pytorch_value,
                jax_value,
                dtype_policy=pt_dtype_policy.name,
                name=name,
                **criteria,
            )
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        pytest.fail("Full-model differences exceeded tolerance:\n" + "\n".join(failures))
