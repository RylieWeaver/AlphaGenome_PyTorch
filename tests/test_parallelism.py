"""FP64 parallelism regression tests on four GPUs.

Run: torchrun --standalone --nproc-per-node=4 --module tests.test_parallelism
On failure, tests/conftest.py exits the worker so torchrun stops its peers.
"""

import copy
import os
import sys
from contextlib import nullcontext
from dataclasses import fields, replace
from types import SimpleNamespace

# os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

from alphagenome_pt import (
    AlphaGenome,
    DataBatch,
    HeadName,
    MetricTree,
    ParallelContext,
    distribute_alphagenome,
    losses,
    small_alphagenome,
    sp_broadcast_data_batch,
    synthetic_batch,
    synthetic_metadata,
)
from alphagenome_pt.distributed import _prepare_fsdp_kwargs, dist_sum
from alphagenome_pt.precision import dtype_policy_context, get_dtype_policy


### HELPERS ###

@pytest.fixture(scope="module")
def init_parallelism(precision_backend):
    """Initialize NCCL and a 2x2 mesh, yielding the device and parallel context."""
    if "RANK" not in os.environ:
        pytest.skip("Run with torchrun --nproc-per-node=4.")

    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() < 4:
        pytest.skip("Parallelism tests require exactly four processes and four GPUs.")

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", device_id=device)

    try:
        parallel_context = ParallelContext(
            init_device_mesh(
                "cuda",
                (2, 2),
                mesh_dim_names=("dp", "sp"),
            )
        )
        yield device, parallel_context
    finally:
        if initialized_here:
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def precision_backend():
    """Use FP64 policy."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    conv_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        with dtype_policy_context(get_dtype_policy("float64"), "cuda"):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = conv_tf32


@pytest.fixture(scope="module")
def fsdp_mesh(init_parallelism):
    _, parallel_context = init_parallelism
    # Shard across SP, replicate across DP
    return DeviceMesh(
        "cuda", parallel_context.mesh.mesh,
        mesh_dim_names=("replicate", "shard"),
    )


@pytest.fixture(params=[
    pytest.param((None, False), id="sp_ddp"),
    pytest.param((ShardingStrategy.FULL_SHARD, False), id="sp_fsdp"),
    pytest.param((ShardingStrategy.HYBRID_SHARD, False), id="sp_fsdp_hybrid"),
    pytest.param((ShardingStrategy._HYBRID_SHARD_ZERO2, False), id="sp_fsdp_hybrid_zero2"),
    pytest.param((ShardingStrategy.HYBRID_SHARD, True), id="sp_fsdp_hybrid_native"),
    pytest.param((ShardingStrategy._HYBRID_SHARD_ZERO2, True), id="sp_fsdp_hybrid_zero2_native"),
])
def parallel_options(request, init_parallelism, fsdp_mesh):
    """Reuse the same comparisons for DDP and each FSDP sharding strategy."""
    device, parallel_context = init_parallelism
    sharding, independent_mesh = request.param
    options = {"mesh": parallel_context.mesh, "sync_bn": True}
    if sharding is None:
        options.update(strategy="sp_ddp", device_ids=[device.index])
    else:
        options.update(
            strategy="sp_fsdp", device_id=device, sharding_strategy=sharding,
        )
        if independent_mesh:
            options["device_mesh"] = fsdp_mesh
    return options


def _model(device, *, mlm_only=False, **kwargs):
    torch.manual_seed(42)

    if mlm_only:
        heads = (HeadName.MASKED_LANGUAGE_MODELING,)
    else:
        heads = tuple(
            head
            for head in HeadName
            if head != HeadName.MASKED_LANGUAGE_MODELING
        )

    # NOTE: K must be a power of two. Changing it affects the proportion
    # of BN overlap, which can change equivalence --> test success/failure.
    K = 128
    return small_alphagenome(
        synthetic_metadata(heads),
        max_seq_len=2048*K,
        num_channels=24,
        transformer_layers=2,
        num_splice_sites=K,
        sync_bn=True,
        dropout=0.0,
        dtype_policy="float64",
        **kwargs,
    ).to(device)


def _batch(model, sample_index, device, *, seq_len=None):
    """Build a DP example with unequal target counts and deliberately missing labels."""
    torch.manual_seed(42)
    batch = synthetic_batch(
        model.metadata,
        batch_size=1,
        seq_len=model.max_seq_len if seq_len is None else seq_len,
        num_splice_sites=4,
    )
    batch.organism_index.fill_(sample_index % 2)

    if batch.mlm is not None:
        # -100 means ignore this token. Keep 1 valid token in DP example 0
        # and 2 in example 1, all in SP rank 0's slice; SP rank 1 has none.
        num_valid_tokens = sample_index + 1
        batch.mlm.fill_(-100)
        batch.mlm[:, :num_valid_tokens] = torch.arange(num_valid_tokens).remainder(5)

    if batch.splice_site_positions is not None:
        # Four donor slots split 2 + 2 across SP ranks. Pad the last two
        # slots (-1) in all four site sets, so SP rank 1 has no valid donor
        # rows contributing to the junction loss.
        batch.splice_site_positions[..., -2:] = -1

    if batch.contact_maps is not None:
        # A missing off-diagonal target, in every track, must be ignored
        # by the contact-map loss rather than make the loss NaN.
        batch.contact_maps[:, 0, 1] = float("nan")

    if batch.atac_mask is not None:
        # Track 0 is observed in DP example 0 but missing in example 1,
        # testing loss weighting when valid counts differ between DP ranks.
        batch.atac_mask[..., 0] = sample_index % 2 == 0

    if batch.splice_sites is not None:
        # An all-zero class vector means no label, so the classification
        # loss must ignore every other sequence position.
        batch.splice_sites[:, ::2] = 0

    return batch.to(device)


def _combine_batches(batches):
    """Combine equal-length synthetic examples along the batch dimension."""
    tensor_fields = {}
    for field in fields(batches[0]):
        values = [getattr(batch, field.name) for batch in batches]
        if isinstance(values[0], torch.Tensor):
            tensor_fields[field.name] = torch.cat(values, dim=0)
    return replace(batches[0], **tensor_fields)


def _assert_close(actual, expected, name, *, relative_l2):
    """Compare with the relative L2 metric used by the DeepMind equivalence tests."""
    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    assert actual.dtype == expected.dtype, f"{name}: dtype mismatch"
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Discrete outputs, such as splice-site positions, must match exactly.
    if not expected.is_floating_point():
        assert torch.equal(actual, expected), f"{name}: discrete values differ"
        return

    # Detach to avoid tracking comparison gradients.
    actual = actual.detach()
    expected = expected.detach()
    difference_norm = torch.linalg.vector_norm(actual - expected).item()
    reference_norm = torch.linalg.vector_norm(expected).item()

    if reference_norm > torch.finfo(torch.float64).tiny:
        error = difference_norm / reference_norm
    else:
        # An all-zero reference matches only an all-zero result.
        error = 0.0 if difference_norm == 0 else float("inf")

    message = (
        f"[rank {rank}] {name}: relative_L2={error:.6g} "
        f"(limit={relative_l2:.6g})"
    )

    assert error <= relative_l2, message


def _assert_outputs(
    actual,
    expected,
    parallel_context,
    *,
    relative_l2,
):
    predictions = actual.predictions
    embeddings = actual.embeddings

    # Each DP replica owns one example from the full reference batch.
    dp_slice = slice(parallel_context.dp_mesh_index, parallel_context.dp_mesh_index + 1)

    assert predictions.keys() == expected.predictions.keys()

    for head, head_predictions in expected.predictions.items():
        assert predictions[head].keys() == head_predictions.keys()

        for name, value in head_predictions.items():
            # Candidate positions are replicated within SP. Other tensors shard
            # on sequence, contact-map rows, or junction donor rows (dimension 1).
            reference = value[dp_slice]
            if name != "splice_site_positions":
                reference = parallel_context.sp_shard(reference, dim=1)
            _assert_close(
                predictions[head][name],
                reference,
                f"{head}/{name}",
                relative_l2=relative_l2,
            )

    for field in fields(expected.embeddings):
        _assert_close(
            getattr(embeddings, field.name),
            parallel_context.sp_shard(
                getattr(expected.embeddings, field.name)[dp_slice], dim=1,
            ),
            field.name,
            relative_l2=relative_l2,
        )

    actual_leaves = dict(actual.tree.iter_leaves())
    expected_leaves = dict(expected.tree.iter_leaves())
    assert actual_leaves.keys() == expected_leaves.keys()

    for path, leaf in actual_leaves.items():
        _assert_close(
            leaf.value,
            expected_leaves[path].value,
            "/".join(path),
            relative_l2=relative_l2,
        )

    mlm_path = ("masked_language_modeling", "cross_entropy")
    if mlm_path in expected_leaves:
        _assert_close(
            actual_leaves[mlm_path].denominator,
            expected_leaves[mlm_path].denominator,
            "global MLM count",
            relative_l2=relative_l2,
        )

    _assert_close(actual.total, expected.total, "total loss", relative_l2=relative_l2)


def _assert_gradients(wrapped, reference, *, relative_l2):
    # FSDP gradients must be materialized before comparing original parameter shapes.
    full_params = (
        FSDP.summon_full_params(wrapped, with_grads=True, writeback=False)
        if isinstance(wrapped, FSDP)
        else nullcontext()
    )

    with full_params:
        actual = dict(wrapped.module.named_parameters())
        assert actual.keys() == dict(reference.named_parameters()).keys()

        for name, parameter in reference.named_parameters():
            gradient = actual[name].grad

            if parameter.grad is None:
                assert gradient is None or torch.count_nonzero(gradient) == 0, name
            else:
                assert gradient is not None, name
                _assert_close(
                    gradient,
                    parameter.grad,
                    f"gradient/{name}",
                    relative_l2=relative_l2,
                )


### FULL-MODEL OUTPUTS, LOSSES, AND GRADIENTS ###
# NOTE: The BatchNorm with overlap segments causes differences
# NOTE: The huge difference between train and eval equivalence. 
# This is largely because the batchnorm is frozen in .eval() mode.
@pytest.mark.parametrize(
    "mlm_only",
    [True, False],
    ids=["mlm_only", "other_heads"],
)
@pytest.mark.parametrize(
    ("training", "output_relative_l2", "gradient_relative_l2"),
    [
        (True, 1e-2, 1e-1),
        (False, 1e-12, 1e-11),
    ],
    ids=["train", "eval"],
)
def test_full_model_equivalence(
    init_parallelism,
    parallel_options,
    training,
    mlm_only,
    output_relative_l2,
    gradient_relative_l2,
):
    device, parallel_context = init_parallelism

    ref_model = _model(device, mlm_only=mlm_only).train(training)
    # The complete batch supplies reference BN statistics on this device.
    ref_model.set_sync_bn(False)
    parallel_model = distribute_alphagenome(
        copy.deepcopy(ref_model), **parallel_options,
    )
    parallel_context = parallel_model.parallel_context

    assert (parallel_context.dp_size, parallel_context.sp_size) == (2, 2)
    assert parallel_context.dp_enabled and parallel_context.sp_enabled
    assert parallel_context.loss_group is dist.group.WORLD
    native_mesh = parallel_options.get("device_mesh")
    expected_group = dist.group.WORLD if native_mesh is None else native_mesh.get_group(1)
    assert parallel_model.process_group is expected_group
    if isinstance(parallel_model, FSDP):
        expected_sharding = (
            ShardingStrategy.FULL_SHARD if native_mesh is None
            else parallel_options["sharding_strategy"]
        )
        assert parallel_model.sharding_strategy == expected_sharding
    assert not hasattr(parallel_model.module, "parallel_context")

    # Data
    dp_batches = [
        _batch(ref_model, sample_index, device)
        for sample_index in range(parallel_context.dp_size)
    ]
    ref_batch = _combine_batches(dp_batches)
    dp_batch = dp_batches[parallel_context.dp_mesh_index]
    parallel_batch = sp_broadcast_data_batch(
        dp_batch if parallel_context.sp_mesh_index == 0 else None,
        parallel_context=parallel_context,
    )

    # Check Outputs
    ref_outputs = ref_model.loss(
        ref_batch,
        return_predictions=True,
        return_embeddings=True,
    )
    parallel_outputs = parallel_model.loss(
        parallel_batch,
        return_predictions=True,
        return_embeddings=True,
    )
    _assert_outputs(
        parallel_outputs,
        ref_outputs,
        parallel_context,
        relative_l2=output_relative_l2,
    )

    # Check gradients
    ref_outputs.total.backward()
    parallel_outputs.total.backward()
    _assert_gradients(
        parallel_model, ref_model, relative_l2=gradient_relative_l2
    )


@pytest.mark.parametrize("sharding", (
    ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2,
))
@pytest.mark.parametrize(
    ("sequence_shape", "sharding_shape"),
    [
        (None, (2, 2)),
        ((1, 4), (2, 2)),
        ((2, 2), (1, 4)),
        ((2, 2), (4, 1)),
    ],
    ids=[
        "no_sp", "shard_within_sp", "shard_across_sp_groups", "sp_single_shard_fallback",
    ],
)
def test_independent_fsdp_mesh(init_parallelism, sharding, sequence_shape, sharding_shape):
    """Compare full-model losses and gradients when SP and FSDP sizes differ."""
    device, _ = init_parallelism
    sequence_mesh = None if sequence_shape is None else init_device_mesh(
        "cuda", sequence_shape, mesh_dim_names=("dp", "sp"),
    )
    fsdp_mesh = init_device_mesh(
        "cuda", sharding_shape, mesh_dim_names=("replicate", "shard"),
    )
    ref_model = _model(device, mlm_only=True).eval()
    parallel_model = distribute_alphagenome(
        copy.deepcopy(ref_model),
        strategy="fsdp" if sequence_mesh is None else "sp_fsdp",
        mesh=sequence_mesh, device_mesh=fsdp_mesh,
        sharding_strategy=sharding, device_id=device,
    )
    context = parallel_model.parallel_context
    assert context.sp_size == (1 if sequence_shape is None else sequence_shape[1])
    assert context.dp_size == (dist.get_world_size() if sequence_shape is None else sequence_shape[0])
    assert context.loss_group is dist.group.WORLD
    full_shard = fsdp_mesh.size(1) == 1
    expected_group = dist.group.WORLD if full_shard else fsdp_mesh.get_group(1)
    assert parallel_model.process_group is expected_group
    expected_sharding = ShardingStrategy.FULL_SHARD if full_shard else sharding
    assert parallel_model.sharding_strategy == expected_sharding

    batches = [_batch(ref_model, i, device, seq_len=8192) for i in range(context.dp_size)]
    expected = ref_model.loss(
        _combine_batches(batches), return_predictions=True, return_embeddings=True,
    )
    actual = parallel_model.loss(
        batches[context.dp_mesh_index], return_predictions=True, return_embeddings=True,
    )
    _assert_outputs(actual, expected, context, relative_l2=1e-12)
    expected.total.backward()
    actual.total.backward()
    _assert_gradients(parallel_model, ref_model, relative_l2=1e-11)


### DIFFERENT SEQUENCE LENGTHS ACROSS DP REPLICAS ###

# NOTE: This test can only be done in eval mode since a rank's batch must
# have equal sequence length across samples and we can't do padding since
# that would affect the BatchNorm.
@pytest.mark.parametrize("mlm_only", [True, False], ids=["mlm_only", "other_heads"])
def test_different_sequence_lengths_equivalence(init_parallelism, parallel_options, mlm_only):
    """Compare losses and gradients with BN frozen for separate reference forwards."""
    device, parallel_context = init_parallelism

    ref_model = _model(device, mlm_only=mlm_only).eval()
    parallel_model = distribute_alphagenome(
        copy.deepcopy(ref_model), **parallel_options,
    )

    batches = [
        _batch(ref_model, i, device, seq_len=ref_model.max_seq_len // (i + 1))
        for i in range(parallel_context.dp_size)
    ]

    # For a single rank, run the batches in a loop to allow for
    # different sequence lengths
    ref_tree = ref_model.loss(batches[0]).tree
    for ref_batch in batches[1:]:
        ref_tree = ref_tree.add(ref_model.loss(ref_batch).tree, detach=False)
    parallel_output = parallel_model.loss(batches[parallel_context.dp_mesh_index])

    # For DP training, the different sequence length batches are
    # already split across ranks
    ref_leaves = dict(ref_tree.iter_leaves())
    parallel_leaves = dict(parallel_output.tree.iter_leaves())
    assert parallel_leaves.keys() == ref_leaves.keys()
    for path, leaf in parallel_leaves.items():
        _assert_close(
            leaf.value, ref_leaves[path].value, "/".join(path),
            relative_l2=1e-12,
        )

    ref_tree.total_loss().backward()
    parallel_output.total.backward()
    _assert_gradients(parallel_model, ref_model, relative_l2=1e-11)


### CHECKPOINT###

def test_parallel_checkpoint_round_trip(
    init_parallelism,
    tmp_path,
    parallel_options,
):
    device, parallel_context = init_parallelism

    model = _model(device, mlm_only=True).eval()
    expected = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    wrapped = distribute_alphagenome(model, **parallel_options)

    paths = [str(tmp_path / "model") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths)
    wrapped.save(paths[0])  # All ranks must participate in FSDP parameter gathering.

    if dist.get_rank() == 0:
        restored = AlphaGenome.load(paths[0], device="cpu")
        assert not hasattr(restored, "parallel_context")
        torch.testing.assert_close(restored.state_dict(), expected)

    dist.barrier()


### CONFIG REJECTION ###

@pytest.mark.parametrize(
    "widths",
    [
        {"first_conv_width": 13},
        {"block_width": 3},
    ],
)
def test_sp_rejects_nondefault_conv_widths(init_parallelism, widths):
    device, parallel_context = init_parallelism

    model = _model(device, mlm_only=True, **widths).eval()
    batch = _batch(model, 0, device)

    with pytest.raises(
        ValueError,
        match="requires first_conv_width=15 and block_width=5",
    ):
        model(batch, mode="embed", parallel_context=parallel_context)


@pytest.mark.parametrize("sharding", (
    ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2,
))
def test_fsdp_rejects_invalid_hybrid_groups(monkeypatch, sharding):
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)

    def mesh(ranks):
        ranks = torch.tensor(ranks)
        return SimpleNamespace(mesh=ranks, ndim=ranks.ndim, size=ranks.size)

    for options, message in (
        ({"process_group": object()}, "Leave process_group"),
        ({"device_mesh": mesh([0, 1, 2, 3])}, "must be 2D"),
        ({"device_mesh": mesh([[0, 1]])}, "every WORLD rank"),
        ({"device_mesh": mesh([[1, 0], [2, 3]])}, "rank order"),
        ({"device_mesh": mesh([[0, 1], [2, 3]]), "process_group": object()},
         "Leave process_group"),
    ):
        with pytest.raises(ValueError, match=message):
            _prepare_fsdp_kwargs(sharding_strategy=sharding, **options)


### MULTINOMIAL WINDOW REGRESSION ###

@pytest.mark.parametrize(
    ("sp_size", "resolution"),
    [(2, 3), (2, 6), (2, 12), (2, 24), (4, 12)],
)
def test_multinomial_windows(init_parallelism, sp_size, resolution):
    device, parallel_context = init_parallelism
    if sp_size == 4:
        parallel_context = ParallelContext(
            init_device_mesh("cuda", (1, 4), mesh_dim_names=("dp", "sp"))
        )
    group = parallel_context.sp_group

    # SP=2 covers local/full windows. SP=4 has six bins per rank, so R=12
    # sums within two separate rank pairs: {0, 1} and {2, 3}.
    full = (
        torch.linspace(0.2, 2.0, 48, device=device, dtype=torch.float64)
        .reshape(1, 24, 2)
        .requires_grad_()
    )
    local = parallel_context.sp_shard(full.detach(), dim=1).clone().requires_grad_()
    mask = torch.tensor([[[True, False]]], device=device)

    def window_loss(prediction, resolution, group=None):
        result = losses.multinomial_loss(
            y_pred=prediction,
            y_true=prediction.detach().square(),
            mask=mask,
            multinomial_resolution=resolution,
            positional_weight=1.0,
            sequence_group=group,
        )
        tree = MetricTree({
            "count": result["loss_total"],
            "positional": result["zero_loss_positional"],
        })
        if group is not None:
            tree = tree.distributed_reduce(group)
        return tree.total_loss()

    expected = window_loss(full, resolution)
    actual = window_loss(local, resolution, group)
    _assert_close(actual, expected, "window loss", relative_l2=1e-10)

    expected.backward()
    actual.backward()
    # Match the gradient averaging that DDP/FSDP normally supplies.
    _assert_close(
        local.grad / parallel_context.sp_size,
        parallel_context.sp_shard(full.grad, dim=1),
        "window gradient",
        relative_l2=1e-10,
    )

    for invalid in (0, -1, 5, 8, 25):
        with pytest.raises(ValueError):
            window_loss(local, invalid, group)


### SPLICE-JUNCTION PREDICTIONS ###

@pytest.mark.parametrize("supplied_positions", [True, False], ids=["supplied", "automatic"])
def test_splice_junction_predictions(init_parallelism, supplied_positions):
    device, parallel_context = init_parallelism

    # Freeze BN so its overlap approximation cannot change discrete top-k choices.
    model = _model(device).eval()
    model.num_splice_sites = 4
    batch = _batch(model, parallel_context.dp_mesh_index, device)
    if not supplied_positions:
        batch = replace(batch, splice_site_positions=None)

    head = HeadName.SPLICE_SITES_JUNCTION.value
    with torch.no_grad():
        expected = model(batch, mode="predict")[head]
        actual = model(batch, mode="predict", parallel_context=parallel_context)[head]

    for name, value in expected.items():
        # Positions are shared by SP peers; counts and masks own donor rows.
        if name != "splice_site_positions":
            value = parallel_context.sp_shard(value, dim=1)
        _assert_close(actual[name], value, f"junctions/{name}", relative_l2=1e-12)


### OUTPUT-HANDLING REGRESSIONS ###

# NOTE: FSDP must reach leaf numerators to attach its pre-backward hooks.
# DDP must also wrap returned tensors in _DDPSink so ignored outputs do not
# leave its reducer waiting for parameter gradients on the next forward.
@pytest.mark.parametrize(
    ("loss_source", "return_outputs"),
    [
        ("total", False),          # Ordinary .loss().total.backward().
        ("total", True),           # Returned pair embeddings are unused by MLM.
        ("tree", False),           # Recompute without other hooked outputs.
        ("tree", True),            # Recompute with unused predictions/embeddings.
        ("leaf", True),            # Backward directly through a nested LossLeaf.
        ("loss_embeddings", True), # Custom loss from .loss()'s embeddings.
        ("embed", False),          # Custom loss from .embed(); pair output unused.
    ],
)
def test_output_container_backward(
    init_parallelism, parallel_options, loss_source, return_outputs,
):
    device, parallel_context = init_parallelism

    ref_model = _model(device, mlm_only=True).eval()
    parallel_model = distribute_alphagenome(
        copy.deepcopy(ref_model), **parallel_options,
    )
    # Both DP replicas use the same example, matching the reference mean loss.
    batch = _batch(ref_model, 0, device)

    def forward_loss(model):
        if loss_source == "embed":
            embeddings = model.embed(batch)
        else:
            output = model.loss(
                batch,
                return_predictions=return_outputs,
                return_embeddings=return_outputs,
            )
            if loss_source == "total":
                return output.total
            if loss_source == "tree":
                return output.tree.total_loss()
            if loss_source == "leaf":
                mlm_losses = output.tree.children["masked_language_modeling"]
                return mlm_losses["cross_entropy"].value
            embeddings = output.embeddings

        return embeddings.embeddings_1bp.square().mean()

    expected = forward_loss(ref_model)
    expected.backward()

    # Repeat to catch unfinished DDP reduction. Check every parameter gradient
    # as well as the value: a successful backward alone could hide missing sync.
    for _ in range(2):
        parallel_model.zero_grad(set_to_none=True)
        actual = forward_loss(parallel_model)
        if loss_source in ("loss_embeddings", "embed"):
            # This custom loss averages local embeddings. Equal SP shard sizes
            # let us average these means over WORLD to match the full sequence.
            group = parallel_model.parallel_context.loss_group
            actual = dist_sum(actual, group=group) / dist.get_world_size(group)

        _assert_close(actual, expected, "output loss", relative_l2=1e-12)
        actual.backward()
        _assert_gradients(parallel_model, ref_model, relative_l2=1e-11)


@pytest.mark.parametrize(
    ("ranks", "names"),
    [
        ([[0, 3], [2, 1]], ("dp", "sp")),
        ([[0, 2], [3, 1]], ("sp", "dp")),
        ([2, 0, 3, 1], ("sp",)),
    ],
    ids=["reordered", "transposed", "sp_only"],
)
def test_mesh_indices_and_broadcast(init_parallelism, ranks, names):
    """Select data by mesh index and broadcast it from SP mesh index zero."""
    device, _ = init_parallelism
    context = ParallelContext(DeviceMesh(device.type, ranks, mesh_dim_names=names))
    rank = dist.get_rank()
    expected_dp = (0, 1, 1, 0)[rank] if "dp" in names else 0
    expected_sp = (0, 1, 0, 1)[rank] if "dp" in names else (1, 3, 0, 2)[rank]
    assert context.dp_mesh_index == expected_dp
    assert context.sp_mesh_index == expected_sp
    assert ParallelContext().dp_mesh_index == ParallelContext().sp_mesh_index == 0
    assert ParallelContext(loss_group=dist.group.WORLD).dp_mesh_index == rank

    expected = DataBatch(
        dna_sequence=["ACGT" if expected_dp == 0 else "TGCA"],
        organism_index=torch.tensor([expected_dp], device=device),
    )
    actual = sp_broadcast_data_batch(
        expected if context.sp_mesh_index == 0 else None,
        parallel_context=context,
    )
    assert actual.dna_sequence == expected.dna_sequence
    assert torch.equal(actual.organism_index, expected.organism_index)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
