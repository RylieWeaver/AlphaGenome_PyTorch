# Adapted from genomicsxai/alphagenome-pytorch,
# tests/integration/test_determinism.py (Apache-2.0).
# No golden numbers. Every assertion compares the model against itself.
#
# Deviations from the source file:
#   - genomicsxai/alphagenome-pytorch builds the full model; this uses
#     small_alphagenome with its default dtype policy instead
#   - genomicsxai/alphagenome-pytorch calls model(onehot, organism_index); this
#     implementation accepts DataBatch and exposes embeddings through model.embed

import pytest
import torch

from .helpers import build_with_batch

pytestmark = pytest.mark.integration


def _embeddings(model, batch):
    with torch.no_grad():
        emb = model.embed(batch)
    return [emb.embeddings_1bp, emb.embeddings_128bp, emb.embeddings_pair]


def _assert_identical(a, b, msg):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0, msg=msg)


class TestDeterminism:
    def test_eval_mode_is_bitwise_reproducible(self):
        model, batch = build_with_batch()
        model.eval()
        _assert_identical(_embeddings(model, batch), _embeddings(model, batch),
                          "same input twice gave different outputs")

    def test_same_seed_gives_same_model(self):
        torch.manual_seed(123)
        m1, batch = build_with_batch()
        torch.manual_seed(123)
        m2, _ = build_with_batch()
        m1.eval(); m2.eval()
        _assert_identical(_embeddings(m1, batch), _embeddings(m2, batch),
                          "same seed gave different models")

    def test_different_seed_gives_different_model(self):
        # Without this, a model that ignored its weights would pass the tests
        # above perfectly.
        torch.manual_seed(1)
        m1, batch = build_with_batch()
        torch.manual_seed(2)
        m2, _ = build_with_batch()
        m1.eval(); m2.eval()
        a = _embeddings(m1, batch)[0]
        b = _embeddings(m2, batch)[0]
        assert not torch.allclose(a, b), "different seeds gave identical outputs"

    def test_different_input_gives_different_output(self):
        # And without this, a model that ignored its input would also pass.
        model, batch_a = build_with_batch()
        _, batch_b = build_with_batch(batch_size=2)
        model.eval()
        a = _embeddings(model, batch_a)[0]
        b = _embeddings(model, batch_b)[0]
        assert not torch.allclose(a, b), "different inputs gave identical outputs"
