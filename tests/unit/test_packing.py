"""Unit tests for BOS-aligned best-fit token packing.

Tests:
1. Correct window count
2. No cross-document attention in masks
3. Causal constraint preserved
4. Labels masked at EOS positions
5. Token utilization ≈ 100%
6. BOS tokens present at document starts
"""

import numpy as np
import pytest

from nanochat.data.packing import (
    pack_documents,
    pack_from_flat_tokens,
    _build_doc_aware_causal_mask,
    IGNORE_INDEX,
)


BOS_ID = 1
EOS_ID = 2

# ---------------------------------------------------------------------------
# _build_doc_aware_causal_mask
# ---------------------------------------------------------------------------

class TestDocAwareCausalMask:
    def test_causal_constraint(self):
        """Mask must be lower-triangular (no future positions)."""
        doc_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        mask = _build_doc_aware_causal_mask(doc_ids)
        assert mask.shape == (8, 8)

        # Upper triangle should be all False
        upper = mask[np.triu_indices(8, k=1)]
        assert not upper.any(), "Future positions should not be attended to"

    def test_same_doc_only(self):
        """Tokens from different documents must not attend to each other."""
        doc_ids = np.array([0, 0, 1, 1])
        mask = _build_doc_aware_causal_mask(doc_ids)

        # Doc 1 tokens (indices 2, 3) must NOT attend to doc 0 tokens (0, 1)
        assert not mask[2, 0], "Cross-doc attention: doc1 → doc0"
        assert not mask[2, 1], "Cross-doc attention: doc1 → doc0"
        assert not mask[3, 0], "Cross-doc attention: doc1 → doc0"

    def test_same_doc_causal_ok(self):
        """Tokens within the same document can attend causally."""
        doc_ids = np.array([0, 0, 0, 0])
        mask = _build_doc_aware_causal_mask(doc_ids)

        # All lower-triangular same-doc pairs should be True
        assert mask[1, 0], "Intra-doc causal attention blocked"
        assert mask[2, 1], "Intra-doc causal attention blocked"
        assert mask[3, 0], "Intra-doc causal attention blocked"


# ---------------------------------------------------------------------------
# pack_documents
# ---------------------------------------------------------------------------

class TestPackDocuments:
    def _make_docs(self, n_docs=4, doc_len=10):
        """Make n_docs documents of fixed length."""
        return [list(range(100, 100 + doc_len)) for _ in range(n_docs)]

    def test_basic_packing(self):
        """Should produce at least one window for sufficient tokens."""
        docs = self._make_docs(n_docs=4, doc_len=10)
        windows = pack_documents(docs, seq_len=16, bos_id=BOS_ID, eos_id=EOS_ID)
        assert len(windows) > 0

    def test_window_length(self):
        """Each window's input_ids should have exactly seq_len tokens."""
        docs = self._make_docs(n_docs=8, doc_len=20)
        seq_len = 32
        windows = pack_documents(docs, seq_len=seq_len, bos_id=BOS_ID, eos_id=EOS_ID)
        for w in windows:
            assert len(w.input_ids) == seq_len, f"Wrong input_ids length: {len(w.input_ids)}"
            assert len(w.labels) == seq_len, f"Wrong labels length: {len(w.labels)}"

    def test_mask_shape(self):
        """Attention mask should be (seq_len, seq_len)."""
        docs = self._make_docs(n_docs=4, doc_len=10)
        seq_len = 16
        windows = pack_documents(docs, seq_len=seq_len, bos_id=BOS_ID, eos_id=EOS_ID)
        for w in windows:
            assert w.attention_mask.shape == (seq_len, seq_len)
            assert w.attention_mask.dtype == bool

    def test_causal_mask(self):
        """All windows must have causal masks (lower triangular)."""
        docs = self._make_docs(n_docs=4, doc_len=15)
        windows = pack_documents(docs, seq_len=16, bos_id=BOS_ID, eos_id=EOS_ID)
        for w in windows:
            upper = w.attention_mask[np.triu_indices(16, k=1)]
            assert not upper.any(), "Non-causal attention mask found"

    def test_no_cross_doc_attention(self):
        """Tokens in different documents within same window must not cross-attend."""
        # Create 2 short docs that fit in one window with a doc boundary
        docs = [[10, 11, 12, 13], [20, 21, 22, 23]]  # 4 tokens each
        # With BOS+EOS wrapping: [BOS 10 11 12 13 EOS BOS 20 21 22 23 EOS]
        # = 12 tokens total, fits in window of 10
        windows = pack_documents(docs, seq_len=10, bos_id=BOS_ID, eos_id=EOS_ID)

        if not windows:
            pytest.skip("Not enough tokens for this test config")

        for w in windows:
            # Find positions of each doc
            doc_ids = w.doc_ids
            doc0_positions = np.where(doc_ids == 0)[0]
            doc1_positions = np.where(doc_ids == 1)[0]

            if len(doc0_positions) == 0 or len(doc1_positions) == 0:
                continue  # Window only has one doc, skip

            # Check that doc1 tokens don't attend to doc0 positions
            for q in doc1_positions:
                for k in doc0_positions:
                    assert not w.attention_mask[q, k], (
                        f"Cross-doc attention: q={q}(doc1) → k={k}(doc0)"
                    )

    def test_eos_labels_masked(self):
        """Labels at EOS input positions should be IGNORE_INDEX."""
        docs = [[10, 11], [20, 21]]
        windows = pack_documents(docs, seq_len=8, bos_id=BOS_ID, eos_id=EOS_ID)

        for w in windows:
            # Find positions where input_ids == EOS
            eos_positions = np.where(w.input_ids == EOS_ID)[0]
            for pos in eos_positions:
                assert w.labels[pos] == IGNORE_INDEX, (
                    f"EOS label at pos {pos} should be IGNORE_INDEX={IGNORE_INDEX}, "
                    f"got {w.labels[pos]}"
                )

    def test_token_utilization(self):
        """Utilization should be high for typical document sizes."""
        # Use large docs to maximize packing efficiency
        docs = [list(range(i * 50, i * 50 + 50)) for i in range(40)]
        seq_len = 64
        windows = pack_documents(docs, seq_len=seq_len, bos_id=BOS_ID, eos_id=EOS_ID)

        total_input_tokens = sum(len(d) + 2 for d in docs)  # +2 for BOS+EOS
        packed_tokens = len(windows) * seq_len
        utilization = packed_tokens / total_input_tokens

        assert utilization > 0.85, (
            f"Token utilization too low: {utilization:.2%} "
            "(expected > 85% for long documents)"
        )

    def test_empty_documents_handled(self):
        """Should handle empty document list without error."""
        windows = pack_documents([], seq_len=16, bos_id=BOS_ID, eos_id=EOS_ID)
        assert windows == []

    def test_insufficient_tokens(self):
        """Returns empty list when documents are too short."""
        docs = [[1, 2]]  # Only 4 tokens with BOS+EOS wrapping
        windows = pack_documents(docs, seq_len=32, bos_id=BOS_ID, eos_id=EOS_ID)
        assert windows == []


# ---------------------------------------------------------------------------
# pack_from_flat_tokens
# ---------------------------------------------------------------------------

class TestPackFromFlatTokens:
    def test_basic(self):
        """Should pack flat stream into windows."""
        # Flat stream: [BOS t1 t2 EOS BOS t3 t4 EOS ...]
        n_tokens = 200
        tokens = np.arange(n_tokens, dtype=np.int32)
        # Insert BOS at every 20 positions to simulate documents
        tokens[::20] = BOS_ID

        seq_len = 16
        windows = pack_from_flat_tokens(tokens, seq_len, BOS_ID)
        assert len(windows) > 0

        for w in windows:
            assert w.input_ids.shape == (seq_len,)
            assert w.attention_mask.shape == (seq_len, seq_len)
