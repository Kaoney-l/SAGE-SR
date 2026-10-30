import math
import pytest
from sage_sr.feedback import AlphaCalibrator, deduplicate_by_root, _subtrees


def test_clipped_trust_and_permanent_conflict_shutdown():
    c = AlphaCalibrator(0.5)
    assert c.update([1.0], [0.0]) == pytest.approx(math.e)
    assert c.update([1.0], [0.0]) == 3.0
    assert c.history[-1]["evidence"] == 1.0
    assert c.update([0.0], [1.0]) == pytest.approx(3 / math.e)
    assert c.update([0.0], [1.0]) == 0.0
    assert c.update([1.0], [0.0]) == 0.0


def test_missing_group_resets_consecutive_conflicts():
    c = AlphaCalibrator()
    c.update([0.1], [0.9])
    before = c.alpha
    c.update([], [0.9])
    assert c.alpha == before and c.conflicts == 0
    c.update([0.1], [0.9])
    assert c.alpha > 0 and c.conflicts == 1


def test_siblings_do_not_multiply_ancestry_support():
    candidates = [{"root_ids": ["ancestor"], "reward": r, "validation_reward": r, "path": [i]}
                  for i, r in enumerate([0.5, 0.7, 0.6])]
    unique = deduplicate_by_root(candidates)
    assert len(unique) == 1 and unique[0]["reward"] == 0.7


def test_motifs_are_complete_trees():
    motifs = _subtrees(["+", "x0", "*", "x0", "x0"])
    assert ("*", "x0", "x0") in motifs
    assert ("+", "x0") not in motifs
