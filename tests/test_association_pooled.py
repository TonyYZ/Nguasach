"""Unit test for the cross-linguistic pooled phoneme x pole test."""

import numpy as np

from nguasach.association import _pooled_test


def _stash_entry(lang, z_pt, iters, rng, n_poles=3, vocab=("p", "t", "k")):
    V = len(vocab)
    z = np.zeros((n_poles, V))
    z[0, 0] = z_pt                                   # plant the signal at pole0 x /p/
    null_z = rng.normal(0, 1, size=(iters, n_poles, V))
    counts = np.full((n_poles, V), 50.0)
    return {"lang": lang, "z": z, "null_z": null_z,
            "counts": counts, "vocab": list(vocab)}


def test_pooled_flags_consistent_cross_family_bias():
    rng = np.random.default_rng(0)
    iters = 200
    # eight languages, eight distinct macro-families
    langs = ["English", "Finnish", "Turkish", "Chinese",
             "Japanese", "Korean", "Thai", "Swahili"]
    stash = [_stash_entry(l, 2.0, iters, rng) for l in langs]
    names = ["A", "B", "C"]
    out = _pooled_test(stash, names, {"A": "", "B": "", "C": ""},
                       iters, seed=0, min_langs=5, min_count=3)

    planted = [c for c in out["cells"] if c["pole"] == "A" and c["phoneme"] == "p"]
    assert planted and planted[0]["significant"]
    assert planted[0]["n_families"] >= 5
    assert planted[0]["family_sign_concord"] == 1.0
    # the noise cells must not be flagged
    assert all(not c["significant"] for c in out["cells"]
               if not (c["pole"] == "A" and c["phoneme"] == "p"))


def test_pooled_rejects_single_family_signal():
    """A bias carried only by Indo-European (many related languages) must fail
    the >=4-family robustness guard even if it clears the raw permutation p."""
    rng = np.random.default_rng(1)
    iters = 200
    ie = ["English", "German", "French", "Italian", "Spanish", "Russian"]
    stash = [_stash_entry(l, 3.0, iters, rng) for l in ie]
    out = _pooled_test(stash, ["A", "B", "C"], {}, iters, seed=0,
                       min_langs=5, min_count=3)
    planted = [c for c in out["cells"] if c["pole"] == "A" and c["phoneme"] == "p"]
    assert planted
    assert planted[0]["n_families"] == 1
    assert not planted[0]["significant"]        # fails family guard
