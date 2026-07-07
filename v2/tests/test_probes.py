"""Report-shape tests for the causal-probe summaries (eval/probes.py)."""

import pytest

from conceptformer.eval.probes import capability_summary, faithfulness_summary


def test_faithfulness_summary_counts_and_base_unknown_split():
    s = faithfulness_summary(
        baseline_correct=[True, True, True, False],
        base_known=[True, False, False, False],
        # swap subset (3 baseline-correct probes): follow/stick/known aligned.
        swap_follow=[True, False, True],
        swap_stick=[False, True, False],
        swap_base_known=[True, False, False],
        ablate_answer_correct=[False, False, True],
        ablate_other_correct=[True, True],
    )
    assert s["n_probes"] == 4
    assert s["baseline_correct"]["correct"] == 3
    assert s["swap_follow"]["n"] == 3
    assert s["swap_follow"]["correct"] == 2
    assert s["swap_follow"]["acc"] == pytest.approx(2 / 3, abs=1e-4)
    # base-UNKNOWN split keeps only the two rows where the frozen LLM lacked the answer.
    assert s["swap_follow_base_unknown"]["n"] == 2
    assert s["swap_follow_base_unknown"]["correct"] == 1
    assert s["swap_stick_base_unknown"]["correct"] == 1
    assert s["ablate_answer_correct"]["acc"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["ablate_other_correct"]["acc"] == 1.0


def test_faithfulness_summary_rejects_misaligned_swap_lists():
    with pytest.raises(ValueError, match="aligned"):
        faithfulness_summary([True], [True], [True, False], [False], [True],
                             [], [])


def test_faithfulness_summary_empty_subsets_do_not_crash():
    s = faithfulness_summary([False], [False], [], [], [], [], [])
    assert s["swap_follow"]["n"] == 0
    assert s["swap_follow_base_unknown"]["n"] == 0
    assert s["ablate_answer_correct"]["acc"] == 0.0


def test_capability_summary():
    s = capability_summary([True, True, False, True], [0.01, 0.05, 0.4, 0.02])
    assert s["n_controls"] == 4
    assert s["greedy_agreement"]["correct"] == 3
    assert s["kl"]["mean"] == pytest.approx(0.12, abs=1e-4)
    assert s["kl"]["median"] == pytest.approx(0.035, abs=1e-4)
    assert s["kl"]["max"] == 0.4
    with pytest.raises(ValueError, match="equal length"):
        capability_summary([True], [0.1, 0.2])
