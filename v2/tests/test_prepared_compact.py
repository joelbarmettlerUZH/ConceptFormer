"""Regression: Prepared stores teacher contexts compactly without changing consumer semantics.

The 100k-corpus runs were kernel-OOM-killed because ~915k Prepared rows held their ~1.1k-token
teacher contexts as Python int lists (~36 B/token); trainer.prepare now stores array('i'). The
consumers use splat-concatenation and len(), which must behave identically for both storages.
"""

from array import array

from conceptformer.train.trainer import Prepared


def test_prepared_accepts_array_ctx_and_consumer_patterns_hold():
    ctx = array("i", [101, 102, 103])
    p = Prepared(
        qid="Q1", teacher_ctx_ids=ctx, student_head_ids=[1], student_tail_ids=[2],
        path=[7, 8],
    )
    # The two consumer patterns in trainer.py: splat-concat with the (list) path, and len().
    assert [*p.teacher_ctx_ids, *p.path] == [101, 102, 103, 7, 8]
    assert len(p.teacher_ctx_ids) == 3
    # 4 bytes per token: the property the OOM fix depends on.
    assert ctx.itemsize == 4
