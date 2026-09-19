from rebuild.assignment_guard import solve


def test_competing_rows_are_one_to_one():
    sets = [
        {1: {"score": 0.95}, 2: {"score": 0.90}},
        {1: {"score": 0.94}, 2: {"score": 0.70}},
    ]
    chosen = solve(sets, lambda row, second: row["score"] >= 0.80 and row["score"] - second >= 0.01, 0.35)
    assert chosen == {0: 2, 1: 1}
    assert len(set(chosen.values())) == len(chosen)


def test_rejected_pair_is_reassigned():
    sets = [
        {1: {"score": 0.99, "bad": True}, 2: {"score": 0.90}},
        {1: {"score": 0.98}, 2: {"score": 0.89}},
    ]
    chosen = solve(
        sets,
        lambda row, second: not row.get("bad") and row["score"] >= 0.35,
        0.35,
    )
    assert chosen == {0: 2, 1: 1}


def test_weak_match_is_not_forced():
    sets = [
        {1: {"score": 0.55}, 2: {"score": 0.56}},
    ]
    chosen = solve(sets, lambda row, second: row["score"] >= 0.80, 0.35)
    assert chosen == {}
