from lctrend.cli import _uniform_sample


def test_uniform_sample_phase_selects_another_deterministic_slice():
    assert _uniform_sample(list("abcdefghij"), 2, 0.5) == ["c", "h"]
