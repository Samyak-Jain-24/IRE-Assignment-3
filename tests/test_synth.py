import numpy as np

from annbench.synth import make_drift_data


def test_generator_is_deterministic():
    a = make_drift_data(0.5, 1000, 500, 100, seed=7)
    b = make_drift_data(0.5, 1000, 500, 100, seed=7)
    np.testing.assert_array_equal(a.old, b.old)
    np.testing.assert_array_equal(a.new, b.new)
    np.testing.assert_array_equal(a.queries, b.queries)


def test_only_the_shift_changes_with_delta():
    c = make_drift_data(0.0, 1000, 500, 100, seed=7)
    d = make_drift_data(2.0, 1000, 500, 100, seed=7)
    np.testing.assert_array_equal(c.old, d.old)                      # old data identical
    shift = d.new - c.new                                             # same draws, constant offset
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(shift[0]), 2.0 * c.radius, rtol=1e-4)
    np.testing.assert_allclose(d.queries - c.queries, np.broadcast_to(shift[0], c.queries.shape), atol=1e-4)
