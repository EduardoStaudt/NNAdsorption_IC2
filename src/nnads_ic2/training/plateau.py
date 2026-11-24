"""Plateau criterion for the validation curve (phase2.plateau_stop).

The curve is flat when the validation y R2 gained less than `min_gain` from each of the last
`n_points` points to the next (with eval_every = 25 000 and n_points = 2: flat for 50 000 steps).
`plateau_step` returns the first step where that happens, or None. With phase2.plateau_stop.enabled
the run stops there; otherwise the step is only recorded (summary.json, "plateau").
"""


def plateau_step(steps, values, min_gain=0.002, n_points=2):
    """First step at which the last `n_points` gains of `values` were all below `min_gain`."""
    n = int(n_points)
    if n < 1:
        raise ValueError("n_points must be >= 1")
    run = 0
    for i in range(1, len(values)):
        run = run + 1 if values[i] - values[i - 1] < min_gain else 0
        if run >= n:
            return steps[i]
    return None
