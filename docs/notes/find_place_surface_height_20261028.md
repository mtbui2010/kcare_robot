# find_place: the surface height is measured (2026-10-28)

Experiment notes moved out of CLAUDE.md. Content unchanged.

`_detect_nearest` (recognition.py) kept only grid points within `place_height_band` (±0.04 m) of the ENV
`height`. At kcare_bucheon's counter (tape 0.915 m, ENV 0.91) the head's get3d read the top at 0.99–1.03 m and
the floor at 0.08 m — the head camera's height / tilt in the TF is off by ~8 cm, more at the back — so only the
cabinet front and the sink were "on the plane": "no free spot on counter". Now one get3d over the whole grid comes
first and `_support_height` takes the most common height within `place_height_search` (0.15 m) of the ENV height
(nearer preferred) as the surface; the band, the support-mask test and the spot choice use it, and the log says
`find_place: the surface measured at 0.993 m (… ENV height 0.915, +0.078)`. The ENV height only says where to look
(and still sets the lift in `move`). The returned spot is the measured 3D point, as before. Read-only check (head
not moved): `kaair_real_capture/find_place_counter_20261028/diag.py <place> [height]`. The camera offset itself
is not fixed.
