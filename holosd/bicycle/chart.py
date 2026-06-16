"""Angle helper shared by the HoloSD core.

Only ``wrap_pi`` is used by the final algorithm (``holosd_final``) and its eval
(``eval_plots``); the original two-point chart machinery lived in the archived
experiment lineage and has been removed.
"""
import numpy as np

PI = np.pi


def wrap_pi(a, xp=np):
    """Wrap angle(s) to [-pi, pi).  wrap_pi(pi) == -pi, wrap_pi(-pi) == -pi."""
    return (a + PI) % (2.0 * PI) - PI
