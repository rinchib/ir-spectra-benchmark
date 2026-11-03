import numpy as np
from scipy.ndimage import gaussian_filter1d

from config import CONFIG, FLOOR


def _safe(y):    return np.clip(np.nan_to_num(y, nan=0.0), 0.0, None)
def _sumnorm(y):
    y = _safe(y); return y / np.clip(y.sum(1, keepdims=True), 1e-12, None)

def per_spectrum_sid(yt, yp, eps=FLOOR):
    a = _safe(yt) + eps; b = _safe(yp) + eps
    a = a / a.sum(1, keepdims=True); b = b / b.sum(1, keepdims=True)
    return (a * np.log(a / b) + b * np.log(b / a)).sum(1)

def _sigma_pts(sigma_cm):  return float(sigma_cm) / CONFIG["grid_step_cm"]

def per_spectrum_sis(yt, yp, broaden_cm=None):
    if broaden_cm:
        s = _sigma_pts(broaden_cm)
        yt = gaussian_filter1d(_safe(yt), s, axis=1, mode="constant", cval=0.0)
        yp = gaussian_filter1d(_safe(yp), s, axis=1, mode="constant", cval=0.0)
    return 1.0 / (1.0 + per_spectrum_sid(yt, yp))

def sis_metric(yt, yp, broaden_cm=None):
    return float(per_spectrum_sis(yt, yp, broaden_cm).mean())

HEADLINE = CONFIG["headline_metric"]
HEADLINE_CM = {"SIS_b10": 10, "SIS_b15": 15, "SIS": None}[HEADLINE]
SELECTION_CM = None if CONFIG.get("selection_metric", "sid") == "sid" else HEADLINE_CM
print("selection on:", ("unbroadened SID" if SELECTION_CM is None else f"broadened SIS sigma={SELECTION_CM}"))
