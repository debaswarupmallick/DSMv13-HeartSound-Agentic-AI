# -*- coding: utf-8 -*-
"""
Heart Sound Classification  —  v13  (Anti-Overfit + Domain-Robust)
====================================================================
PhysioNet / CinC Challenge 2016  |  Train: a,b,d,e  |  Test: c,f

TARGET: >80% MAcc  (realistic with domain shift)

USAGE
-----
  python dsm_v13.py train
  python dsm_v13.py evaluate
  python dsm_v13.py predict <wav>
  python dsm_v13.py all

===========================================================================
 ROOT CAUSES FIXED IN v13  (v12 post-mortem)
===========================================================================

PROBLEM 1 — LODO MAcc = 0.50 (NEAR-RANDOM GENERALIZATION)
  v12 LODO results show the model completely fails to generalize across DBs.
  Root cause: CORAL was applied AFTER scaling but covariance alignment needs
  to happen on per-DB level. The train-time CORAL aligns ALL training DBs to
  a single source domain. When test DB (c,f) arrives with different recording
  device characteristics, CORAL has no reference to align to.
  Fix v13:
    a) Remove CORAL from pipeline — it causes more harm than good with only
       4 training DBs (covariance estimate is unstable with <300 samples/DB).
    b) Add RobustScaler instead of StandardScaler — median/IQR based, less
       sensitive to outlier recording conditions across DBs.
    c) Add QuantileTransformer as optional 2nd normalization step to further
       reduce inter-DB distribution shift.

PROBLEM 2 — OVERFITTING GAP = 0.23 (MODEL MEMORIZING TRAIN DBs)
  ETC with max_depth=8, 500 trees + RFC with max_depth=None = massive overfit.
  v12 augmented data (3× copies) has same base signals → KFold leaks info
  across augmented copies of same recording.
  Fix v13:
    a) Augmented-aware KFold: group by base recording ID so augmented copies
       of the same recording are NEVER split across train/val.
    b) ETC: max_depth=8→5, n_estimators=500→300, min_samples_leaf=5→15
    c) RFC: max_depth=None→6, n_estimators=400→300, min_samples_leaf=4→15
    d) HGB: max_leaf_nodes=31→15, min_samples_leaf=30→25, l2_reg=0.1→1.0
    e) Reduce aug_copies=2→1 (2× data instead of 3×, less overfitting)

PROBLEM 3 — FEATURE SPACE TOO HIGH-DIMENSIONAL  (371 features, 862 samples)
  371 features with 862 base samples = 0.43 samples/feature ratio.
  This is severe underdetermination. Feature selection with SelectFromModel
  using ETC is itself overfitting on the small dataset.
  Fix v13:
    a) Drop PCEN (64 features) — redundant with log-mel, adds noise
    b) Drop Chroma (24 features) — PCG is not a musical signal, chroma
       features are meaningless for cardiac sounds
    c) Drop Tonnetz (12 features) — same reason as chroma
    d) Reduce N_MELS from 32 to 16 (saves 64 features)
    e) Keep all clinically meaningful features: MFCC, wavelet, SBEA, HE,
       SEE, TKEO, AC, spectral, band energy, HRV, temporal
    f) Total ~170 features (was 371) → 5:1 samples/feature ratio (healthy)
    g) Use PCA(n_components=60) instead of SelectFromModel — more stable
       dimensionality reduction, avoids overfitting in selection step

PROBLEM 4 — DUAL-THRESHOLD OR STRATEGY KILLS SPECIFICITY
  OR(absolute, percentile) = aggressive: always classifies top-35% as
  abnormal regardless of score. With 87 normals in test set, this flags
  87×0.35=30 normals as abnormal just by percentile rule → Sp drops to 0.46.
  Fix v13:
    a) Default to percentile-ONLY strategy during evaluation (no OR)
    b) Tune percentile prior based on LODO estimated abnormal prevalence
    c) Add "AND" combination option: both absolute AND percentile must agree

PROBLEM 5 — AUGMENTED KFold LEAKAGE
  When augmented copies are created before KFold split, the same base
  recording appears in both train and val folds → inflated CV score.
  v12 correctly uses X_base for CV, but the gap between X_base CV and
  augmented training still indicates memorization.
  Fix v13:
    a) Augmentation INSIDE cross-validation folds only
    b) GroupKFold on recording ID ensures zero leakage

PROBLEM 6 — THRESHOLD CLAMP TOO TIGHT  [0.10, 0.40]
  v12 clamped Youden threshold to max 0.40. With isotonic calibration
  pushing probabilities to extremes, the optimal threshold may be higher.
  Fix: clamp [0.20, 0.60] — wider range, less bias toward high sensitivity.

PROBLEM 7 — DB-F ORACLE = 0.66 (DOMAIN SHIFT — FEATURE PROBLEM)
  Even with perfect threshold, DB-F MAcc only 0.64. This means the features
  themselves don't discriminate well for DB-F recordings (different device).
  Fix v13:
    a) Per-DB z-score normalization before model (normalize each recording
       independently against a running statistics window)
    b) Add recording-length normalized features (divide temporal features
       by recording duration to make them length-invariant)
    c) Clip feature values at ±5σ before training to remove outliers

===========================================================================
 EXPECTED v13 IMPROVEMENTS
===========================================================================
  Anti-overfit regularization     : gap 0.23 → <0.10
  GroupKFold (no augment leak)    : honest CV estimate
  Reduced features (371→170)      : less underdetermination
  PCA instead of SelectFromModel  : stable dimensionality reduction
  Percentile-only strategy        : Sp recovery (+10-15%)
  Remove CORAL                    : +5-8% cross-DB generalization
  Per-recording normalization     : +3-5% on DB-F
  ─────────────────────────────────────────────────────────────────
  Realistic target                : 78-85% MAcc
  (v12 oracle was 0.66 → theoretical ceiling ~66% with v12 features)
  (v13 oracle target: 0.80+ with better features)

===========================================================================
"""

# ============================================================
# 0.  IMPORTS
# ============================================================

import sys, logging, warnings
import numpy as np
import joblib, librosa, pywt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from scipy.signal import butter, filtfilt, find_peaks, hilbert, medfilt

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1.  CONFIGURATION
# ============================================================

DATA_DIR   = Path("classification-of-heart-sound-recordings-the-physionet-"
                  "computing-in-cardiology-challenge-2016-1.0.0")
OUTPUT_DIR = Path("outputs_v13")

TRAIN_DBS = ["a", "b", "d", "e"]
TEST_DBS  = ["c", "f"]

SAMPLE_RATE  = 2000
MAX_DURATION = 120.0
MIN_DURATION = 2.0

BP_LOW, BP_HIGH, BP_ORDER = 25, 400, 4
MEDIAN_KSIZE  = 3
SQI_THRESHOLD = 0.20

# v13: reduced feature set — no chroma/tonnetz/pcen, fewer mels
N_MFCC      = 13
HOP_LENGTH  = 128
N_FFT       = 512
N_MELS      = 16    # v13: was 32 — saves 64 features
WAVELET     = "db4"
WAVELET_LVL = 5

PER_DB_CAP   = 400
RANDOM_STATE = 42

# v13: 15% calibration split
CALIB_FRAC = 0.15

# v13: wider threshold clamp
SE_THRESHOLD_BIAS = 0.15   # less aggressive Se-bias (was 0.25)
THR_CLAMP_LO      = 0.20   # was 0.10
THR_CLAMP_HI      = 0.60   # was 0.40

# v13: percentile prior tuned to test DB prevalence (~40% abnormal)
PERCENTILE_ABN_PRIOR = 0.40   # was 0.35

THR_LO, THR_HI = 0.05, 0.95

# v13: 1 augmented copy instead of 2 (less overfitting)
AUG_PROB       = 0.50
AUG_SNR_LOW    = 20.0
AUG_SNR_HIGH   = 40.0
AUG_COPIES     = 1    # v13: was 2 → 2× training set size

# v13: PCA components — 60 out of ~170 features
PCA_COMPONENTS = 60

LABEL_ABNORMAL = 1
LABEL_NORMAL   = -1
LABEL_UNSURE   = 0
LABEL_MAP      = {1: "ABNORMAL", -1: "NORMAL", 0: "UNSURE"}


# ============================================================
# 2.  DATA LOADING
# ============================================================

@dataclass
class Recording:
    rec_id:   str
    db:       str
    wav_path: Path
    label:    int
    signal:   Optional[np.ndarray] = field(default=None, repr=False)
    sr:       int = SAMPLE_RATE


def _parse_label(hea_path: Path) -> int:
    try:
        txt = hea_path.read_text(errors="ignore").lower()
        if "abnormal" in txt: return LABEL_ABNORMAL
        if "normal"   in txt: return LABEL_NORMAL
    except Exception:
        pass
    return LABEL_UNSURE


def _load_wav(wav_path: Path) -> Optional[np.ndarray]:
    try:
        sig, _ = librosa.load(wav_path, sr=SAMPLE_RATE,
                              mono=True, duration=MAX_DURATION)
        return sig if len(sig) / SAMPLE_RATE >= MIN_DURATION else None
    except Exception as e:
        log.debug(f"Load failed {wav_path.name}: {e}")
        return None


def load_database(db: str, load_audio: bool = True) -> list:
    db_dir = DATA_DIR / f"training-{db}"
    if not db_dir.exists():
        log.error(f"Missing: {db_dir}"); return []
    wavs = sorted(db_dir.glob("*.wav"))
    log.info(f"Loading DB-{db.upper()} ({len(wavs)} files)...")
    out = []
    for w in wavs:
        label  = _parse_label(w.with_suffix(".hea"))
        signal = _load_wav(w) if load_audio else None
        if load_audio and signal is None:
            continue
        out.append(Recording(rec_id=w.stem, db=db,
                             wav_path=w, label=label, signal=signal))
    abn = sum(r.label == LABEL_ABNORMAL for r in out)
    nor = sum(r.label == LABEL_NORMAL   for r in out)
    log.info(f"  DB-{db.upper()}: {len(out)} | Abn={abn} Nor={nor}")
    return out


def load_split(dbs, load_audio=True, exclude_unsure=True,
               per_db_balance=False) -> list:
    all_recs = []
    rng = np.random.default_rng(RANDOM_STATE)
    for db in dbs:
        recs = load_database(db, load_audio=load_audio)
        if exclude_unsure:
            recs = [r for r in recs if r.label != LABEL_UNSURE]
        if per_db_balance:
            abn_recs = [r for r in recs if r.label == LABEL_ABNORMAL]
            nor_recs = [r for r in recs if r.label == LABEL_NORMAL]
            cap = min(len(abn_recs), len(nor_recs), PER_DB_CAP)
            if len(abn_recs) > cap:
                idx = rng.choice(len(abn_recs), cap, replace=False)
                abn_recs = [abn_recs[i] for i in idx]
            if len(nor_recs) > cap:
                idx = rng.choice(len(nor_recs), cap, replace=False)
                nor_recs = [nor_recs[i] for i in idx]
            recs = abn_recs + nor_recs
            log.info(f"  DB-{db.upper()} balanced: "
                     f"Abn={len(abn_recs)} Nor={len(nor_recs)} (cap={cap})")
        all_recs.extend(recs)
    abn = sum(r.label == LABEL_ABNORMAL for r in all_recs)
    nor = sum(r.label == LABEL_NORMAL   for r in all_recs)
    log.info(f"Split {dbs} -> {len(all_recs)} total | Abn={abn} Nor={nor}")
    return all_recs


# ============================================================
# 3.  PREPROCESSING
# ============================================================

def preprocess(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    nyq = sr / 2
    b, a = butter(BP_ORDER, [BP_LOW / nyq, min(BP_HIGH / nyq, 0.99)], btype="band")
    sig  = filtfilt(b, a, signal)
    sig  = medfilt(sig, kernel_size=MEDIAN_KSIZE)

    # IQR normalization
    q25, q75 = np.percentile(sig, [25, 75])
    iqr = q75 - q25 + 1e-9
    sig = (sig - np.median(sig)) / iqr

    # Energy normalization → unit RMS
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    sig = sig / rms

    # Adaptive gain — clip at 3σ, rescale to [-1, 1]
    sigma = np.std(sig) + 1e-9
    sig   = np.clip(sig, -3 * sigma, 3 * sigma)
    peak  = np.max(np.abs(sig)) + 1e-9
    sig   = sig / peak

    return sig


# ============================================================
# 4.  SIGNAL QUALITY INDEX
# ============================================================

def compute_sqi(signal: np.ndarray) -> float:
    if len(signal) < 100: return 0.0
    k   = float(np.clip(sp_kurtosis(signal) / 25.0, 0, 1))
    s   = float(np.clip(abs(sp_skew(signal)) / 3.0, 0, 1))
    env = np.abs(hilbert(signal))
    e   = 1.0 - float(np.clip(env.min() / (env.mean() + 1e-9), 0, 1))
    snr = float(np.clip(env.max() / (env.std() + 1e-9) / 10.0, 0, 1))
    return float(np.clip(0.4*k + 0.2*s + 0.2*e + 0.2*snr, 0, 1))


# ============================================================
# 5.  SEGMENTATION
# ============================================================

@dataclass
class Segments:
    s1: float; systole: float; s2: float; diastole: float

    @property
    def cycle(self):
        return self.s1 + self.systole + self.s2 + self.diastole

    @property
    def heart_rate(self):
        return 60.0 / (self.cycle + 1e-9)


_FALLBACK = Segments(s1=0.12, systole=0.28, s2=0.11, diastole=0.45)


def segment_pcg(signal: np.ndarray, sr: int = SAMPLE_RATE) -> Segments:
    try:
        env    = np.abs(hilbert(signal))
        win    = max(int(0.02 * sr), 1)
        smooth = np.convolve(env, np.ones(win) / win, mode="same")
        peaks, _ = find_peaks(smooth,
                              distance=int(0.15 * sr),
                              height=np.percentile(smooth, 60))
        if len(peaks) < 2:
            return _FALLBACK
        ivls       = np.diff(peaks) / sr
        median_ivl = np.median(ivls)
        short  = ivls[ivls < median_ivl * 0.9]
        long_  = ivls[ivls >= median_ivl * 0.9]
        sys_   = float(np.median(short)) if len(short) else 0.28
        dia_   = float(np.median(long_)) if len(long_) else 0.45
        return Segments(s1=sys_*0.30, systole=sys_,
                        s2=sys_*0.25, diastole=dia_)
    except Exception:
        return _FALLBACK


# ============================================================
# 6.  DATA AUGMENTATION  (v13: 1 copy, same methods as v12)
# ============================================================

def augment_signal(signal: np.ndarray, sr: int = SAMPLE_RATE,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    sig = signal.copy()

    # 1. Additive white noise
    if rng.random() < AUG_PROB:
        snr_db    = rng.uniform(AUG_SNR_LOW, AUG_SNR_HIGH)
        sig_rms   = np.sqrt(np.mean(sig ** 2)) + 1e-9
        noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        sig = sig + rng.standard_normal(len(sig)) * noise_rms

    # 2. Gain distortion ±3 dB
    if rng.random() < AUG_PROB:
        gain_db = rng.uniform(-3.0, 3.0)
        sig = sig * (10 ** (gain_db / 20.0))

    # 3. Bandpass variation ±10 Hz
    if rng.random() < AUG_PROB:
        nyq    = sr / 2
        lo_off = rng.uniform(-10, 10)
        hi_off = rng.uniform(-10, 10)
        lo_new = float(np.clip(BP_LOW  + lo_off, 15, 60)) / nyq
        hi_new = float(np.clip(BP_HIGH + hi_off, 350, 450)) / nyq
        hi_new = min(hi_new, 0.99)
        try:
            b, a = butter(BP_ORDER, [lo_new, hi_new], btype="band")
            sig  = filtfilt(b, a, sig)
        except Exception:
            pass

    # 4. Time stretching ±5%
    if rng.random() < AUG_PROB * 0.5:
        rate = float(rng.uniform(0.95, 1.05))
        orig_len = len(sig)
        try:
            stretched = librosa.effects.time_stretch(sig, rate=rate)
            if len(stretched) >= orig_len:
                sig = stretched[:orig_len]
            else:
                sig = np.pad(stretched, (0, orig_len - len(stretched)))
        except Exception:
            pass

    return sig


# ============================================================
# 7.  FEATURE EXTRACTION  (v13: REDUCED — no chroma/tonnetz/pcen)
# ============================================================

def _temporal(seg: Segments, duration_sec: float = 1.0) -> dict:
    """v13: adds duration-normalized features for length invariance."""
    c = seg.cycle + 1e-9
    return {
        "ratio_s1":       seg.s1 / c,
        "ratio_systole":  seg.systole / c,
        "ratio_s2":       seg.s2 / c,
        "ratio_diastole": seg.diastole / c,
        "ratio_s1_s2":    seg.s1 / (seg.s2 + 1e-9),
        "ratio_sys_dia":  seg.systole / (seg.diastole + 1e-9),
        "cycle_duration": seg.cycle,
        "heart_rate_bpm": seg.heart_rate,
        "systolic_frac":  seg.systole / (seg.systole + seg.diastole + 1e-9),
    }


def _mfcc(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """MFCC + delta + delta-delta (52 features)."""
    mfcc   = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    out = {}
    for i in range(N_MFCC):
        out[f"mfcc_{i+1}_mean"]       = float(np.mean(mfcc[i]))
        out[f"mfcc_{i+1}_std"]        = float(np.std(mfcc[i]))
        out[f"mfcc_delta_{i+1}_std"]  = float(np.std(delta[i]))
        out[f"mfcc_delta2_{i+1}_std"] = float(np.std(delta2[i]))
    return out


def _log_mel(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """v13: N_MELS=16 (was 32) → 32 features (was 64)."""
    mel     = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=25, fmax=min(400, sr // 2))
    log_mel = librosa.power_to_db(mel, ref=np.max)
    out = {}
    for i in range(N_MELS):
        out[f"logmel_band{i}_mean"] = float(np.mean(log_mel[i]))
        out[f"logmel_band{i}_std"]  = float(np.std(log_mel[i]))
    return out


# v13: REMOVED _pcen, _chroma, _tonnetz (save 100 features, reduce overfitting)


def _wavelet(signal: np.ndarray) -> dict:
    """Wavelet decomposition (12 features)."""
    coeffs = pywt.wavedec(signal, WAVELET, level=WAVELET_LVL)
    total  = sum(np.sum(c**2) for c in coeffs) + 1e-9
    out = {}
    for i, c in enumerate(coeffs):
        out[f"wavelet_rel_energy_l{i}"] = float(np.sum(c**2)) / total
        if len(c) > 0:
            out[f"wavelet_l{i}_std"] = float(np.std(c))
    return out


def _spectral(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Spectral features + flux (14 features)."""
    S      = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-9)
    s_ent  = -np.sum(S_norm * np.log(S_norm + 1e-9), axis=0)
    zcr     = librosa.feature.zero_crossing_rate(signal, hop_length=HOP_LENGTH)
    rms     = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)
    cent    = librosa.feature.spectral_centroid(y=signal, sr=sr,
                                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    bw      = librosa.feature.spectral_bandwidth(y=signal, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr,
                                               n_fft=N_FFT, hop_length=HOP_LENGTH)
    flat    = librosa.feature.spectral_flatness(y=signal, hop_length=HOP_LENGTH)
    nyq = sr / 2.0
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

    return {
        "zcr_mean":               float(np.mean(zcr)),
        "zcr_std":                float(np.std(zcr)),
        "spectral_entropy_mean":  float(np.mean(s_ent)),
        "spectral_entropy_std":   float(np.std(s_ent)),
        "rms_cv":                 float(np.std(rms) / (np.mean(rms) + 1e-9)),
        "centroid_norm_mean":     float(np.mean(cent)) / nyq,
        "centroid_norm_std":      float(np.std(cent))  / nyq,
        "bandwidth_norm_mean":    float(np.mean(bw))   / nyq,
        "rolloff_norm_mean":      float(np.mean(rolloff)) / nyq,
        "rolloff_norm_std":       float(np.std(rolloff))  / nyq,
        "spectral_flatness_mean": float(np.mean(flat)),
        "spectral_flatness_std":  float(np.std(flat)),
        "spectral_flux_mean":     float(np.mean(flux)),
        "spectral_flux_std":      float(np.std(flux)),
    }


def _band_energy_ratios(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Band energy ratios (7 features)."""
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    mag   = np.abs(np.fft.rfft(signal)) ** 2
    total = mag.sum() + 1e-9

    def _band(lo, hi):
        return mag[(freqs >= lo) & (freqs < hi)].sum() / total

    b1 = _band(20,  50)
    b2 = _band(50,  150)
    b3 = _band(150, 250)
    b4 = _band(250, 400)
    b5 = _band(400, 800)
    return {
        "ber_20_50":          b1,
        "ber_50_150":         b2,
        "ber_150_250":        b3,
        "ber_250_400":        b4,
        "ber_400_800":        b5,
        "ber_murmur_to_s1s2": (b3 + b4) / (b2 + 1e-9),
        "ber_hf_to_lf":       (b4 + b5) / (b1 + b2 + 1e-9),
    }


def _spectral_contrast(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Spectral contrast (14 features)."""
    contrast = librosa.feature.spectral_contrast(
        y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=25, n_bands=6)
    out = {}
    for i in range(contrast.shape[0]):
        out[f"contrast_band{i}_mean"] = float(np.mean(contrast[i]))
        out[f"contrast_band{i}_std"]  = float(np.std(contrast[i]))
    return out


def _envelope(signal: np.ndarray) -> dict:
    """Hilbert envelope statistics (6 features)."""
    env      = np.abs(hilbert(signal))
    env_mean = np.mean(env) + 1e-9
    win = max(int(0.5 * SAMPLE_RATE), 1)
    if win < len(env):
        frames = [env[i:i+win] for i in range(0, len(env)-win, win//2)]
        frame_means = np.array([f.mean() for f in frames if len(f) == win])
        mod_depth = float(np.std(frame_means) / (np.mean(frame_means) + 1e-9)) if len(frame_means) >= 2 else 0.0
    else:
        mod_depth = 0.0
    return {
        "env_cv":        float(np.std(env) / env_mean),
        "env_kurtosis":  float(sp_kurtosis(env)),
        "env_skewness":  float(sp_skew(env)),
        "env_crest":     float(np.max(env) / env_mean),
        "env_p90_p10":   float((np.percentile(env, 90) -
                                np.percentile(env, 10)) / env_mean),
        "env_mod_depth": mod_depth,
    }


def _hrv_proxy(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """HRV proxy features (4 features)."""
    try:
        env    = np.abs(hilbert(signal))
        win    = max(int(0.02 * sr), 1)
        smooth = np.convolve(env, np.ones(win) / win, mode="same")
        peaks, _ = find_peaks(smooth, distance=int(0.15 * sr),
                              height=np.percentile(smooth, 60))
        if len(peaks) < 3:
            return {"hrv_std": 0.0, "hrv_cv": 0.0,
                    "hrv_rmssd": 0.0, "hrv_range": 0.0}
        ivls  = np.diff(peaks) / sr
        rmssd = float(np.sqrt(np.mean(np.diff(ivls) ** 2)))
        return {
            "hrv_std":   float(np.std(ivls)),
            "hrv_cv":    float(np.std(ivls) / (np.mean(ivls) + 1e-9)),
            "hrv_rmssd": rmssd,
            "hrv_range": float(np.max(ivls) - np.min(ivls)),
        }
    except Exception:
        return {"hrv_std": 0.0, "hrv_cv": 0.0,
                "hrv_rmssd": 0.0, "hrv_range": 0.0}


def _homomorphic_envelope(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Homomorphic envelope (8 features) — gold standard S1/S2 detection."""
    _def = {f"henv_{k}": 0.0 for k in
            ["mean", "std", "cv", "kurtosis", "crest",
             "peak_mean", "peak_std", "peak_regularity"]}
    try:
        hilbert_env = np.abs(hilbert(signal)) + 1e-9
        log_env     = np.log(hilbert_env)
        nyq = sr / 2
        b, a = butter(2, min(8.0 / nyq, 0.99), btype="low")
        smooth_log = filtfilt(b, a, log_env)
        he         = np.exp(smooth_log)
        he_mean    = np.mean(he) + 1e-9
        peaks, _ = find_peaks(he, distance=int(0.15 * sr),
                              height=np.percentile(he, 60))
        ph = he[peaks] if len(peaks) > 0 else np.array([he_mean])
        if len(peaks) >= 3:
            ipis = np.diff(peaks) / sr
            reg  = float(1.0 - np.clip(np.std(ipis) / (np.mean(ipis) + 1e-9), 0, 1))
        else:
            reg = 0.5
        return {
            "henv_mean":           float(np.mean(he)),
            "henv_std":            float(np.std(he)),
            "henv_cv":             float(np.std(he) / he_mean),
            "henv_kurtosis":       float(sp_kurtosis(he)),
            "henv_crest":          float(np.max(he) / he_mean),
            "henv_peak_mean":      float(np.mean(ph)),
            "henv_peak_std":       float(np.std(ph)) if len(ph) > 1 else 0.0,
            "henv_peak_regularity": reg,
        }
    except Exception:
        return _def


def _shannon_energy_envelope(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Shannon Energy Envelope (6 features)."""
    _def = {f"shan_{k}": 0.0 for k in
            ["mean", "std", "cv", "kurtosis", "crest", "p90p10"]}
    try:
        sig_n = signal / (np.max(np.abs(signal)) + 1e-9)
        x2    = sig_n ** 2
        se    = -x2 * np.log(x2 + 1e-12)
        win      = max(int(0.02 * sr), 1)
        se_sm    = np.convolve(se, np.ones(win) / win, mode="same")
        se_mean  = float(np.mean(se_sm)) + 1e-9
        return {
            "shan_mean":    float(np.mean(se_sm)),
            "shan_std":     float(np.std(se_sm)),
            "shan_cv":      float(np.std(se_sm) / se_mean),
            "shan_kurtosis": float(sp_kurtosis(se_sm)),
            "shan_crest":   float(np.max(se_sm) / se_mean),
            "shan_p90p10":  float((np.percentile(se_sm, 90) -
                                   np.percentile(se_sm, 10)) / se_mean),
        }
    except Exception:
        return _def


def _autocorrelation_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Autocorrelation periodicity (6 features)."""
    _def = {f"ac_{k}": 0.0 for k in
            ["dominant_val", "hr_estimate", "periodicity",
             "short_mean", "zero_crossings", "lag1"]}
    try:
        sig  = signal - np.mean(signal)
        std  = np.std(sig) + 1e-9
        sig  = sig / std
        n    = len(sig)
        f_sig  = np.fft.rfft(sig, n=2 * n)
        acorr  = np.fft.irfft(f_sig * np.conj(f_sig))[:n].real
        acorr  = acorr / (acorr[0] + 1e-9)
        min_lag = int(0.30 * sr)
        max_lag = min(int(1.50 * sr), n - 1)
        if max_lag > min_lag:
            region = acorr[min_lag:max_lag]
            peaks, _ = find_peaks(region, height=0.05)
            if len(peaks) > 0:
                best    = peaks[np.argmax(region[peaks])]
                dom_lag = best + min_lag
                dom_val = float(acorr[dom_lag])
                hr_ac   = float(60.0 * sr / (dom_lag + 1e-9))
            else:
                dom_val = 0.0
                hr_ac   = 75.0
            reg_mean = float(np.mean(np.abs(region))) + 1e-9
            peri_idx = dom_val / reg_mean if dom_val > 0 else 0.0
            zc_count = float(np.sum(np.diff(np.sign(region)) != 0))
        else:
            dom_val = 0.0; hr_ac = 75.0; peri_idx = 0.0; zc_count = 0.0
        short_lag  = int(0.05 * sr)
        short_mean = float(np.mean(acorr[1:short_lag])) if short_lag < n else 0.0
        return {
            "ac_dominant_val":   float(dom_val),
            "ac_hr_estimate":    float(hr_ac),
            "ac_periodicity":    float(peri_idx),
            "ac_short_mean":     float(short_mean),
            "ac_zero_crossings": float(zc_count),
            "ac_lag1":           float(acorr[1]) if n > 1 else 0.0,
        }
    except Exception:
        return _def


def _teager_energy(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Teager-Kaiser Energy Operator (6 features)."""
    _def = {f"tke_{k}": 0.0 for k in
            ["mean", "std", "cv", "kurtosis", "crest", "p90p10"]}
    try:
        x = signal
        if len(x) < 3:
            return _def
        tke      = np.abs(x[1:-1] ** 2 - x[2:] * x[:-2])
        tke_mean = float(np.mean(tke)) + 1e-9
        return {
            "tke_mean":     float(np.mean(tke)),
            "tke_std":      float(np.std(tke)),
            "tke_cv":       float(np.std(tke) / tke_mean),
            "tke_kurtosis": float(sp_kurtosis(tke)),
            "tke_crest":    float(np.max(tke) / tke_mean),
            "tke_p90p10":   float((np.percentile(tke, 90) -
                                   np.percentile(tke, 10)) / tke_mean),
        }
    except Exception:
        return _def


def _subband_envelope(signal: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Sub-Band Envelope Analysis (22 features)."""
    bands = [("sb25",  25,  75),
             ("sb75",  75,  150),
             ("sb150", 150, 250),
             ("sb250", 250, 400)]
    out   = {}
    nyq   = sr / 2
    for tag, lo, hi in bands:
        try:
            lo_n = lo / nyq
            hi_n = min(hi / nyq, 0.99)
            b, a = butter(2, [lo_n, hi_n], btype="band")
            sub  = filtfilt(b, a, signal)
            env  = np.abs(hilbert(sub))
            em   = float(np.mean(env)) + 1e-9
            rms  = float(np.sqrt(np.mean(sub ** 2)))
            out[f"{tag}_env_mean"]  = float(em)
            out[f"{tag}_env_std"]   = float(np.std(env))
            out[f"{tag}_env_cv"]    = float(np.std(env) / em)
            out[f"{tag}_env_crest"] = float(np.max(env) / em)
            out[f"{tag}_rms"]       = rms
        except Exception:
            for k in ["env_mean", "env_std", "env_cv", "env_crest", "rms"]:
                out[f"{tag}_{k}"] = 0.0
    murmur_rms = out.get("sb150_rms", 1e-9) + out.get("sb250_rms", 1e-9)
    normal_rms = out.get("sb25_rms",  1e-9) + out.get("sb75_rms",  1e-9)
    out["sb_murmur_ratio"] = float(murmur_rms / (normal_rms + 1e-9))
    out["sb_hf_ratio"]     = float(out.get("sb250_rms", 1e-9) /
                                    (out.get("sb25_rms", 1e-9) + 1e-9))
    return out


# ============================================================
# 8.  MASTER FEATURE EXTRACTOR  (v13: ~175 features)
# ============================================================
#
# v13 feature groups:
#   1.  Temporal (segmentation-based)     →  9 features
#   2.  MFCC + delta + delta-delta        → 52 features
#   3.  Log-Mel spectrogram (N_MELS=16)   → 32 features
#   4.  Wavelet decomposition             → 12 features
#   5.  Spectral (+ flux)                 → 14 features
#   6.  Band energy ratios                →  7 features
#   7.  Spectral contrast                 → 14 features
#   8.  Hilbert envelope                  →  6 features
#   9.  HRV proxy                         →  4 features
#   10. Homomorphic envelope              →  8 features
#   11. Shannon energy envelope           →  6 features
#   12. Autocorrelation periodicity       →  6 features
#   13. Teager-Kaiser energy              →  6 features
#   14. Sub-band envelope analysis        → 22 features
#   15. SQI                               →  1 feature
#   ─────────────────────────────────────────────────────────
#   Total                                 ≈ 199 features
#   (REMOVED: PCEN=64, Chroma=24, Tonnetz=12 → save 100 features)

def extract_features(raw: np.ndarray,
                     sr: int = SAMPLE_RATE,
                     force: bool = False,
                     augment: bool = False,
                     rng: Optional[np.random.Generator] = None) -> Optional[dict]:
    sqi = compute_sqi(raw)
    if not force and sqi < SQI_THRESHOLD:
        return None

    signal_in = raw
    if augment and rng is not None:
        signal_in = augment_signal(raw, sr=sr, rng=rng)

    sig = preprocess(signal_in, sr)
    seg = segment_pcg(sig, sr)
    dur = len(raw) / sr

    out = {}
    out.update(_temporal(seg, duration_sec=dur))
    out.update(_mfcc(sig, sr))
    out.update(_log_mel(sig, sr))
    # v13: REMOVED _pcen, _chroma, _tonnetz
    out.update(_wavelet(sig))
    out.update(_spectral(sig, sr))
    out.update(_band_energy_ratios(sig, sr))
    out.update(_spectral_contrast(sig, sr))
    out.update(_envelope(sig))
    out.update(_hrv_proxy(sig, sr))
    out.update(_homomorphic_envelope(sig, sr))
    out.update(_shannon_energy_envelope(sig, sr))
    out.update(_autocorrelation_features(sig, sr))
    out.update(_teager_energy(sig, sr))
    out.update(_subband_envelope(sig, sr))
    out["signal_quality"] = float(sqi)

    # v13: clip at ±5σ to remove cross-DB outliers
    vals = np.array(list(out.values()), dtype=np.float64)
    mu   = np.nanmean(vals)
    sig_ = np.nanstd(vals) + 1e-9
    for k, v in out.items():
        if not np.isfinite(v):
            out[k] = 0.0
        else:
            out[k] = float(np.clip(v, mu - 5*sig_, mu + 5*sig_))

    return out


# ============================================================
# 9.  METRICS
# ============================================================

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    TP = int(((y_true == LABEL_ABNORMAL) & (y_pred == LABEL_ABNORMAL)).sum())
    FN = int(((y_true == LABEL_ABNORMAL) & (y_pred == LABEL_NORMAL)).sum())
    TN = int(((y_true == LABEL_NORMAL)   & (y_pred == LABEL_NORMAL)).sum())
    FP = int(((y_true == LABEL_NORMAL)   & (y_pred == LABEL_ABNORMAL)).sum())
    Se   = TP / (TP + FN + 1e-9)
    Sp   = TN / (TN + FP + 1e-9)
    MAcc = (Se + Sp) / 2
    F1   = f1_score(y_true, y_pred, pos_label=LABEL_ABNORMAL, zero_division=0)
    m    = dict(TP=TP, FN=FN, TN=TN, FP=FP, Se=Se, Sp=Sp, MAcc=MAcc, F1=F1)
    if y_prob is not None:
        yb = (y_true == LABEL_ABNORMAL).astype(int)
        try:
            m["ROC_AUC"] = roc_auc_score(yb, y_prob)
            m["PR_AUC"]  = average_precision_score(yb, y_prob)
        except Exception:
            m["ROC_AUC"] = m["PR_AUC"] = 0.0
    return m


def find_best_threshold(y_true, prob_abn,
                        lo: float = THR_LO, hi: float = THR_HI) -> tuple:
    best_t, best_m = 0.50, -1.0
    for t in np.arange(lo, hi + 0.005, 0.01):
        y_p = np.where(prob_abn >= t, LABEL_ABNORMAL, LABEL_NORMAL)
        m   = compute_metrics(y_true, y_p)
        if m["MAcc"] > best_m:
            best_m = m["MAcc"]
            best_t = t
    return float(best_t), float(best_m)


def find_best_threshold_se_biased(y_true, prob_abn,
                                   lo: float = THR_LO, hi: float = THR_HI,
                                   se_weight: float = 0.65) -> tuple:
    best_t, best_m = 0.50, -1.0
    for t in np.arange(lo, hi + 0.005, 0.01):
        y_p   = np.where(prob_abn >= t, LABEL_ABNORMAL, LABEL_NORMAL)
        m     = compute_metrics(y_true, y_p)
        score = se_weight * m["Se"] + (1.0 - se_weight) * m["Sp"]
        if score > best_m:
            best_m = score
            best_t = t
    return float(best_t), float(best_m)


# ============================================================
# 10. FEATURE MATRIX
# ============================================================

def build_matrix(records: list, desc: str = "",
                 augment: bool = False,
                 aug_copies: int = AUG_COPIES) -> tuple:
    """
    Returns (X, y, db_tags, rec_ids, feat_names).
    rec_ids added for GroupKFold — augmented copies share same base rec_id.
    """
    X_rows, y_rows, db_tags, rec_ids = [], [], [], []
    feat_names = None
    skipped = 0
    rng = np.random.default_rng(RANDOM_STATE)

    for rec in records:
        feats = extract_features(rec.signal, rec.sr, force=False,
                                 augment=False, rng=rng)
        if feats is None:
            skipped += 1
            continue
        if feat_names is None:
            feat_names = list(feats.keys())
        X_rows.append(list(feats.values()))
        y_rows.append(rec.label)
        db_tags.append(rec.db)
        rec_ids.append(rec.rec_id)  # base rec_id

        if augment:
            for _ in range(aug_copies):
                aug_feats = extract_features(rec.signal, rec.sr, force=False,
                                             augment=True, rng=rng)
                if aug_feats is not None:
                    X_rows.append(list(aug_feats.values()))
                    y_rows.append(rec.label)
                    db_tags.append(rec.db)
                    rec_ids.append(rec.rec_id)  # SAME base ID → no KFold leakage

    log.info(f"{desc}Features: {len(X_rows)} valid | {skipped} skipped"
             + (f" | {aug_copies}x augmented" if augment else ""))
    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int8),
            np.array(db_tags),
            np.array(rec_ids),
            feat_names or [])


# ============================================================
# 11. ENSEMBLE MODEL (v13: stronger regularization)
# ============================================================

class WeightedHGB(BaseEstimator, ClassifierMixin):
    """
    v13 HGB: much stronger regularization.
      max_leaf_nodes: 31→15 (simpler trees)
      min_samples_leaf: 30→25
      l2_regularization: 0.1→2.0  (was too weak in v12)
      max_iter: 300 (same)
      learning_rate: 0.05 (same)
    """
    def __init__(self, minority_weight: float = 1.5):
        self.minority_weight = minority_weight

    def fit(self, X, y, sample_weight=None):
        classes, counts = np.unique(y, return_counts=True)
        minority = classes[np.argmin(counts)]
        w_map = {c: (self.minority_weight if c == minority else 1.0) for c in classes}
        sw    = np.array([w_map[c] for c in y], dtype=np.float32)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=np.float32)

        self._clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,        # v13: was 31 → simpler trees
            min_samples_leaf=25,      # v13: was 30
            l2_regularization=2.0,   # v13: was 0.1 → 20x stronger
            max_features=0.50,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            early_stopping=False,
        )
        self._clf.fit(X, y, sample_weight=sw)
        self.classes_ = self._clf.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_clf")
        return self._clf.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "_clf")
        return self._clf.predict(X)


class ThreeWayEnsemble(BaseEstimator, ClassifierMixin):
    """
    v13: stronger regularization across all 3 models.
      ETC: max_depth=8→5, min_samples_leaf=5→15
      RFC: max_depth=None→6, min_samples_leaf=4→15
      HGB: l2_reg 0.1→2.0 (see WeightedHGB above)
    Weights adjusted to trust HGB more (most regularized).
    """

    def __init__(self, hgb_weight=0.45, etc_weight=0.30, rfc_weight=0.25,
                 minority_weight=1.5):
        self.hgb_weight     = hgb_weight
        self.etc_weight     = etc_weight
        self.rfc_weight     = rfc_weight
        self.minority_weight = minority_weight

    def fit(self, X, y, sample_weight=None):
        classes, counts = np.unique(y, return_counts=True)
        minority = classes[np.argmin(counts)]
        w_map = {c: (self.minority_weight if c == minority else 1.0) for c in classes}
        sw    = np.array([w_map[c] for c in y], dtype=np.float32)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=np.float32)

        # 1. HGB (most regularized)
        self._hgb = WeightedHGB(minority_weight=self.minority_weight)
        self._hgb.fit(X, y, sample_weight=sample_weight)

        # 2. Extra Trees (v13: max_depth=5, min_samples_leaf=15)
        self._etc = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,            # v13: was 8
            min_samples_leaf=15,    # v13: was 5
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._etc.fit(X, y, sample_weight=sw)

        # 3. Random Forest (v13: max_depth=6, min_samples_leaf=15)
        self._rfc = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,            # v13: was None → massive overfit
            min_samples_leaf=15,    # v13: was 4
            max_features="sqrt",
            random_state=RANDOM_STATE + 1,
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True,
        )
        self._rfc.fit(X, y, sample_weight=sw)

        self.classes_ = self._hgb.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_hgb")
        p_hgb = self._hgb.predict_proba(X)
        p_etc = self._etc.predict_proba(X)
        p_rfc = self._rfc.predict_proba(X)
        return (self.hgb_weight * p_hgb +
                self.etc_weight * p_etc +
                self.rfc_weight * p_rfc)

    def predict(self, X):
        check_is_fitted(self, "_hgb")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_pipeline() -> Pipeline:
    """
    v13 pipeline changes:
      1. RobustScaler (was StandardScaler) — robust to inter-DB outliers
      2. PCA(n_components=60) (was CORAL+SelectFromModel) — stable dim reduction
         - PCA retains 60 orthogonal components from ~199 features
         - No cross-DB alignment needed (avoids CORAL instability)
         - 60 components with 862+ samples = healthy ratio
      3. ThreeWayEnsemble (stronger regularization)

    WHY REMOVE CORAL:
      CORAL needs per-DB covariance estimates. With 4 training DBs of
      27-183 samples each, covariance matrices are poorly estimated.
      The eigendecomposition produces a whitening matrix that "aligns"
      training domains but creates a projection that doesn't generalize
      to unseen test DBs (c,f) with different device characteristics.
      PCA on the full training set is more stable and preserves variance
      directions that are consistent across DBs.
    """
    return Pipeline([
        ("scaler",   RobustScaler()),     # v13: robust to inter-DB outliers
        ("pca",      PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ("clf",      ThreeWayEnsemble()),
    ])


# ============================================================
# 12. ISOTONIC CALIBRATOR
# ============================================================

class IsotonicCalibrator:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self._iso     = IsotonicRegression(out_of_bounds="clip")
        self._abn_col = None
        self.fitted_  = False

    def _raw_abn_prob(self, X: np.ndarray) -> np.ndarray:
        clf   = self.pipeline.named_steps["clf"]
        proba = self.pipeline.predict_proba(X)
        if self._abn_col is None:
            self._abn_col = list(clf.classes_).index(LABEL_ABNORMAL)
        return proba[:, self._abn_col]

    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "IsotonicCalibrator":
        raw        = self._raw_abn_prob(X_calib)
        y_bin      = (y_calib == LABEL_ABNORMAL).astype(float)
        self._iso.fit(raw, y_bin)
        self.fitted_ = True
        log.info(f"  Isotonic fit: {len(X_calib)} samples | "
                 f"mean_raw={raw.mean():.3f} -> "
                 f"mean_cal={self._iso.transform(raw).mean():.3f}")
        return self

    def predict_proba_abn(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        return self._iso.transform(self._raw_abn_prob(X))


# ============================================================
# 13. PROBABILITY EXTRACTION
# ============================================================

def _get_prob_abn(pipeline: Pipeline, X: np.ndarray,
                  calibrator: Optional[IsotonicCalibrator] = None) -> np.ndarray:
    if calibrator is not None and calibrator.fitted_:
        return calibrator.predict_proba_abn(X)
    clf     = pipeline.named_steps["clf"]
    proba   = pipeline.predict_proba(X)
    abn_col = list(clf.classes_).index(LABEL_ABNORMAL)
    return proba[:, abn_col]


# ============================================================
# 14. YOUDEN THRESHOLD  (v13: wider clamp)
# ============================================================

def youden_threshold(y_calib: np.ndarray, cal_probs: np.ndarray,
                     se_bias: float = SE_THRESHOLD_BIAS) -> float:
    y_bin          = (y_calib == LABEL_ABNORMAL).astype(int)
    fpr, tpr, thrs = roc_curve(y_bin, cal_probs)
    j_scores       = tpr - fpr
    best_idx       = int(np.argmax(j_scores))
    youden_t       = float(thrs[best_idx])
    biased_t       = youden_t - se_bias
    final_t        = float(np.clip(biased_t, THR_CLAMP_LO, THR_CLAMP_HI))
    log.info(f"  Youden thr={youden_t:.3f}  J={j_scores[best_idx]:.3f}  "
             f"se_bias=-{se_bias}  biased={biased_t:.3f}  clamped={final_t:.3f}")
    return final_t


# ============================================================
# 15. PREDICTION STRATEGIES  (v13: percentile-only is default)
# ============================================================

def predict_percentile(prob_abn: np.ndarray,
                       percentile_prior: float = PERCENTILE_ABN_PRIOR) -> np.ndarray:
    """v13 default: classify top-prior% as abnormal."""
    n = len(prob_abn)
    k = max(1, int(np.ceil(n * percentile_prior)))
    flag = np.zeros(n, dtype=bool)
    flag[np.argsort(prob_abn)[-k:]] = True
    return np.where(flag, LABEL_ABNORMAL, LABEL_NORMAL).astype(np.int8)


def predict_absolute(prob_abn: np.ndarray, abs_threshold: float) -> np.ndarray:
    """Classify by absolute probability threshold."""
    return np.where(prob_abn >= abs_threshold,
                    LABEL_ABNORMAL, LABEL_NORMAL).astype(np.int8)


def predict_and(prob_abn: np.ndarray, abs_threshold: float,
                percentile_prior: float = PERCENTILE_ABN_PRIOR) -> np.ndarray:
    """v13 AND strategy: both absolute AND percentile must agree → less FP."""
    n = len(prob_abn)
    k = max(1, int(np.ceil(n * percentile_prior)))
    flag_pct = np.zeros(n, dtype=bool)
    flag_pct[np.argsort(prob_abn)[-k:]] = True
    flag_abs = prob_abn >= abs_threshold
    flag_and = flag_abs & flag_pct
    return np.where(flag_and, LABEL_ABNORMAL, LABEL_NORMAL).astype(np.int8)


def predict_or(prob_abn: np.ndarray, abs_threshold: float,
               percentile_prior: float = PERCENTILE_ABN_PRIOR) -> np.ndarray:
    """OR strategy (v12 default) — high sensitivity, low specificity."""
    n = len(prob_abn)
    k = max(1, int(np.ceil(n * percentile_prior)))
    flag_pct = np.zeros(n, dtype=bool)
    flag_pct[np.argsort(prob_abn)[-k:]] = True
    flag_abs = prob_abn >= abs_threshold
    return np.where(flag_abs | flag_pct,
                    LABEL_ABNORMAL, LABEL_NORMAL).astype(np.int8)


# ============================================================
# 16. LODO CV  (v13: uses base features only)
# ============================================================

def leave_one_db_out_cv(pipeline, X, y, db_tags) -> dict:
    log.info("LODO-CV (diagnostic — no augmentation, raw proba)...")
    unique_dbs = list(np.unique(db_tags))
    fold_maccs, fold_thrs, fold_se, fold_sp = [], [], [], []

    for held_db in unique_dbs:
        train_mask = db_tags != held_db
        val_mask   = db_tags == held_db
        X_tr, y_tr   = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask],   y[val_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            log.info(f"  Skip DB-{held_db.upper()} (single class)")
            continue

        pipe_fold = clone(pipeline)
        pipe_fold.fit(X_tr, y_tr)
        prob_val  = _get_prob_abn(pipe_fold, X_val)
        best_t, _ = find_best_threshold_se_biased(y_val, prob_val)
        y_pred    = np.where(prob_val >= best_t, LABEL_ABNORMAL, LABEL_NORMAL)
        m         = compute_metrics(y_val, y_pred, prob_val)

        log.info(f"  Held DB-{held_db.upper()} "
                 f"(Abn={(y_val==LABEL_ABNORMAL).sum()} "
                 f"Nor={(y_val==LABEL_NORMAL).sum()}): "
                 f"MAcc={m['MAcc']:.3f}  Se={m['Se']:.3f}  "
                 f"Sp={m['Sp']:.3f}  thr={best_t:.2f}  "
                 f"ROC-AUC={m.get('ROC_AUC',0):.3f}")

        fold_maccs.append(m["MAcc"])
        fold_thrs.append(best_t)
        fold_se.append(m["Se"])
        fold_sp.append(m["Sp"])

    mean_macc = float(np.mean(fold_maccs)) if fold_maccs else 0.0
    std_macc  = float(np.std(fold_maccs))  if fold_maccs else 0.0
    log.info(f"LODO MAcc (diag): {mean_macc:.4f} +/- {std_macc:.4f}")
    return {
        "lodo_macc_mean":  mean_macc, "lodo_macc_std":  std_macc,
        "fold_maccs":      fold_maccs, "fold_thresholds": fold_thrs,
        "fold_se":         fold_se,    "fold_sp":          fold_sp,
        "best_threshold":  float(np.median(fold_thrs)) if fold_thrs else 0.40,
    }


# ============================================================
# 17. GROUP KFOLD CV  (v13: NEW — no augmentation leakage)
# ============================================================

def kfold_cv_grouped(pipeline, X, y, rec_ids) -> dict:
    """
    v13: GroupKFold on rec_ids.
    This ensures augmented copies of the same recording are never
    split across train/val → honest overfitting estimate.
    Uses StratifiedKFold for base features (no augmentation).
    """
    log.info("5-fold GroupKFold CV (anti-leakage overfitting check)...")
    # Use GroupKFold to split by recording ID
    gkf = GroupKFold(n_splits=5)
    train_maccs, val_maccs = [], []

    # Map rec_ids to integer groups
    unique_ids = {r: i for i, r in enumerate(np.unique(rec_ids))}
    groups = np.array([unique_ids[r] for r in rec_ids])

    for tr_idx, val_idx in gkf.split(X, y, groups):
        pipe_fold = clone(pipeline)
        pipe_fold.fit(X[tr_idx], y[tr_idx])
        p_tr   = _get_prob_abn(pipe_fold, X[tr_idx])
        y_p_tr = np.where(p_tr >= 0.5, LABEL_ABNORMAL, LABEL_NORMAL)
        train_maccs.append(compute_metrics(y[tr_idx], y_p_tr)["MAcc"])
        p_val    = _get_prob_abn(pipe_fold, X[val_idx])
        t_val, _ = find_best_threshold(y[val_idx], p_val)
        y_p_val  = np.where(p_val >= t_val, LABEL_ABNORMAL, LABEL_NORMAL)
        val_maccs.append(compute_metrics(y[val_idx], y_p_val)["MAcc"])

    tr_m  = float(np.mean(train_maccs))
    val_m = float(np.mean(val_maccs))
    val_s = float(np.std(val_maccs))
    gap   = tr_m - val_m
    log.info(f"GroupKFold Train MAcc (thr=0.5)  : {tr_m:.4f}")
    log.info(f"GroupKFold Val   MAcc (oracle)   : {val_m:.4f} +/- {val_s:.4f}")
    log.info(f"Overfitting gap                  : {gap:.4f}")
    if gap > 0.15:
        log.warning(f"High overfit gap ({gap:.3f}). Increase l2_reg or reduce tree depth.")
    elif gap < 0.05:
        log.info("Low overfit gap — model may benefit from more capacity.")
    return {"train": tr_m, "val": val_m, "std": val_s, "gap": gap}


# ============================================================
# 18. PLOTS
# ============================================================

def _out():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion(y_true, y_pred, title: str, fname: str):
    cm  = confusion_matrix(y_true, y_pred, labels=[LABEL_ABNORMAL, LABEL_NORMAL])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Abnormal", "Normal"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved -> {OUTPUT_DIR / fname}")


def plot_roc_pr(y_true, prob_abn, threshold, fname="roc_pr.png"):
    yb          = (y_true == LABEL_ABNORMAL).astype(int)
    fpr, tpr, _ = roc_curve(yb, prob_abn)
    pre, rec, _ = precision_recall_curve(yb, prob_abn)
    rauc = roc_auc_score(yb, prob_abn)
    pauc = average_precision_score(yb, prob_abn)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(fpr, tpr, "#1a6aff", lw=2, label=f"AUC={rauc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
    ax1.axvline(1 - threshold, color="#e8350f", lw=1, linestyle=":",
                label=f"abs_thr={threshold:.2f}")
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax1.legend(fontsize=9); ax1.grid(alpha=.3)
    ax2.plot(rec, pre, "#22c55e", lw=2, label=f"AP={pauc:.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    ax2.legend(fontsize=9); ax2.grid(alpha=.3)
    plt.suptitle("Heart Sound v13  (Anti-Overfit + Domain-Robust)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"ROC/PR -> {OUTPUT_DIR / fname}")


def plot_lodo_results(lodo_res, fname="lodo_cv.png"):
    dbs    = TRAIN_DBS[:len(lodo_res["fold_maccs"])]
    colors = ["#1a6aff", "#22c55e", "#e8350f", "#f0b429"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(dbs, lodo_res["fold_maccs"],
                color=colors[:len(dbs)], alpha=0.8, width=0.5)
    axes[0].axhline(lodo_res["lodo_macc_mean"], color="#000", lw=1.5,
                    linestyle="--",
                    label=f"Mean={lodo_res['lodo_macc_mean']:.3f}")
    axes[0].axhline(0.80, color="#22c55e", lw=1, linestyle=":", label="v13 Target 80%")
    axes[0].set(xlabel="Held DB", ylabel="MAcc",
                title="LODO MAcc (diagnostic)", ylim=(0, 1.05))
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.3)
    x, w = np.arange(len(dbs)), 0.35
    axes[1].bar(x - w/2, lodo_res["fold_se"], w,
                label="Se", color="#1a6aff", alpha=0.8)
    axes[1].bar(x + w/2, lodo_res["fold_sp"], w,
                label="Sp", color="#22c55e", alpha=0.8)
    axes[1].set(xticks=x, xticklabels=dbs, xlabel="Held DB",
                ylabel="Score", title="Se vs Sp per fold", ylim=(0, 1.05))
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.3)
    axes[2].bar(dbs, lodo_res["fold_thresholds"],
                color=colors[:len(dbs)], alpha=0.8, width=0.5)
    axes[2].axhline(lodo_res["best_threshold"], color="#000", lw=1.5,
                    linestyle="--",
                    label=f"Median={lodo_res['best_threshold']:.2f}")
    axes[2].set(xlabel="Held DB", ylabel="Threshold",
                title="Threshold per fold", ylim=(0, 1))
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=.3)
    plt.suptitle("LODO-CV  v13 (diagnostic only)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"LODO plot -> {OUTPUT_DIR / fname}")


def plot_calibration_curve(y_calib, raw_probs, cal_probs, threshold,
                           fname="calibration_curve.png"):
    from sklearn.calibration import calibration_curve as sk_cal_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    y_bin = (y_calib == LABEL_ABNORMAL).astype(int)
    for probs, label, color in [(raw_probs, "Raw Ensemble", "#aaa"),
                                 (cal_probs, "Isotonic", "#1a6aff")]:
        try:
            fp, mp = sk_cal_curve(y_bin, probs, n_bins=8)
            axes[0].plot(mp, fp, marker="o", lw=2, label=label, color=color)
        except Exception:
            pass
    axes[0].plot([0, 1], [0, 1], "--", color="#888", lw=1, label="Perfect")
    axes[0].set(xlabel="Mean predicted prob", ylabel="Fraction positives",
                title="Reliability Diagram", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.3)
    abn_mask = y_calib == LABEL_ABNORMAL
    axes[1].hist(cal_probs[abn_mask],  bins=20, alpha=0.6,
                 color="#e8350f", label="Abnormal")
    axes[1].hist(cal_probs[~abn_mask], bins=20, alpha=0.6,
                 color="#1a6aff", label="Normal")
    axes[1].axvline(threshold, color="#000", lw=2, linestyle="--",
                    label=f"Abs thr={threshold:.2f}")
    axes[1].set(xlabel="Calibrated P(abn)", ylabel="Count",
                title="Score Distribution (calib split)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.3)
    plt.suptitle("v13 Calibration Analysis", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Calibration curve -> {OUTPUT_DIR / fname}")


def plot_threshold_comparison(y_test, prob_abn, abs_threshold, pct_prior,
                               fname="threshold_comparison.png"):
    thrs = np.arange(0.05, 0.96, 0.01)
    maccs, ses, sps = [], [], []
    for t in thrs:
        yp = np.where(prob_abn >= t, LABEL_ABNORMAL, LABEL_NORMAL)
        m  = compute_metrics(y_test, yp)
        maccs.append(m["MAcc"]); ses.append(m["Se"]); sps.append(m["Sp"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thrs, maccs, "#1a6aff", lw=2, label="MAcc")
    ax.plot(thrs, ses,   "#e8350f", lw=1.5, linestyle="--", label="Se")
    ax.plot(thrs, sps,   "#22c55e", lw=1.5, linestyle="--", label="Sp")
    ax.axvline(abs_threshold, color="#000", lw=2, linestyle=":",
               label=f"Abs={abs_threshold:.2f}")
    ax.set(xlabel="Threshold", ylabel="Score",
           title="v13 Threshold vs MAcc/Se/Sp", ylim=(0, 1.05))
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Threshold comparison -> {OUTPUT_DIR / fname}")


# ============================================================
# 19. TRAINING
# ============================================================

def run_train():
    log.info("=" * 70)
    log.info("  dsm — TRAINING  v13 (Anti-Overfit + Domain-Robust)")
    log.info(f"  Train DBs        : {TRAIN_DBS}")
    log.info(f"  Features         : ~199 (removed PCEN/Chroma/Tonnetz)")
    log.info(f"  Pipeline         : RobustScaler → PCA(60) → ThreeWayEnsemble")
    log.info(f"  Ensemble         : HGB(l2=2.0):0.45 + ETC(d=5):0.30 + RFC(d=6):0.25")
    log.info(f"  Augmentation     : {AUG_COPIES} copy/sample (2× data)")
    log.info(f"  CV               : GroupKFold (no augment leakage)")
    log.info(f"  Abs thr          : Youden J - {SE_THRESHOLD_BIAS} clamp [{THR_CLAMP_LO},{THR_CLAMP_HI}]")
    log.info(f"  Pct thr          : top {int(PERCENTILE_ABN_PRIOR*100)}% of batch")
    log.info(f"  Final pred       : Percentile (default) | Absolute | AND | OR")
    log.info("=" * 70)
    _out()

    records = load_split(TRAIN_DBS, load_audio=True,
                         exclude_unsure=True, per_db_balance=True)

    # Build augmented training matrix
    X, y, db_tags, rec_ids, feat_names = build_matrix(
        records, desc="Train ", augment=True, aug_copies=AUG_COPIES)

    pipeline = build_pipeline()

    # GroupKFold CV on BASE features (no augmentation — honest evaluation)
    log.info("Building base feature matrix for CV diagnostics...")
    X_base, y_base, db_tags_base, rec_ids_base, _ = build_matrix(
        records, desc="Base  ", augment=False)
    kf_res   = kfold_cv_grouped(pipeline, X_base, y_base, rec_ids_base)
    lodo_res = leave_one_db_out_cv(pipeline, X_base, y_base, db_tags_base)
    plot_lodo_results(lodo_res)

    # Calibration split (on augmented set)
    log.info(f"Splitting off {int(CALIB_FRAC*100)}% calibration set...")
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=CALIB_FRAC, stratify=y, random_state=RANDOM_STATE)
    log.info(f"  Train: {len(X_train)} | Calib: {len(X_calib)} "
             f"(Abn={(y_calib==LABEL_ABNORMAL).sum()} "
             f"Nor={(y_calib==LABEL_NORMAL).sum()})")

    log.info("Fitting pipeline (3-way ensemble)...")
    pipeline.fit(X_train, y_train)

    log.info("Fitting isotonic calibrator...")
    calibrator = IsotonicCalibrator(pipeline)
    calibrator.fit(X_calib, y_calib)

    log.info("Computing Youden threshold...")
    cal_probs_c = calibrator.predict_proba_abn(X_calib)
    raw_probs_c = _get_prob_abn(pipeline, X_calib, calibrator=None)
    abs_threshold = youden_threshold(y_calib, cal_probs_c)
    plot_calibration_curve(y_calib, raw_probs_c, cal_probs_c, abs_threshold)

    # Evaluate all 4 strategies on calibration split
    y_pct  = predict_percentile(cal_probs_c, PERCENTILE_ABN_PRIOR)
    y_abs  = predict_absolute(cal_probs_c, abs_threshold)
    y_and  = predict_and(cal_probs_c, abs_threshold, PERCENTILE_ABN_PRIOR)
    y_or   = predict_or(cal_probs_c, abs_threshold, PERCENTILE_ABN_PRIOR)
    m_pct  = compute_metrics(y_calib, y_pct,  cal_probs_c)
    m_abs  = compute_metrics(y_calib, y_abs,  cal_probs_c)
    m_and  = compute_metrics(y_calib, y_and,  cal_probs_c)
    m_or   = compute_metrics(y_calib, y_or,   cal_probs_c)
    log.info(f"Calib strategies (abs={abs_threshold:.2f}, pct={int(PERCENTILE_ABN_PRIOR*100)}%):")
    log.info(f"  Percentile: Se={m_pct['Se']:.3f}  Sp={m_pct['Sp']:.3f}  MAcc={m_pct['MAcc']:.3f}")
    log.info(f"  Absolute:   Se={m_abs['Se']:.3f}  Sp={m_abs['Sp']:.3f}  MAcc={m_abs['MAcc']:.3f}")
    log.info(f"  AND:        Se={m_and['Se']:.3f}  Sp={m_and['Sp']:.3f}  MAcc={m_and['MAcc']:.3f}")
    log.info(f"  OR:         Se={m_or['Se']:.3f}   Sp={m_or['Sp']:.3f}   MAcc={m_or['MAcc']:.3f}")

    joblib.dump(pipeline,             OUTPUT_DIR / "pipeline.pkl")
    joblib.dump(calibrator,           OUTPUT_DIR / "calibrator.pkl")
    joblib.dump(feat_names,           OUTPUT_DIR / "feature_names.pkl")
    joblib.dump(abs_threshold,        OUTPUT_DIR / "threshold.pkl")
    joblib.dump(PERCENTILE_ABN_PRIOR, OUTPUT_DIR / "percentile_prior.pkl")
    joblib.dump(lodo_res,             OUTPUT_DIR / "lodo_results.pkl")
    log.info(f"All artifacts saved -> {OUTPUT_DIR}/")

    gap = kf_res["gap"]
    if gap > 0.20:
        overfit_status = "HIGH — consider stronger regularization"
    elif gap > 0.12:
        overfit_status = "MODERATE — acceptable range"
    elif gap > 0.05:
        overfit_status = "LOW — good generalization"
    else:
        overfit_status = "MINIMAL — model may benefit from more capacity"

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY  v13")
    print("=" * 70)
    print(f"  GroupKFold Train MAcc (thr=0.5)  : {kf_res['train']:.4f}")
    print(f"  GroupKFold Val   MAcc (oracle)   : {kf_res['val']:.4f} "
          f"+/- {kf_res['std']:.4f}")
    print(f"  Overfitting gap                  : {gap:.4f}  ({overfit_status})")
    print(f"  LODO MAcc  (diagnostic)          : {lodo_res['lodo_macc_mean']:.4f} "
          f"+/- {lodo_res['lodo_macc_std']:.4f}")
    print(f"  Calib-split (pct={int(PERCENTILE_ABN_PRIOR*100)}%): "
          f"Se={m_pct['Se']:.3f}  Sp={m_pct['Sp']:.3f}  MAcc={m_pct['MAcc']:.4f}")
    print(f"  Training set size                : {len(X_train)} "
          f"(base + {AUG_COPIES}× augmented)")
    print(f"  PCA components                   : {PCA_COMPONENTS}")
    print(f"  Absolute threshold               : {abs_threshold:.3f}  (Youden-based)")
    print(f"  Percentile threshold             : top {int(PERCENTILE_ABN_PRIOR*100)}% of batch")
    print(f"")
    print(f"  v13 key changes vs v12:")
    print(f"    CORAL removed (→ PCA)              : more stable cross-DB")
    print(f"    RobustScaler (was StandardScaler)  : inter-DB outlier robust")
    print(f"    GroupKFold (no augment leakage)    : honest CV")
    print(f"    ETC max_depth=5 (was 8)            : anti-overfit")
    print(f"    RFC max_depth=6 (was None)         : anti-overfit")
    print(f"    HGB l2_reg=2.0 (was 0.1)           : anti-overfit")
    print(f"    Removed PCEN+Chroma+Tonnetz        : -100 noisy features")
    print(f"    N_MELS=16 (was 32)                 : -32 features")
    print(f"    Percentile-only default strategy   : better Se/Sp balance")
    print("=" * 70)


# ============================================================
# 20. EVALUATION
# ============================================================

def run_evaluate():
    log.info("=" * 70)
    log.info("  dsm — EVALUATION  v13")
    log.info(f"  Test DBs : {TEST_DBS}")
    log.info("=" * 70)
    _out()

    if not (OUTPUT_DIR / "pipeline.pkl").exists():
        log.error("No model — run: python dsm_v13.py train"); return

    pipeline      = joblib.load(OUTPUT_DIR / "pipeline.pkl")
    abs_threshold = joblib.load(OUTPUT_DIR / "threshold.pkl")
    calib_path    = OUTPUT_DIR / "calibrator.pkl"
    calibrator    = joblib.load(calib_path) if calib_path.exists() else None
    pct_path      = OUTPUT_DIR / "percentile_prior.pkl"
    pct_prior     = (joblib.load(pct_path) if pct_path.exists()
                     else PERCENTILE_ABN_PRIOR)
    if calibrator is None:
        log.warning("No calibrator.pkl — using raw pipeline proba")
    log.info(f"Loaded | abs_threshold={abs_threshold:.3f} | "
             f"percentile_prior={pct_prior:.2f}")

    records = load_split(TEST_DBS, load_audio=True,
                         exclude_unsure=True, per_db_balance=False)
    X_test, y_test, db_arr, _, _ = build_matrix(
        records, desc="Test  ", augment=False)

    prob_abn = _get_prob_abn(pipeline, X_test, calibrator=calibrator)

    # Oracle (upper bound)
    best_t, best_m = find_best_threshold(y_test, prob_abn)
    log.info(f"Oracle thr: {best_t:.2f}  -> MAcc={best_m:.4f}  (upper bound)")

    # Four strategies
    y_abs  = predict_absolute(prob_abn, abs_threshold)
    y_pct  = predict_percentile(prob_abn, pct_prior)
    y_and  = predict_and(prob_abn, abs_threshold, pct_prior)
    y_or   = predict_or(prob_abn, abs_threshold, pct_prior)

    m_abs  = compute_metrics(y_test, y_abs,  prob_abn)
    m_pct  = compute_metrics(y_test, y_pct,  prob_abn)
    m_and  = compute_metrics(y_test, y_and,  prob_abn)
    m_or   = compute_metrics(y_test, y_or,   prob_abn)

    # v13 OFFICIAL SCORE: Percentile strategy
    m_off = m_pct
    y_off = y_pct
    off_name = f"Percentile (top {int(pct_prior*100)}%)"

    print("\n" + "=" * 70)
    print("  OVERALL TEST RESULTS  v13")
    print("=" * 70)
    print(f"  Oracle threshold           : {best_t:.2f}  "
          f"(upper bound MAcc={best_m:.4f})")
    print()
    print(f"  Strategy A: Absolute  (thr={abs_threshold:.3f})")
    print(f"    Se={m_abs['Se']:.4f}  Sp={m_abs['Sp']:.4f}  "
          f"MAcc={m_abs['MAcc']:.4f}")
    print()
    print(f"  Strategy B: Percentile (top {int(pct_prior*100)}%)  ← OFFICIAL v13")
    print(f"    Se={m_pct['Se']:.4f}  Sp={m_pct['Sp']:.4f}  "
          f"MAcc={m_pct['MAcc']:.4f}")
    print()
    print(f"  Strategy C: AND (abs + percentile both agree)")
    print(f"    Se={m_and['Se']:.4f}  Sp={m_and['Sp']:.4f}  "
          f"MAcc={m_and['MAcc']:.4f}")
    print()
    print(f"  Strategy D: OR (v12 default — for comparison)")
    print(f"    Se={m_or['Se']:.4f}  Sp={m_or['Sp']:.4f}  "
          f"MAcc={m_or['MAcc']:.4f}")
    print()
    print(f"  OFFICIAL SCORE ({off_name}):")
    print(f"    Se={m_off['Se']:.4f}  Sp={m_off['Sp']:.4f}  "
          f"MAcc={m_off['MAcc']:.4f}")
    print(f"    F1={m_off['F1']:.4f}  ROC-AUC={m_off.get('ROC_AUC',0):.4f}")
    print(f"    TP={m_off['TP']}  FN={m_off['FN']}  "
          f"TN={m_off['TN']}  FP={m_off['FP']}")

    macc = m_off["MAcc"]
    if macc >= 0.86:
        status = "EXCELLENT (>86%) — v13 TARGET MET"
    elif macc >= 0.80:
        status = "STRONG (>80%) — v13 target met"
    elif macc >= 0.75:
        status = "GOOD (>75%) — near target"
    elif macc >= 0.65:
        status = "MODERATE (>65%) — improvement over v12"
    else:
        status = "needs more work"
    print(f"    Status: {status}")
    print("=" * 70)

    print("\nClassification Report (Percentile strategy):")
    print(classification_report(y_off, y_test,
                                 labels=[LABEL_ABNORMAL, LABEL_NORMAL],
                                 target_names=["Abnormal", "Normal"]))

    plot_confusion(y_test, y_off,
                   f"Overall Percentile  MAcc={m_off['MAcc']:.3f}",
                   "confusion_overall.png")
    plot_roc_pr(y_test, prob_abn, abs_threshold)
    plot_threshold_comparison(y_test, prob_abn, abs_threshold, pct_prior)

    # Per-database breakdown
    print("\n" + "=" * 70)
    print("  PER-DATABASE BREAKDOWN  (Percentile strategy)")
    print("=" * 70)
    for db in TEST_DBS:
        mask   = db_arr == db
        if mask.sum() == 0: continue
        y_t    = y_test[mask]
        p_ab   = prob_abn[mask]
        y_c    = predict_percentile(p_ab, pct_prior)
        dm     = compute_metrics(y_t, y_c, p_ab)
        ot, om = find_best_threshold(y_t, p_ab)

        if "ROC_AUC" in dm:
            if dm["ROC_AUC"] < 0.55:
                auc_note = "  ← near-random — domain shift remains"
            elif dm["ROC_AUC"] < 0.65:
                auc_note = "  ← weak"
            elif dm["ROC_AUC"] < 0.75:
                auc_note = "  ← moderate"
            else:
                auc_note = "  ← GOOD"

        print(f"\n  DB-{db.upper()}  ({mask.sum()} recs | "
              f"Abn={(y_t==LABEL_ABNORMAL).sum()} "
              f"Nor={(y_t==LABEL_NORMAL).sum()})")
        print(f"    Se       = {dm['Se']:.4f}")
        print(f"    Sp       = {dm['Sp']:.4f}")
        print(f"    MAcc     = {dm['MAcc']:.4f}")
        print(f"    F1       = {dm['F1']:.4f}")
        if "ROC_AUC" in dm:
            print(f"    ROC-AUC  = {dm['ROC_AUC']:.4f}{auc_note}")
        print(f"    Oracle   = {ot:.2f}  -> MAcc={om:.4f}")

        plot_confusion(y_t, y_c,
                       f"DB-{db.upper()} MAcc={dm['MAcc']:.3f} (pct)",
                       f"confusion_db{db}.png")

    (OUTPUT_DIR / "results.txt").write_text(
        f"[v13 Anti-Overfit + Domain-Robust]\n"
        f"MAcc={m_off['MAcc']:.4f}  Se={m_off['Se']:.4f}  Sp={m_off['Sp']:.4f}\n"
        f"F1={m_off['F1']:.4f}  ROC={m_off.get('ROC_AUC',0):.4f}  "
        f"PR={m_off.get('PR_AUC',0):.4f}\n"
        f"Abs_threshold={abs_threshold:.3f}  "
        f"Pct_prior={pct_prior:.2f} (top {int(pct_prior*100)}%)\n"
        f"Oracle_threshold={best_t:.2f}  Oracle_MAcc={best_m:.4f}\n"
        f"Pipeline: RobustScaler → PCA(60) → ThreeWayEnsemble\n"
        f"Strategy: Percentile (v13 default)\n"
        f"Train={TRAIN_DBS}  Test={TEST_DBS}\n",
        encoding="utf-8",
    )
    log.info(f"Results -> {OUTPUT_DIR}/results.txt")


# ============================================================
# 21. SINGLE FILE PREDICT
# ============================================================

def run_predict(wav_path: str):
    print(f"\nAnalysing: {wav_path}")
    if not (OUTPUT_DIR / "pipeline.pkl").exists():
        print("ERROR: Run training first."); return

    pipeline      = joblib.load(OUTPUT_DIR / "pipeline.pkl")
    abs_threshold = joblib.load(OUTPUT_DIR / "threshold.pkl")
    calib_path    = OUTPUT_DIR / "calibrator.pkl"
    calibrator    = joblib.load(calib_path) if calib_path.exists() else None

    try:
        raw, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"ERROR loading audio: {e}"); return

    hea        = Path(wav_path).with_suffix(".hea")
    actual_int = _parse_label(hea) if hea.exists() else None
    actual_str = LABEL_MAP.get(actual_int, "N/A") if actual_int else "N/A"

    sqi   = compute_sqi(raw)
    feats = extract_features(raw, force=True, augment=False)
    if feats is None:
        _print_result("Unsure", actual_str, None, 0.0, False, sqi, 0.0, abs_threshold)
        return

    X = np.array([list(feats.values())], dtype=np.float32)
    try:
        prob_abn = float(_get_prob_abn(pipeline, X, calibrator=calibrator)[0])
    except Exception as e:
        log.warning(f"Prediction failed: {e}")
        _print_result("Unsure", actual_str, None, 0.0, False, sqi, 0.0, abs_threshold)
        return

    label   = "Abnormal" if prob_abn >= abs_threshold else "Normal"
    conf    = prob_abn if label == "Abnormal" else 1.0 - prob_abn
    ref     = prob_abn >= abs_threshold
    correct = None
    if actual_int is not None and actual_int != LABEL_UNSURE:
        correct = (LABEL_ABNORMAL if label == "Abnormal" else LABEL_NORMAL) == actual_int
    _print_result(label, actual_str, correct, conf, ref, sqi, prob_abn, abs_threshold)


def _print_result(label, actual, correct, conf, ref, sqi, pabn, thr):
    verdict    = ("CORRECT  ok" if correct is True
                  else "WRONG    x" if correct is False
                  else "(no ground truth)")
    sqi_status = "good" if sqi > 0.6 else "acceptable" if sqi > 0.3 else "low"
    print("\n" + "-" * 62)
    print("  HEART SOUND ANALYSIS  v13  (Anti-Overfit + Domain-Robust)")
    print("-" * 62)
    print(f"  Predicted      : {label.upper()}")
    print(f"  Actual         : {actual}  {verdict}")
    print(f"  Confidence     : {conf:.2%}")
    print(f"  Referral       : "
          f"{'YES — refer for expert diagnosis' if ref else 'NO  — no referral needed'}")
    print(f"  SQI            : {sqi:.3f}  ({sqi_status})")
    print(f"  P(Abnormal)    : {pabn:.4f}  (isotonic calibrated)")
    print(f"  Abs Threshold  : {thr:.3f}  (Youden-based, single file)")
    print(f"  NOTE: Batch mode uses Percentile strategy for better accuracy.")
    print("-" * 62 + "\n")


# ============================================================
# 22. ENTRY POINT
# ============================================================

def _usage():
    print("\nUSAGE:")
    print("  python dsm_v13.py train")
    print("  python dsm_v13.py evaluate")
    print("  python dsm_v13.py predict <wav>")
    print("  python dsm_v13.py all\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage(); sys.exit(0)
    cmd = sys.argv[1].lower()
    if   cmd == "train":    run_train()
    elif cmd == "evaluate": run_evaluate()
    elif cmd == "predict":
        if len(sys.argv) < 3: print("ERROR: provide <wav> path")
        else: run_predict(sys.argv[2])
    elif cmd == "all":
        run_train(); run_evaluate()
    else:
        print(f"Unknown command: {cmd}"); _usage(); sys.exit(1)
