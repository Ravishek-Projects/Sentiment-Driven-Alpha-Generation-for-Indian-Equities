# statistical_tests.py
# Statistical significance tests for evaluating model performance.
#
# Tests implemented
# ─────────────────
# 1. Bootstrap confidence intervals — non-parametric CI for any metric
#    (Sharpe, hit rate, IC). Handles fat-tailed financial return distributions.
#
# 2. Deflated Sharpe Ratio (DSR) — López de Prado (2018).
#    Adjusts the Sharpe ratio for multiple-testing bias.
#    When you test 4 model variants (text-only, text+tech, full, ensemble),
#    one of them will look good by chance. DSR accounts for this.
#
# 3. Diebold-Mariano (DM) test — tests whether model A's forecast errors
#    are statistically different from model B's.
#    H₀: equal predictive accuracy. p < 0.05 → reject H₀ → A ≠ B.
#
# 4. Paired t-test on hit rates across walk-forward folds.
#
# Usage
#   python statistical_tests.py
#   python statistical_tests.py --input eval_results\walk_forward_results.csv

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_phase5 import (
    EVAL_DIR, BOOTSTRAP_N_SAMPLES, CONFIDENCE_LEVEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stat_tests")


# ---------------------------------------------------------------------------
# 1. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn,
    n_samples: int = BOOTSTRAP_N_SAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    random_seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data         : 1-D array of observations (e.g. daily returns or per-trade P&L)
    statistic_fn : function mapping array → scalar (e.g. np.mean, sharpe_fn)
    n_samples    : number of bootstrap resamples
    confidence   : confidence level (default 0.95)

    Returns
    -------
    dict with keys: observed, lower, upper, se
    """
    rng      = np.random.default_rng(random_seed)
    observed = statistic_fn(data)
    boot_stats = np.array([
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_samples)
    ])
    alpha = (1 - confidence) / 2
    lower = float(np.nanpercentile(boot_stats, alpha * 100))
    upper = float(np.nanpercentile(boot_stats, (1 - alpha) * 100))
    se    = float(np.nanstd(boot_stats, ddof=1))
    return {
        "observed": float(observed),
        "lower":    lower,
        "upper":    upper,
        "se":       se,
        "ci_pct":   int(confidence * 100),
    }


# ---------------------------------------------------------------------------
# 2. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio — López de Prado (2018).
    Corrects for the bias introduced by multiple testing (trying many model
    configurations and reporting the best).

    Parameters
    ----------
    sharpe_observed : Sharpe ratio of the best model (annualised)
    n_trials        : number of models / hyperparameter configurations tried
    n_observations  : number of independent observations (trading days)
    skewness        : skewness of strategy returns (0 = symmetric)
    kurtosis        : excess kurtosis of strategy returns (0 = Gaussian)

    Returns
    -------
    float : probability that the observed Sharpe is genuine (0 → 1)
            Values > 0.95 indicate statistical significance at 5% level.
    """
    if n_trials < 1 or n_observations < 2:
        return 0.0

    # Expected maximum Sharpe under H₀ (random strategies)
    # Approximation from Ledoit & Wolf (2008) via De Prado
    gamma  = 0.5772156649   # Euler–Mascheroni constant
    E_max  = (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma     * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Adjusted Sharpe (accounts for non-normality of returns)
    sr_star = E_max * np.sqrt(
        (1 - skewness * sharpe_observed
         + (kurtosis - 1) / 4 * sharpe_observed ** 2)
        / (n_observations - 1)
    )

    # DSR = P(SR_obs > SR_benchmark)
    dsr = float(stats.norm.cdf(
        (sharpe_observed - sr_star)
        * np.sqrt(n_observations - 1)
        / np.sqrt(1 - skewness * sharpe_observed
                  + (kurtosis - 1) / 4 * sharpe_observed ** 2)
    ))
    return max(0.0, min(1.0, dsr))


# ---------------------------------------------------------------------------
# 3. Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H₀: E[d_t] = 0, where d_t = loss(e_a_t) - loss(e_b_t)
    Uses squared error loss: loss(e) = e².
    A negative DM statistic means model A has smaller errors than model B.

    Parameters
    ----------
    errors_a : forecast errors from model A (predicted - actual)
    errors_b : forecast errors from model B
    h        : forecast horizon (1 for one-step-ahead)

    Returns
    -------
    dict with: dm_stat, p_value, conclusion
    """
    d = errors_a ** 2 - errors_b ** 2    # loss differential
    n = len(d)
    if n < 2:
        return {"dm_stat": np.nan, "p_value": np.nan, "conclusion": "insufficient data"}

    d_bar   = d.mean()
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    gamma_0 = np.var(d, ddof=1)
    # Newey-West HAC variance estimate for h-step ahead forecast
    nw_var  = gamma_0
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1] if lag < n else 0
        nw_var   += 2 * (1 - lag / h) * gamma_lag

    se      = np.sqrt(max(nw_var, 1e-12) / n)
    dm_stat = float(d_bar / se)

    # HLN correction factor
    k        = ((n + 1 - 2 * h + h * (h - 1) / n) / n) ** 0.5
    dm_hln   = dm_stat * k
    p_value  = float(2 * stats.t.sf(abs(dm_hln), df=n - 1))

    if p_value < 0.05:
        winner  = "A (LSTM)" if dm_stat < 0 else "B (Transformer)"
        concl   = f"Reject H₀ (p={p_value:.4f}) — {winner} is significantly better"
    else:
        concl   = f"Fail to reject H₀ (p={p_value:.4f}) — no significant difference"

    return {
        "dm_stat":    round(dm_hln, 4),
        "p_value":    round(p_value, 4),
        "conclusion": concl,
    }


# ---------------------------------------------------------------------------
# 4. Paired t-test on walk-forward hit rates
# ---------------------------------------------------------------------------

def paired_hit_rate_test(
    hit_rates_a: np.ndarray,
    hit_rates_b: np.ndarray,
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> dict:
    """
    Paired t-test: are the per-fold hit rates significantly different?
    Used to compare e.g. ensemble vs LSTM across walk-forward folds.
    """
    diff    = hit_rates_a - hit_rates_b
    t_stat, p_value = stats.ttest_rel(hit_rates_a, hit_rates_b)
    return {
        "label_a":    label_a,
        "label_b":    label_b,
        "mean_a":     float(np.mean(hit_rates_a)),
        "mean_b":     float(np.mean(hit_rates_b)),
        "mean_diff":  float(diff.mean()),
        "t_stat":     round(float(t_stat), 4),
        "p_value":    round(float(p_value), 4),
        "significant":p_value < 0.05,
        "conclusion": (
            f"{label_a} significantly better (p={p_value:.4f})"
            if (p_value < 0.05 and diff.mean() > 0)
            else (
                f"{label_b} significantly better (p={p_value:.4f})"
                if (p_value < 0.05 and diff.mean() < 0)
                else f"No significant difference (p={p_value:.4f})"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Full test battery — run on walk-forward results CSV
# ---------------------------------------------------------------------------

def run_all_tests(results_csv: Path) -> dict:
    """
    Load walk-forward results and run all statistical tests.
    Returns a nested dict of all test results.
    """
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Walk-forward results not found: {results_csv}\n"
            "Run walk_forward_eval.py first."
        )

    df = pd.read_csv(results_csv)
    log.info(f"[stat] Loaded {len(df)} folds from {results_csv}")

    output = {}

    # ── 1. Bootstrap CIs for ensemble hit rate ───────────────────────────
    hit_ens = df["hit_rate_ens"].dropna().values
    if len(hit_ens) >= 5:
        ci = bootstrap_ci(hit_ens, np.mean)
        output["bootstrap_hit_rate_ensemble"] = ci
        log.info(
            f"[stat] Ensemble hit rate: {ci['observed']:.4f} "
            f"[{ci['lower']:.4f}, {ci['upper']:.4f}] {ci['ci_pct']}% CI"
        )

    # Bootstrap CI for IC
    ic_ens = df["ic_ens"].dropna().values
    if len(ic_ens) >= 5:
        ci_ic = bootstrap_ci(ic_ens, np.mean)
        output["bootstrap_ic_ensemble"] = ci_ic
        log.info(
            f"[stat] Ensemble IC:       {ci_ic['observed']:.4f} "
            f"[{ci_ic['lower']:.4f}, {ci_ic['upper']:.4f}] {ci_ic['ci_pct']}% CI"
        )

    # ── 2. Deflated Sharpe Ratio ─────────────────────────────────────────
    # Compute from portfolio_snapshots if available
    try:
        import duckdb
        from config_phase5 import DB_PATH
        con = duckdb.connect(str(DB_PATH), read_only=True)
        snap = con.execute(
            "SELECT daily_return FROM portfolio_snapshots ORDER BY snapshot_date"
        ).df()
        con.close()

        if len(snap) > 10:
            r        = snap["daily_return"].dropna().values
            ann_sr   = r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252))
            skew     = float(pd.Series(r).skew())
            kurt     = float(pd.Series(r).kurtosis())  # excess kurtosis
            n_trials = 4   # text-only, text+tech, tech-only, full (Phase 3 ablation)

            dsr = deflated_sharpe_ratio(ann_sr, n_trials, len(r), skew, kurt + 3)
            output["deflated_sharpe"] = {
                "observed_sharpe": round(float(ann_sr), 4),
                "n_trials":        n_trials,
                "n_observations":  len(r),
                "skewness":        round(skew, 4),
                "excess_kurtosis": round(kurt, 4),
                "dsr_probability": round(dsr, 4),
                "significant":     dsr > CONFIDENCE_LEVEL,
            }
            log.info(
                f"[stat] Deflated Sharpe: SR={ann_sr:.4f}  "
                f"DSR_prob={dsr:.4f}  significant={dsr > CONFIDENCE_LEVEL}"
            )
    except Exception as exc:
        log.warning(f"[stat] DSR skipped (no portfolio_snapshots): {exc}")

    # ── 3. Diebold-Mariano: LSTM vs Transformer ──────────────────────────
    # Approximate errors from RMSE (we don't have per-sample errors stored,
    # so we simulate using fold RMSE and sample counts as weights)
    if "rmse_lstm" in df.columns and "rmse_tf" in df.columns:
        # Fold-level RMSEs used as per-fold error representatives
        e_lstm = df["rmse_lstm"].dropna().values
        e_tf   = df["rmse_tf"].dropna().values
        if len(e_lstm) >= 5 and len(e_tf) >= 5:
            dm = diebold_mariano_test(e_lstm, e_tf)
            output["diebold_mariano_lstm_vs_tf"] = dm
            log.info(f"[stat] DM test (LSTM vs TF): {dm['conclusion']}")

    # ── 4. Paired t-test: ensemble vs best single model ──────────────────
    if "hit_rate_ens" in df.columns:
        for col, label in [("hit_rate_lstm", "LSTM"), ("hit_rate_tf", "Transformer")]:
            a  = df["hit_rate_ens"].dropna().values
            b  = df[col].dropna().values
            n  = min(len(a), len(b))
            if n >= 5:
                t_res = paired_hit_rate_test(a[:n], b[:n], "Ensemble", label)
                output[f"paired_t_ensemble_vs_{label.lower()}"] = t_res
                log.info(f"[stat] Paired t (Ensemble vs {label}): {t_res['conclusion']}")

    # Save results
    out_path = EVAL_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"[stat] Results saved → {out_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument(
        "--input",
        default=str(EVAL_DIR / "walk_forward_results.csv"),
        help="Path to walk_forward_results.csv",
    )
    args = parser.parse_args()
    run_all_tests(Path(args.input))
