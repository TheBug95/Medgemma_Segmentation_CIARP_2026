# Metrics generation on the results from inference
"""
Evaluacion robusta de clasificacion binaria con test set pequeno (n=26).

Metodos:
  - MÉTODO 1: Bootstrap estratificado (IC por percentiles) -> remuestrea con
    reemplazo DENTRO de cada clase, manteniendo 12 normal / 14 glaucoma.
  - MÉTODO 2: Clopper-Pearson (IC exacto para proporciones; sens/spec/acc/ppv/npv).

REPLICABILIDAD: el `seed` controla el remuestreo del bootstrap. Clopper-Pearson es
exacto/determinista (no usa seed).

Uso con datos REALES (desde el notebook de orquestacion):
    from metrics_generation import compute_all_metrics, print_report
    res = compute_all_metrics(y_true, y_proba, y_pred, n_bootstrap=5000, ci=0.95, seed=42)
    print_report(res)

Convencion de clases: 0 = normal, 1 = glaucoma (positivo).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from scipy import stats


# =============================================================================
# Metricas puntuales (sobre el test completo)
# =============================================================================
def point_metrics(y_true, y_proba, y_pred) -> dict:
    """Metricas puntuales sobre el test completo (sin IC)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "auc": float(roc_auc_score(y_true, y_proba)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
    }


# =============================================================================
# MÉTODO 1: BOOTSTRAP ESTRATIFICADO
# =============================================================================
def stratified_bootstrap_metrics(
    y_true, y_proba, y_pred, n_bootstrap: int = 5000, ci: float = 0.95, seed: int = 42
) -> dict:
    """
    Bootstrap estratificado: en cada remuestra mantiene la proporcion de clases
    (muestreo con reemplazo dentro de normales y dentro de glaucoma). Estandar en
    evaluacion medica con n pequeno. `seed` -> replicable.

    Returns:
        {metrica: {mean, std, median, ci_lower, ci_upper, n_valid}} para
        auc/accuracy/sensitivity/specificity/f1/ppv/npv.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)

    rng = np.random.RandomState(seed)
    idx_normal = np.where(y_true == 0)[0]
    idx_glaucoma = np.where(y_true == 1)[0]

    keys = ("auc", "accuracy", "sensitivity", "specificity", "f1", "ppv", "npv")
    metrics: dict[str, list] = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        boot_normal = rng.choice(idx_normal, size=len(idx_normal), replace=True)
        boot_glaucoma = rng.choice(idx_glaucoma, size=len(idx_glaucoma), replace=True)
        boot_idx = np.concatenate([boot_normal, boot_glaucoma])

        yt, yp, ypr = y_true[boot_idx], y_pred[boot_idx], y_proba[boot_idx]
        if len(np.unique(yt)) < 2:  # AUC requiere ambas clases
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        metrics["auc"].append(roc_auc_score(yt, ypr))
        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["sensitivity"].append(recall_score(yt, yp, pos_label=1, zero_division=0))
        metrics["specificity"].append(recall_score(yt, yp, pos_label=0, zero_division=0))
        metrics["f1"].append(f1_score(yt, yp, pos_label=1, zero_division=0))
        metrics["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        metrics["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)

    alpha = 1 - ci
    results = {}
    for key, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        results[key] = {
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)) if arr.size else float("nan"),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))) if arr.size else float("nan"),
            "n_valid": int(arr.size),
        }
    return results


# =============================================================================
# MÉTODO 2: CLOPPER-PEARSON (exacto, determinista)
# =============================================================================
def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Intervalo de confianza exacto de Clopper-Pearson para una proporcion k/n."""
    if n == 0:
        return (0.0, 0.0)
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lower, upper)


def clopper_pearson_metrics(y_true, y_pred, alpha: float = 0.05) -> dict:
    """Sens/spec/acc/ppv/npv con su IC exacto de Clopper-Pearson (k/n + intervalo)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_total = int(tn + fp + fn + tp)
    return {
        "sensitivity": {"k": int(tp), "n": int(tp + fn), "ci": clopper_pearson_ci(tp, tp + fn, alpha)},
        "specificity": {"k": int(tn), "n": int(tn + fp), "ci": clopper_pearson_ci(tn, tn + fp, alpha)},
        "accuracy": {"k": int(tp + tn), "n": n_total, "ci": clopper_pearson_ci(tp + tn, n_total, alpha)},
        "ppv": {"k": int(tp), "n": int(tp + fp), "ci": clopper_pearson_ci(tp, tp + fp, alpha)},
        "npv": {"k": int(tn), "n": int(tn + fn), "ci": clopper_pearson_ci(tn, tn + fn, alpha)},
    }


# =============================================================================
# Orquestador + reporte
# =============================================================================
def compute_all_metrics(
    y_true, y_proba, y_pred, n_bootstrap: int = 5000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Metricas puntuales + Bootstrap estratificado + Clopper-Pearson, en un dict
    JSON-serializable. `seed` fija el remuestreo del bootstrap (replicable)."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "point": point_metrics(y_true, y_proba, y_pred),
        "bootstrap": stratified_bootstrap_metrics(y_true, y_proba, y_pred, n_bootstrap, ci, seed),
        "clopper_pearson": clopper_pearson_metrics(y_true, y_pred, alpha=1 - ci),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_total": int(len(y_true)),
        "n_glaucoma": int((y_true == 1).sum()),
        "n_normal": int((y_true == 0).sum()),
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "ci": float(ci),
    }


def print_report(res: dict) -> None:
    """Imprime un reporte legible de compute_all_metrics."""
    print("=" * 72)
    print(f"EVALUACION TEST (n={res['n_total']}: {res['n_normal']} normal, {res['n_glaucoma']} glaucoma)")
    print(f"seed={res['seed']} | bootstrap={res['n_bootstrap']} | CI={res['ci']:.0%}")
    print("=" * 72)
    print(f"Matriz de confusion [[TN, FP], [FN, TP]] = {res['confusion_matrix']}")
    print(f"\nMÉTODO 1 - Bootstrap estratificado ({res['n_bootstrap']} remuestras)")
    print(f"{'Metrica':13s}{'Puntual':>10s}    mean [95% CI bootstrap]")
    print("-" * 72)
    for m in ("auc", "accuracy", "sensitivity", "specificity", "f1", "ppv", "npv"):
        p = res["point"][m]
        b = res["bootstrap"][m]
        print(f"{m.upper():13s}{p:>10.4f}    {b['mean']:.4f} [{b['ci_lower']:.4f} - {b['ci_upper']:.4f}]")
    print("\nMÉTODO 2 - Clopper-Pearson (IC exacto)")
    print("-" * 72)
    for m, d in res["clopper_pearson"].items():
        lo, hi = d["ci"]
        rate = d["k"] / d["n"] if d["n"] else float("nan")
        print(f"  {m:12s}: {d['k']}/{d['n']} = {rate:.4f} [{lo:.4f} - {hi:.4f}]")


if __name__ == "__main__":
    # Demo con datos simulados (REEMPLAZAR por las salidas reales del modelo).
    _rng = np.random.RandomState(42)
    _y_true = np.array([0] * 12 + [1] * 14)
    _y_proba = np.concatenate([_rng.beta(2, 5, 12), _rng.beta(5, 2, 14)])
    _y_pred = (_y_proba > 0.5).astype(int)
    print_report(compute_all_metrics(_y_true, _y_proba, _y_pred, n_bootstrap=5000, seed=42))
