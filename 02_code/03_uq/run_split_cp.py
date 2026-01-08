# -*- coding: utf-8 -*-
"""
Split Conformal Prediction (Split CP) for point forecasts.

Inputs (from a baseline run dir):
- predictions_unified.csv  columns: Code,date,y_true,y_pred,split
- calib_residuals.csv      columns: Code,date,y_true,y_pred,residual,abs_residual

Outputs (to same run dir):
- predictions_with_cp.csv  adds pi_lower, pi_upper
- cp_summary.json          qhat, alpha, calib coverage, width stats

Usage:
python 02_code\\03_uq\\run_split_cp.py --run_dir 03_runs\\baseline_xgb\\20260101_211256 --alpha 0.1
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "01_data").exists() and (cur / "02_code").exists() and (cur / "03_runs").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError("找不到项目根目录（需要包含 01_data/02_code/03_runs）。")


def ensure_date_str(df: pd.DataFrame) -> pd.DataFrame:
    # 统一 date 为 YYYY-MM-DD 字符串（不强制转 datetime，避免格式差异）
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def infer_files(run_dir: Path, pred_path: Path, calib_path: Path):
    """
    允许用户传错文件名：通过列名判断哪个是哪个。
    """
    df_a = pd.read_csv(pred_path)
    df_b = pd.read_csv(calib_path)

    a_cols = set(df_a.columns)
    b_cols = set(df_b.columns)

    is_calib_a = ("abs_residual" in a_cols) or ("residual" in a_cols)
    is_calib_b = ("abs_residual" in b_cols) or ("residual" in b_cols)

    is_pred_a = ("split" in a_cols) and ("y_pred" in a_cols)
    is_pred_b = ("split" in b_cols) and ("y_pred" in b_cols)

    if is_calib_a and is_pred_b:
        df_calib, df_pred = df_a, df_b
    elif is_calib_b and is_pred_a:
        df_calib, df_pred = df_b, df_a
    else:
        # fallback：按文件名猜（更严格也行，但这里尽量不阻塞你）
        df_pred = df_a
        df_calib = df_b

    # 基本校验
    need_pred = {"Code", "date", "y_pred", "split"}
    need_calib = {"Code", "date", "y_true", "y_pred"}

    if not need_pred.issubset(set(df_pred.columns)):
        raise ValueError(f"无法识别 predictions_unified：缺少列 {need_pred - set(df_pred.columns)}")
    if not need_calib.issubset(set(df_calib.columns)):
        raise ValueError(f"无法识别 calib_residuals：缺少列 {need_calib - set(df_calib.columns)}")

    # 如果 calib 没 abs_residual，就自己算
    if "abs_residual" not in df_calib.columns:
        if "residual" in df_calib.columns:
            df_calib["abs_residual"] = df_calib["residual"].abs()
        else:
            df_calib["abs_residual"] = (df_calib["y_true"] - df_calib["y_pred"]).abs()

    df_pred = ensure_date_str(df_pred)
    df_calib = ensure_date_str(df_calib)

    return df_pred, df_calib


def conformal_qhat(abs_residuals: np.ndarray, alpha: float) -> float:
    """
    Split CP 的有限样本分位数（常用/论文可写版本）：
    qhat = sorted_abs[ ceil((n+1)*(1-alpha)) - 1 ]
    """
    abs_residuals = np.asarray(abs_residuals, dtype=float)
    abs_residuals = abs_residuals[~np.isnan(abs_residuals)]
    n = abs_residuals.size
    if n == 0:
        raise ValueError("校准集 abs_residual 全是 NaN，无法计算 qhat。")

    s = np.sort(abs_residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))  # 1..n+1
    k = min(max(k, 1), n)                    # clamp to 1..n
    return float(s[k - 1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=r"03_runs\baseline_xgb\20260101_211256",
                        help="例如：03_runs/baseline_xgb/20260101_211256")
    parser.add_argument("--pred_path", type=str, default=None, help="可选：手动指定 predictions_unified.csv")
    parser.add_argument("--calib_path", type=str, default=None, help="可选：手动指定 calib_residuals.csv")
    parser.add_argument("--alpha", type=float, default=0.1, help="显著性水平 alpha，例如 0.1 -> 90%区间")
    parser.add_argument("--clip0", action="store_true", help="是否将下界截断到 0（计数任务建议开）")
    args = parser.parse_args()

    root = find_project_root(Path(__file__).parent)
    run_dir = (root / args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到 run_dir：{run_dir}")

    pred_path = Path(args.pred_path).resolve() if args.pred_path else (run_dir / "predictions_unified.csv")
    calib_path = Path(args.calib_path).resolve() if args.calib_path else (run_dir / "calib_residuals.csv")

    if not pred_path.exists():
        raise FileNotFoundError(f"找不到：{pred_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"找不到：{calib_path}")

    df_pred, df_calib = infer_files(run_dir, pred_path, calib_path)

    alpha = float(args.alpha)
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须在 (0,1) 内。")

    qhat = conformal_qhat(df_calib["abs_residual"].values, alpha)

    # 给 unified 加区间
    df_out = df_pred.copy()
    df_out["pi_lower"] = df_out["y_pred"] - qhat
    df_out["pi_upper"] = df_out["y_pred"] + qhat
    if args.clip0:
        df_out["pi_lower"] = df_out["pi_lower"].clip(lower=0.0)

    df_out["alpha"] = alpha
    df_out["qhat"] = qhat

    out_csv = run_dir / "predictions_with_cp.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 计算 calib 覆盖率（用 df_pred 里的 calib split；更贴近“区间最终应用对象”）
    df_calib_pred = df_out[df_out["split"] == "calib"].copy()
    calib_cov = None
    if "y_true" in df_calib_pred.columns:
        tmp = df_calib_pred.dropna(subset=["y_true", "pi_lower", "pi_upper"])
        if len(tmp) > 0:
            calib_cov = float(((tmp["y_true"] >= tmp["pi_lower"]) & (tmp["y_true"] <= tmp["pi_upper"])).mean())

    width = (df_out["pi_upper"] - df_out["pi_lower"]).astype(float)
    summary = {
        "run_dir": str(run_dir),
        "alpha": alpha,
        "qhat": qhat,
        "n_calib_abs_residual": int(df_calib["abs_residual"].dropna().shape[0]),
        "calib_coverage_on_unified_calib_split": calib_cov,
        "interval_width_mean": float(width.mean()),
        "interval_width_median": float(width.median()),
        "interval_width_p90": float(width.quantile(0.9)),
        "clip0": bool(args.clip0),
        "files": {
            "predictions_with_cp": str(out_csv),
        }
    }
    (run_dir / "cp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Split CP done")
    print("run_dir:", run_dir)
    print("alpha:", alpha)
    print("qhat:", qhat)
    print("calib coverage (on unified calib split):", calib_cov)
    print("output:", out_csv)


if __name__ == "__main__":
    main()
