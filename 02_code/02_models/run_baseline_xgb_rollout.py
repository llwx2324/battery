# -*- coding: utf-8 -*-
"""
XGBoost baseline (monthly) + recursive rollout (2025-01 -> 2030-12)

Fixes:
- Pitfall #1: remove the OTHER target column from features (PEV model must not use CEV_number and vice versa)
- Pitfall #2: update rolling features during rollout, including generic names:
    rolling_mean_3 / rolling_sum_12 / rolling_std_6 / rolling_min_12 / rolling_max_12

Outputs:
- 03_runs/baseline_xgb/<run_id>/model.joblib
- 03_runs/baseline_xgb/<run_id>/feature_cols.json
- 03_runs/baseline_xgb/<run_id>/predictions_unified.csv   (Code,date,y_true,y_pred,split)
- 03_runs/baseline_xgb/<run_id>/calib_residuals.csv       (Code,date,y_true,y_pred,residual,abs_residual)
- 03_runs/baseline_xgb/<run_id>/run_config.json
"""

import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise RuntimeError(
        "未能导入 xgboost。请先执行：pip install -U xgboost\n"
        f"原始错误：{e}"
    )


# -----------------------------
# utils
# -----------------------------
def find_project_root(start: Path) -> Path:
    """向上寻找包含 01_data/02_code/03_runs 的项目根目录。"""
    cur = start.resolve()
    for _ in range(12):
        if (cur / "01_data").exists() and (cur / "02_code").exists() and (cur / "03_runs").exists():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        "找不到项目根目录（需要包含 01_data/02_code/03_runs）。"
        "请从 D:\\thesis\\thesis 下运行或把脚本放对位置。"
    )


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """确保存在 date 列（每月用该月1号表示）。"""
    if "date" not in df.columns:
        if "Year" in df.columns and "Month" in df.columns:
            df["date"] = pd.to_datetime(
                df["Year"].astype(int).astype(str) + "-" +
                df["Month"].astype(int).astype(str).str.zfill(2) + "-01"
            )
        else:
            raise KeyError("数据中找不到 date 或 Year/Month 列，无法构造时间索引。")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def select_feature_cols(df: pd.DataFrame, y_col: str) -> list:
    """
    自动选特征：
    - 去掉标识列/目标列
    - ✅ 去掉“另一个目标列”（避免未来不可得特征）
    - 只保留数值列
    """
    drop_cols = {y_col}
    for c in ["City", "Province", "CN_CITY", "Code", "date"]:
        if c in df.columns:
            drop_cols.add(c)

    # ✅ 关键：去掉另一目标列（PEV 模型不许用 CEV_number，反之亦然）
    other_y = None
    if y_col == "PEV_number":
        other_y = "CEV_number"
    elif y_col == "CEV_number":
        other_y = "PEV_number"
    if other_y and other_y in df.columns:
        drop_cols.add(other_y)

    cand = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    num_cols = cand.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        raise ValueError("自动选特征失败：没有任何数值列可用。请检查特征工程输出。")
    return num_cols


def parse_lag_cols(columns, y_col: str):
    """找出类似 PEV_number_lag_1 这种列。返回 {lag_k: colname}"""
    pat = re.compile(rf"^{re.escape(y_col)}_lag_(\d+)$")
    out = {}
    for c in columns:
        m = pat.match(c)
        if m:
            out[int(m.group(1))] = c
    return out


def parse_roll_cols(columns, y_col: str):
    """
    解析带 y_col 前缀的 rolling 列名，兼容：
      PEV_number_roll_mean_3
      PEV_number_rolling_mean_3
      PEV_number_roll_3_mean
    支持 stat: mean/std/sum/min/max
    返回 list[(stat, window, colname)]
    """
    outs = []

    p1 = re.compile(rf"^{re.escape(y_col)}_roll_(mean|std|sum|min|max)_(\d+)$")
    p2 = re.compile(rf"^{re.escape(y_col)}_rolling_(mean|std|sum|min|max)_(\d+)$")
    p3 = re.compile(rf"^{re.escape(y_col)}_roll_(\d+)_(mean|std|sum|min|max)$")

    for c in columns:
        m = p1.match(c)
        if m:
            stat, w = m.group(1), int(m.group(2))
            outs.append((stat, w, c))
            continue
        m = p2.match(c)
        if m:
            stat, w = m.group(1), int(m.group(2))
            outs.append((stat, w, c))
            continue
        m = p3.match(c)
        if m:
            w, stat = int(m.group(1)), m.group(2)
            outs.append((stat, w, c))
            continue

    return outs


def parse_generic_roll_cols(columns):
    """
    ✅ 解析通用 rolling 列名（不带 y_col 前缀）：
      rolling_mean_3 / rolling_sum_12 / rolling_std_6 / rolling_min_12 / rolling_max_12
    返回 list[(stat, window, colname)]
    """
    outs = []
    pat = re.compile(r"^rolling_(mean|std|sum|min|max)_(\d+)$")
    for c in columns:
        m = pat.match(c)
        if m:
            stat = m.group(1)
            w = int(m.group(2))
            outs.append((stat, w, c))
    return outs


def compute_roll_stats(arr: np.ndarray, stat: str):
    if arr.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr, ddof=0))
    if stat == "sum":
        return float(np.sum(arr))
    if stat == "min":
        return float(np.min(arr))
    if stat == "max":
        return float(np.max(arr))
    return np.nan


# -----------------------------
# main
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--y_col", type=str, default="PEV_number", help="目标列：PEV_number 或 CEV_number")
    parser.add_argument("--train_path", type=str, default=r"01_data/processed/dataset_train.parquet")
    parser.add_argument("--calib_path", type=str, default=r"01_data/processed/dataset_calib.parquet")
    parser.add_argument("--features_path", type=str, default=r"01_data/processed/features_monthly.parquet",
                        help="包含 2016-2030 全量特征的表（含外生特征 + lag/rolling列名）")
    parser.add_argument("--run_dir", type=str, default=r"03_runs/baseline_xgb", help="run 根目录")
    parser.add_argument("--seed", type=int, default=42)

    # 递推范围（建议固定）
    parser.add_argument("--rollout_start", type=str, default="2025-01-01")
    parser.add_argument("--rollout_end", type=str, default="2030-12-01")

    # XGB 参数（够用先跑通；后面可网格/optuna）
    parser.add_argument("--n_estimators", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)

    args = parser.parse_args()

    # project root
    root = find_project_root(Path(__file__).parent)
    train_path = root / args.train_path
    calib_path = root / args.calib_path
    features_path = root / args.features_path
    run_root = root / args.run_dir

    if not train_path.exists():
        raise FileNotFoundError(f"找不到：{train_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"找不到：{calib_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"找不到：{features_path}")

    # run id
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df_train = pd.read_parquet(train_path)
    df_calib = pd.read_parquet(calib_path)
    df_train = ensure_date(df_train)
    df_calib = ensure_date(df_calib)

    y_col = args.y_col
    if y_col not in df_train.columns:
        raise KeyError(f"训练集没有目标列 {y_col}，请检查 y_col 是否写错。")

    # -----------------------------
    # feature columns (fix pitfall #1)
    # -----------------------------
    feature_cols = select_feature_cols(df_train, y_col)

    # fit
    X_train = df_train[feature_cols]
    y_train = df_train[y_col].astype(float)

    X_calib = df_calib[feature_cols]
    y_calib = df_calib[y_col].astype(float)

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    # calib prediction + residuals
    calib_pred = model.predict(X_calib)
    calib_resid = y_calib.values - calib_pred
    calib_abs = np.abs(calib_resid)

    mae = mean_absolute_error(y_calib, calib_pred)
    rmse = float(np.sqrt(mean_squared_error(y_calib, calib_pred)))  # sklearn 旧版本兼容

    df_calib_out = pd.DataFrame({
        "Code": df_calib["Code"].astype(str) if "Code" in df_calib.columns else "",
        "date": df_calib["date"].dt.strftime("%Y-%m-%d"),
        "y_true": y_calib.values,
        "y_pred": calib_pred,
        "residual": calib_resid,
        "abs_residual": calib_abs,
    })
    df_calib_out.to_csv(out_dir / "calib_residuals.csv", index=False, encoding="utf-8-sig")

    # save model + feature cols + config
    dump(model, out_dir / "model.joblib")
    (out_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    run_cfg = {
        "run_id": run_id,
        "y_col": y_col,
        "train_path": str(train_path),
        "calib_path": str(calib_path),
        "features_path": str(features_path),
        "feature_cols_count": len(feature_cols),
        "calib_MAE": float(mae),
        "calib_RMSE": float(rmse),
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "seed": args.seed,
            "tree_method": "hist",
        },
        "rollout_start": args.rollout_start,
        "rollout_end": args.rollout_end,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # -----------------------------
    # recursive rollout: 2025-01 ~ 2030-12
    # -----------------------------
    df_feat = pd.read_parquet(features_path)
    df_feat = ensure_date(df_feat)

    if "Code" not in df_feat.columns:
        raise KeyError("features_monthly.parquet 必须包含 Code 列。")

    df_feat["Code"] = df_feat["Code"].astype(str)

    start = pd.to_datetime(args.rollout_start)
    end = pd.to_datetime(args.rollout_end)
    all_months = pd.date_range(start=start, end=end, freq="MS")  # month start

    # -----------------------------
    # columns to update (fix pitfall #2)
    # -----------------------------
    lag_cols = parse_lag_cols(df_feat.columns, y_col)

    # ✅ 合并两类 rolling：带前缀 + 通用 rolling_*
    roll_cols = parse_roll_cols(df_feat.columns, y_col) + parse_generic_roll_cols(df_feat.columns)

    # 只更新“确实会被模型用到的列”，避免无意义写入
    lag_cols = {k: v for k, v in lag_cols.items() if v in feature_cols}
    roll_cols = [(stat, w, c) for (stat, w, c) in roll_cols if c in feature_cols]

    # 如果训练用到某些列，但 features 表里没有，就先补齐
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan

    if y_col not in df_feat.columns:
        raise KeyError(f"features_monthly.parquet 没有 {y_col}，无法构建历史序列。")

    # 只保留需要的列（减少内存）
    keep_cols = ["Code", "date", y_col] + feature_cols
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df_feat.columns]))
    df_feat = df_feat[keep_cols].copy()
    df_feat.sort_values(["Code", "date"], inplace=True)

    # 建历史 y（真值到 2024-12 或最后一个非空）
    hist_by_code = {}
    for code, g in df_feat.groupby("Code", sort=False):
        g_obs = g.dropna(subset=[y_col])
        if len(g_obs) == 0:
            continue
        hist_by_code[code] = g_obs.sort_values("date")[y_col].astype(float).to_list()

    if len(hist_by_code) == 0:
        raise RuntimeError("没有任何城市有历史 y（真值），无法做递推。请检查 features_monthly 里 y_col 是否正确。")

    # rollout
    preds_rows = []

    for dt in all_months:
        month_df = df_feat[df_feat["date"] == dt].copy()
        if month_df.empty:
            raise RuntimeError(f"features 表里缺少月份：{dt.strftime('%Y-%m')} 的行，无法递推。")

        month_df["Code"] = month_df["Code"].astype(str)

        # 对每个城市更新 lag/rolling（用 hist_by_code 里的真值+预测值）
        for i, row in month_df.iterrows():
            code = row["Code"]
            if code not in hist_by_code:
                continue
            hist = hist_by_code[code]

            # lag
            for k, colname in lag_cols.items():
                month_df.at[i, colname] = hist[-k] if len(hist) >= k else np.nan

            # rolling
            for stat, w, colname in roll_cols:
                window_vals = np.array(hist[-w:], dtype=float) if len(hist) >= 1 else np.array([], dtype=float)
                month_df.at[i, colname] = compute_roll_stats(window_vals, stat)

        # 预测
        X_month = month_df[feature_cols]
        y_pred = model.predict(X_month)

        # 写回历史（递推核心）
        for code, pred_val in zip(month_df["Code"].tolist(), y_pred.tolist()):
            if code in hist_by_code:
                hist_by_code[code].append(float(pred_val))

        # 输出当月
        preds_rows.append(pd.DataFrame({
            "Code": month_df["Code"].values,
            "date": month_df["date"].dt.strftime("%Y-%m-%d").values,
            "y_true": month_df[y_col].values,  # 未来通常 NaN
            "y_pred": y_pred,
            "split": "future_rollout",
        }))

    df_future = pd.concat(preds_rows, ignore_index=True)

    # -----------------------------
    # unified predictions: train + calib + future_rollout
    # -----------------------------
    train_pred = model.predict(df_train[feature_cols])
    df_train_out = pd.DataFrame({
        "Code": df_train["Code"].astype(str) if "Code" in df_train.columns else "",
        "date": df_train["date"].dt.strftime("%Y-%m-%d"),
        "y_true": df_train[y_col].astype(float).values,
        "y_pred": train_pred,
        "split": "train",
    })

    df_calib_pred_out = pd.DataFrame({
        "Code": df_calib["Code"].astype(str) if "Code" in df_calib.columns else "",
        "date": df_calib["date"].dt.strftime("%Y-%m-%d"),
        "y_true": df_calib[y_col].astype(float).values,
        "y_pred": calib_pred,
        "split": "calib",
    })

    df_unified = pd.concat([df_train_out, df_calib_pred_out, df_future], ignore_index=True)
    df_unified.sort_values(["Code", "date"], inplace=True)
    df_unified.to_csv(out_dir / "predictions_unified.csv", index=False, encoding="utf-8-sig")

    # -----------------------------
    # quick sanity prints
    # -----------------------------
    print("\n[OK] baseline_xgb + rollout 完成")
    print(f"run_dir: {out_dir}")
    print(f"calib MAE : {mae:.6f}")
    print(f"calib RMSE: {rmse:.6f}")
    print(f"feature_cols_count: {len(feature_cols)}")
    print(f"unified predictions rows: {len(df_unified)}")
    print(f"future rows: {len(df_future)}  (months={len(all_months)} * cities≈{df_future['Code'].nunique()})")

    # 检查：另一目标列是否被剔除
    if y_col == "PEV_number" and "CEV_number" in feature_cols:
        print("[WARN] feature_cols 仍包含 CEV_number（不应发生）")
    if y_col == "CEV_number" and "PEV_number" in feature_cols:
        print("[WARN] feature_cols 仍包含 PEV_number（不应发生）")

    # 检查：如果 rolling_* 在特征里，看看未来首月缺失率（越低越好）
    rolling_in_use = [c for c in feature_cols if c.startswith("rolling_") or (y_col in c and ("roll_" in c or "rolling_" in c))]
    if len(rolling_in_use) > 0:
        first_month_df = df_feat[df_feat["date"] == all_months[0]].copy()
        nan_rate = float(first_month_df[rolling_in_use].isna().mean().mean())
        print(f"[CHK] rollout first month rolling features NaN rate: {nan_rate:.6f} (only for rolling cols used by model)")


if __name__ == "__main__":
    main()
