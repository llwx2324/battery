# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_PATH = PROJECT_ROOT / "01_data" / "processed" / "features_monthly.parquet"
OUT_DIR = PROJECT_ROOT / "01_data" / "processed"
OUT_TABLE_DIR = PROJECT_ROOT / "05_tables" / "splits"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PEV_number"   # 先按主线只做 PEV
KEYS = ["Code", "Year", "Month"]

# 你主线的时间切分（可改）
TRAIN_END = (2022, 12)
CALIB_START = (2023, 1)
CALIB_END = (2024, 12)
TEST_START = (2025, 1)
TEST_END = (2030, 12)

# 训练必须可用的特征列（后续可以扩展）
REQUIRED_FEATURES = [
    "PEV_number_lag_1", "PEV_number_lag_3", "PEV_number_lag_12",
    "rolling_mean_3", "rolling_sum_12", "rolling_std_6",
    "total_GDP", "Urbanization_rate", "Baidu_index", "people", "Urban_population",
    "PEV_Subsidy funds", "CEV_Subsidy funds",
    "month", "quarter", "is_year_end",
]

def ym_to_int(y, m):
    return int(y) * 100 + int(m)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"找不到特征文件：{IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    df["Code"] = df["Code"].astype(str)

    # 排序保证一切按时间
    df = df.sort_values(["Code", "Year", "Month"]).reset_index(drop=True)

    # 构造 ym 方便切分
    df["ym"] = df.apply(lambda r: ym_to_int(r["Year"], r["Month"]), axis=1)

    train_end = ym_to_int(*TRAIN_END)
    calib_start = ym_to_int(*CALIB_START)
    calib_end = ym_to_int(*CALIB_END)
    test_start = ym_to_int(*TEST_START)
    test_end = ym_to_int(*TEST_END)

    # 标记 split
    df["split"] = "other"
    df.loc[df["ym"] <= train_end, "split"] = "train"
    df.loc[(df["ym"] >= calib_start) & (df["ym"] <= calib_end), "split"] = "calib"
    df.loc[(df["ym"] >= test_start) & (df["ym"] <= test_end), "split"] = "test"

    # === 训练/校准需要：y非空 + 特征非空 ===
    feat_cols = [c for c in REQUIRED_FEATURES if c in df.columns]

    # 如果某些列不存在，立刻报错（避免悄悄少特征）
    missing_feats = set(REQUIRED_FEATURES) - set(feat_cols)
    if missing_feats:
        raise ValueError(f"缺少特征列：{sorted(missing_feats)}")

    def usable_mask(dfx, need_y: bool):
        m = dfx[feat_cols].notna().all(axis=1)
        if need_y:
            m = m & dfx[TARGET].notna()
        return m

    train_df = df[df["split"] == "train"].copy()
    calib_df = df[df["split"] == "calib"].copy()
    test_df  = df[df["split"] == "test"].copy()

    # 过滤不可用样本（train/calib 必须）
    train_df = train_df[usable_mask(train_df, need_y=True)].copy()
    calib_df = calib_df[usable_mask(calib_df, need_y=True)].copy()

    # test 不要求 y（未来为空），但要求特征可用（至少 lag/rolling 要有）
    test_df = test_df[usable_mask(test_df, need_y=False)].copy()

    # 输出
    out_train = OUT_DIR / "dataset_train.parquet"
    out_calib = OUT_DIR / "dataset_calib.parquet"
    out_test  = OUT_DIR / "dataset_test.parquet"

    train_df.to_parquet(out_train, index=False)
    calib_df.to_parquet(out_calib, index=False)
    test_df.to_parquet(out_test, index=False)

    # 统计表（写论文很方便）
    summary = pd.DataFrame([
        {"split": "train", "rows": len(train_df), "years": f"<= {TRAIN_END[0]}-{TRAIN_END[1]:02d}"},
        {"split": "calib", "rows": len(calib_df), "years": f"{CALIB_START[0]}-{CALIB_START[1]:02d} ~ {CALIB_END[0]}-{CALIB_END[1]:02d}"},
        {"split": "test",  "rows": len(test_df),  "years": f"{TEST_START[0]}-{TEST_START[1]:02d} ~ {TEST_END[0]}-{TEST_END[1]:02d}"},
    ])
    summary_path = OUT_TABLE_DIR / "splits_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("[OK] split done")
    print(f"train: {out_train}  rows={len(train_df)}")
    print(f"calib: {out_calib}  rows={len(calib_df)}")
    print(f"test : {out_test}   rows={len(test_df)}")
    print(f"summary: {summary_path}")

if __name__ == "__main__":
    main()
