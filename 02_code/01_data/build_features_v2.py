# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "01_data" / "processed" / "clean_monthly_panel.csv"
OUT_PATH = PROJECT_ROOT / "01_data" / "processed" / "features_monthly.csv"

def main():
    df = pd.read_csv(IN_PATH, dtype={"Code": str})
    df["Code"] = df["Code"].astype(str).str.strip()

    # 排序非常关键：rolling/shift 必须按时间顺序
    df = df.sort_values(["Code", "Year", "Month"]).reset_index(drop=True)

    # ===== 1) 滞后特征（只用过去）=====
    g = df.groupby("Code", sort=False)
    df["PEV_number_lag_1"]  = g["PEV_number"].shift(1)
    df["PEV_number_lag_3"]  = g["PEV_number"].shift(3)
    df["PEV_number_lag_12"] = g["PEV_number"].shift(12)

    # ===== 2) 滚动特征（基于 lag_1 再 rolling，避免泄露）=====
    # 过去3个月均值：用 (t-1,t-2,t-3)
    df["rolling_mean_3"] = (
        g["PEV_number_lag_1"].rolling(3, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )

    # 过去12个月累计：用 (t-1 ... t-12)
    df["rolling_sum_12"] = (
        g["PEV_number_lag_1"].rolling(12, min_periods=12).sum()
        .reset_index(level=0, drop=True)
    )

    # 过去6个月波动：用 (t-1 ... t-6)
    df["rolling_std_6"] = (
        g["PEV_number_lag_1"].rolling(6, min_periods=6).std()
        .reset_index(level=0, drop=True)
    )

    # ===== 3) 日历特征 =====
    df["month"] = df["Month"]
    df["quarter"] = ((df["Month"] - 1) // 3 + 1).astype(int)
    df["is_year_end"] = (df["Month"] == 12).astype(int)

    # （可选）不删 City 也行；你现在删了也OK
    df = df.drop(columns=["City", "Province"], errors="ignore")

    # ===== (B) 强制 Code 为字符串，避免后续 join 对不齐 =====
    df["Code"] = df["Code"].astype(str)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # （推荐）parquet 能保留类型，比 csv 稳
    OUT_PARQUET = OUT_PATH.with_suffix(".parquet")
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"[OK] saved to: {OUT_PATH}")
    print(f"[OK] saved to: {OUT_PARQUET}")

    # 简单自检：每个城市的前几个月 rolling 应该是 NaN
    # print(df[df["Code"] == df["Code"].iloc[0]][["Year","Month","PEV_number","rolling_mean_3","rolling_sum_12"]].head(15))

if __name__ == "__main__":
    main()
