import pandas as pd
import numpy as np
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "01_data" / "processed" / "clean_monthly_panel.csv"
OUTPUT_PATH = PROJECT_ROOT / "01_data" / "processed" / "features_monthly.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：构造滞后、滚动和日期特征"""

    # 滞后特征（Lag features）
    df["PEV_number_lag_1"] = df.groupby("Code")["PEV_number"].shift(1)
    df["PEV_number_lag_3"] = df.groupby("Code")["PEV_number"].shift(3)
    df["PEV_number_lag_12"] = df.groupby("Code")["PEV_number"].shift(12)

    # 滚动特征（Rolling features）
    rolling_mean_3 = df.groupby("Code")["PEV_number"].rolling(3).mean().shift(-2).reset_index(level=0, drop=True)  # 滚动3个月的平均
    df["rolling_mean_3"] = rolling_mean_3

    rolling_sum_12 = df.groupby("Code")["PEV_number"].rolling(12).sum().shift(-11).reset_index(level=0, drop=True)  # 滚动12个月的和
    df["rolling_sum_12"] = rolling_sum_12

    rolling_std_6 = df.groupby("Code")["PEV_number"].rolling(6).std().shift(-5).reset_index(level=0, drop=True)  # 滚动6个月的标准差
    df["rolling_std_6"] = rolling_std_6

    # 日期相关特征（Calendar features）
    df["month"] = df["Month"]  # 直接保留 Month 列
    df["quarter"] = ((df["Month"] - 1) // 3) + 1  # 当前季度
    df["is_year_end"] = (df["Month"] == 12).astype(int)  # 是否年末

    # 删除不必要的列（City, Province 等）
    df = df.drop(columns=["City", "Province"], errors="ignore")

    return df


def main():
    # 读取已经清洗的数据
    df = pd.read_csv(INPUT_PATH)

    # 构建特征
    features_df = build_features(df)

    # 保存处理后的数据
    features_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] 特征工程完成，文件保存至：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
