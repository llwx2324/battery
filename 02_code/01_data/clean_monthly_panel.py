# -*- coding: utf-8 -*-
"""
数据清洗脚本：
- 清理 City、Province 列的尾部空格
- 强制 `Code` 为字符串类型，避免类型不一致导致后续 join 问题
- 转换 `Year`, `Month` 为合适的类型
- 删除无用列，确保数据的格式统一
"""

from pathlib import Path
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_PATH = PROJECT_ROOT / "01_data" / "raw" / "2016-2030_PEV and CEV_month+data.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "01_data" / "processed" / "clean_monthly_panel.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数据：处理列名、空格、数据类型等"""

    # 统一列名，去掉多余空格
    df.columns = [str(c).strip() for c in df.columns]

    # 清理字符串列（比如 Province/City），去掉尾部空格
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # 将 `Code` 强制为字符串类型
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str).str.strip()

    # 转换 `Year` 和 `Month` 为数字类型
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Month" in df.columns:
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    # 删除无用的列，如果有必要的列，注释掉这一行
    df = df.drop(columns=["Province", "CN_CITY"], errors="ignore")

    # 返回清理后的数据
    return df


def main():
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"文件不存在：{MONTHLY_PATH}")

    # 读取数据
    df = pd.read_excel(MONTHLY_PATH)

    # 数据清洗
    cleaned_df = clean_data(df)

    # 输出清理后的数据
    cleaned_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] 数据清洗完成，清理后的文件保存为：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
