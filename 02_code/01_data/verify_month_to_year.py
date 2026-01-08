# -*- coding: utf-8 -*-
"""
任务1：月→年一致性验证（PEV）
- 月度表按 (Code, Year) 求和 monthly_sum
- 对齐 annual_PEV_data 的 (Code, Year) 年值 annual_value
- 输出误差统计 + TOP10 城市
- 自动定位项目根（不受 PyCharm 右键运行 cwd 影响）
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# 关键：用脚本位置定位项目根
# .../<root>/02_code/01_data/verify_month_to_year.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONTHLY_PATH = PROJECT_ROOT / "01_data" / "raw" / "2016-2030_PEV and CEV_month+data.xlsx"
ANNUAL_PATH  = PROJECT_ROOT / "01_data" / "raw" / "annual_PEV_data.xlsx"

OUT_DIR = PROJECT_ROOT / "05_tables" / "month2year_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_cols_and_objects(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].str.lower().isin(["nan", "none", "null"]), c] = np.nan
    return df


def main():
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] OUT_DIR      = {OUT_DIR}")

    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"月度文件不存在：{MONTHLY_PATH}")
    if not ANNUAL_PATH.exists():
        raise FileNotFoundError(f"年度文件不存在：{ANNUAL_PATH}")

    monthly = pd.read_excel(MONTHLY_PATH)
    annual  = pd.read_excel(ANNUAL_PATH)

    monthly = strip_cols_and_objects(monthly)
    annual  = strip_cols_and_objects(annual)

    # 关键键处理：Code 强制为 str，Year/Month 转数值
    if "Code" in monthly.columns:
        monthly["Code"] = monthly["Code"].astype(str).str.strip()
    if "Code" in annual.columns:
        annual["Code"] = annual["Code"].astype(str).str.strip()

    for df in (monthly, annual):
        if "Year" not in df.columns:
            raise ValueError("缺少 Year 列")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    if "Month" in monthly.columns:
        monthly["Month"] = pd.to_numeric(monthly["Month"], errors="coerce").astype("Int64")

    # 列存在性检查
    for col in ["Code", "Year", "PEV_number"]:
        if col not in monthly.columns:
            raise ValueError(f"月度表缺少列：{col}")
    for col in ["Code", "Year", "PEV_number"]:
        if col not in annual.columns:
            raise ValueError(f"年度表缺少列：{col}")

    # 只对 monthly 中 PEV_number 非空的年份做一致性验证
    # （因为你现在月度 PEV_number 有 40% 缺失，基本就是未来年份没标签）
    monthly_valid = monthly.dropna(subset=["Code", "Year"])
    annual_valid  = annual.dropna(subset=["Code", "Year"])

    years_with_label = sorted(monthly_valid.loc[monthly_valid["PEV_number"].notna(), "Year"].dropna().unique().tolist())
    print(f"[INFO] 月度表有 PEV_number 标签的年份（用于对齐验证）：{years_with_label[:5]} ... 共 {len(years_with_label)} 年")

    # 月度按年汇总
    m_agg = (monthly_valid[monthly_valid["Year"].isin(years_with_label)]
             .groupby(["Code", "Year"], dropna=False)["PEV_number"]
             .sum(min_count=1)
             .reset_index()
             .rename(columns={"PEV_number": "monthly_sum"}))

    # 年度按键汇总（防御性：万一有重复）
    a_agg = (annual_valid[annual_valid["Year"].isin(years_with_label)]
             .groupby(["Code", "Year"], dropna=False)["PEV_number"]
             .sum(min_count=1)
             .reset_index()
             .rename(columns={"PEV_number": "annual_value"}))

    merged = pd.merge(m_agg, a_agg, on=["Code", "Year"], how="inner")

    # 补充城市名（用于TOP10展示）
    name_col = "City" if "City" in monthly.columns else ("CN_CITY" if "CN_CITY" in monthly.columns else None)
    if name_col:
        code2name = (monthly[["Code", name_col]]
                     .dropna()
                     .groupby("Code")[name_col]
                     .agg(lambda x: x.value_counts().index[0])
                     .reset_index())
        merged = merged.merge(code2name, on="Code", how="left")

    # 误差
    merged["diff"] = merged["monthly_sum"] - merged["annual_value"]
    merged["abs_diff"] = merged["diff"].abs()
    merged["rel_diff"] = np.where(
        merged["annual_value"].abs() > 1e-12,
        merged["diff"] / merged["annual_value"],
        np.nan
    )
    merged["abs_rel_diff"] = merged["rel_diff"].abs()

    # 统计
    mae = float(np.nanmean(merged["abs_diff"]))
    rmse = float(np.sqrt(np.nanmean(np.square(merged["diff"]))))
    mape = float(np.nanmean(merged["abs_rel_diff"]) * 100.0)

    summary_all = pd.DataFrame([{
        "N_pairs": int(len(merged)),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "MAX_abs_diff": float(np.nanmax(merged["abs_diff"])) if len(merged) else np.nan,
    }])

    summary_by_year = (merged.groupby("Year", dropna=False)
                       .apply(lambda g: pd.Series({
                           "N_pairs": int(len(g)),
                           "MAE": float(np.nanmean(g["abs_diff"])),
                           "RMSE": float(np.sqrt(np.nanmean(np.square(g["diff"])))),
                           "MAPE_%": float(np.nanmean(g["abs_rel_diff"]) * 100.0),
                           "MAX_abs_diff": float(np.nanmax(g["abs_diff"])) if len(g) else np.nan,
                       }))
                       .reset_index())

    top10 = merged.sort_values("abs_diff", ascending=False).head(10)

    # 输出
    merged.to_csv(OUT_DIR / "month2year_pairwise_errors.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT_DIR / "month2year_summary_all.csv", index=False, encoding="utf-8-sig")
    summary_by_year.to_csv(OUT_DIR / "month2year_summary_by_year.csv", index=False, encoding="utf-8-sig")
    top10.to_csv(OUT_DIR / "month2year_top10_absdiff.csv", index=False, encoding="utf-8-sig")

    print("\n==== 输出完成 ====")
    print(f"[OUT] {OUT_DIR / 'month2year_summary_all.csv'}")
    print(f"[OUT] {OUT_DIR / 'month2year_top10_absdiff.csv'}")

    print("\n==== 总体统计预览 ====")
    print(summary_all.to_string(index=False))

    show_cols = ["Code", "Year"]
    if name_col and name_col in top10.columns:
        show_cols.append(name_col)
    show_cols += ["monthly_sum", "annual_value", "diff", "abs_diff", "rel_diff"]
    print("\n==== TOP10 预览（按 abs_diff）====")
    print(top10[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
