# -*- coding: utf-8 -*-
"""
Data Profiling Script（稳定定位项目根，不受运行目录影响）

无论你从哪里运行（VSCode/终端/右键），都会输出到：
- <项目根>/05_tables/data_profile/data_profile_summary.csv
- <项目根>/00_admin/data_map.md
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt


# =========================
# 关键修复：用 __file__ 定位项目根
# 当前脚本路径：<root>/02_code/01_data/profile_data_files.py
# parents[0]=01_data, parents[1]=02_code, parents[2]=<root>
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT_TABLE_DIR = PROJECT_ROOT / "05_tables" / "data_profile"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_MAP_MD = PROJECT_ROOT / "00_admin" / "data_map.md"
OUT_MAP_MD.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 你项目里“关键数据路径”（相对项目根）
# 说明：如果你的原始数据不在 01_data/raw，而在 01_data/external，
# 你只要把 path 改一下即可
# =========================
DATA_SPECS = [
    {
        "name": "monthly_panel",
        "desc": "城市级月度面板（时空预测核心数据）",
        "path": "01_data/raw/2016-2030_PEV and CEV_month+data.xlsx",
        "expected_keys": ["Code", "Year", "Month"],
        "targets": ["PEV_number", "CEV_number"],
    },
    {
        "name": "annual_PEV",
        "desc": "城市级年度 PEV（用于月→年一致性验证）",
        "path": "01_data/raw/annual_PEV_data.xlsx",
        "expected_keys": ["Code", "Year"],
        "targets": ["PEV_number"],
    },
    {
        "name": "annual_CEV",
        "desc": "城市级年度 CEV（用于月→年一致性验证）",
        "path": "01_data/raw/annual_CEV_data.xlsx",
        "expected_keys": ["Code", "Year"],
        "targets": ["CEV_number"],
    },
]


def read_table(fp: Path):
    if not fp.exists():
        return None, "missing", ""

    try:
        suf = fp.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(fp), "csv", ""
        if suf in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(fp)
            sheet = xls.sheet_names[0]
            return pd.read_excel(fp, sheet_name=sheet), "excel", sheet
        return None, "unsupported", ""
    except Exception as e:
        return None, "error", repr(e)


def strip_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].str.lower().isin(["nan", "none", "null"]), c] = np.nan
    return df


def infer_time(df: pd.DataFrame):
    info = {"year_min": "", "year_max": "", "month_min": "", "month_max": ""}
    if "Year" in df.columns:
        y = pd.to_numeric(df["Year"], errors="coerce")
        if y.notna().any():
            info["year_min"] = int(y.min())
            info["year_max"] = int(y.max())
    if "Month" in df.columns:
        m = pd.to_numeric(df["Month"], errors="coerce")
        if m.notna().any():
            info["month_min"] = int(m.min())
            info["month_max"] = int(m.max())
    return info


def key_check(df: pd.DataFrame, keys):
    keys_exist = [k for k in keys if k in df.columns]
    if not keys_exist:
        return "", "", ""
    missing = int(df[keys_exist].isna().any(axis=1).sum())
    dup = int(df[keys_exist].dropna().duplicated().sum())
    return ",".join(keys_exist), missing, dup


def main():
    # 打印一下，方便你确认“项目根”和“输出目录”到底指向哪里
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] OUT_TABLE_DIR = {OUT_TABLE_DIR}")
    print(f"[INFO] OUT_MAP_MD    = {OUT_MAP_MD}")

    rows = []
    md = []
    md.append("# 数据地图（Data Map）")
    md.append("")
    md.append(f"- 自动生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- 项目根目录：`{PROJECT_ROOT}`")
    md.append("")

    for spec in DATA_SPECS:
        fp = PROJECT_ROOT / spec["path"]
        df, ftype, sheet = read_table(fp)

        row = {
            "name": spec["name"],
            "desc": spec["desc"],
            "path": spec["path"],
            "abs_path": str(fp),
            "exists": fp.exists(),
            "file_type": ftype,
            "sheet": sheet,
        }

        md.append(f"## {spec['name']}")
        md.append(f"- 描述：{spec['desc']}")
        md.append(f"- 路径：`{spec['path']}`")
        md.append(f"- 绝对路径：`{fp}`")

        if df is None:
            row.update({"n_rows": "", "n_cols": ""})
            rows.append(row)
            md.append(f"- 状态：❌ 无法读取（{ftype}）")
            md.append("")
            continue

        df = strip_df(df)
        row["n_rows"] = int(len(df))
        row["n_cols"] = int(df.shape[1])

        row.update(infer_time(df))

        k_used, k_missing, k_dup = key_check(df, spec["expected_keys"])
        row["keys_used"] = k_used
        row["key_missing_rows"] = k_missing
        row["key_duplicate_rows"] = k_dup

        for t in spec["targets"]:
            row[f"missing_{t}"] = float(df[t].isna().mean()) if t in df.columns else ""

        rows.append(row)

        md.append(f"- 行列数：{row['n_rows']} × {row['n_cols']}")
        md.append(f"- 时间范围：Year {row.get('year_min','')}–{row.get('year_max','')}, Month {row.get('month_min','')}–{row.get('month_max','')}")
        if k_used:
            md.append(f"- 主键：{k_used}")
            md.append(f"  - 缺失行：{k_missing}")
            md.append(f"  - 重复行：{k_dup}")
        else:
            md.append("- 主键：TODO（未检测到 expected_keys）")

        if spec["targets"]:
            md.append(f"- 目标变量：{', '.join(spec['targets'])}")
        else:
            md.append("- 目标变量：（无）")

        md.append("- 口径说明：TODO")
        md.append("")

    df_out = pd.DataFrame(rows)
    csv_path = OUT_TABLE_DIR / "data_profile_summary.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    OUT_MAP_MD.write_text("\n".join(md), encoding="utf-8")

    print("==== Data profiling completed ====")
    print(f"[TABLE] {csv_path}")
    print(f"[MAP ] {OUT_MAP_MD}")


if __name__ == "__main__":
    main()
