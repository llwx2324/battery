# Thesis Workspace

本目录用于：代码、数据快照、实验结果、图表、论文草稿。

## 目录说明
- 01_data/
  - raw/          原始数据（不改动，必要时只做复制）
  - processed/    清洗/特征工程后的数据
  - external/     外部补充数据（可选）
- 02_code/
  - src/          可复用代码（模型、特征、评估、工具）
  - scripts/      可运行脚本（训练、评估、导出）
  - notebooks/    交互式探索（可选）
- 03_runs/
  - baseline_xgb/ xgboost基线实验
  - stgcn/        时空模型实验
  - interval_cp/  共形预测区间实验
  - mc/           MC传播实验
- 04_figures/     出图（论文用）
- 05_tables/      表格（论文用）
- 06_paper/       论文写作（outline/draft/assets）
- env/            环境依赖与脚本
