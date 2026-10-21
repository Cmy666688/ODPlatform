# data — 数据集目录

- `raw/` — 原始数据(只读,绝不就地修改)。结构: `raw/<数据集名>/images/` + `annotations/`
- `processed/` — 派生数据集(可再生,含冻结后的 train/val/test 划分)

数据内容不进 git(见根 `.gitignore`),走外部存储或 Git LFS。
