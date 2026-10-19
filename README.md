# DyHyBind

输入为已准备好的融合特征。

```text
01_run.py          固定参数训练、测试与主表汇总
model.py           动态超图与残差对比学习模型
trainer.py         训练、评估及指标计算
layers.py          模型所需基础层
requirements.txt   运行依赖
```

安装依赖后，从本目录运行。CSV 需要 `name`、`labels` 两列；特征目录包含每条蛋白的 `<name>.npy`，形状为 `[残基数, 特征维数]`。标签可为 `00110` 字符串或 `[0, 0, 1, 1, 0]`。

```bash
python -m pip install -r requirements.txt
python 01_run.py --csv_path data/train.csv --test_csv_path data/test1.csv data/test2.csv --feature_dir features --results_dir results --num_knn 6 --CL_margin 0.8 --CL_noise_max 0.20
```

命令只训练一组参数，默认五折。不同测试集有单独特征目录时，按顺序传入 `--test_feature_dir features/test1 features/test2`。

输出：各折 `best_model.pth`、`fold_metrics.csv`，以及各测试集指标按折取均值的 `main_table.csv`。
