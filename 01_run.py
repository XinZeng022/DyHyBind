"""Train one DyHyBind configuration and report independent-test metrics."""

import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from p2_1 import OptInit, ProteinGCN
from trainer_p2 import Protein_Trainer


class ProDataset(Dataset):
    def __init__(self, frame, feature_dir):
        self.frame = frame.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        features = np.load(self.feature_dir / f"{row['name']}.npy", allow_pickle=False)
        text = row["labels"].strip()
        labels = ast.literal_eval(text) if text.startswith("[") else list(text)
        labels = np.asarray([int(value) for value in labels], dtype=np.int64)
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError(f"Feature/label dimensions do not match for {row['name']}.")
        if not len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Expected nonempty binary residue labels for {row['name']}.")
        if not np.isfinite(features).all():
            raise ValueError(f"Nonfinite features for {row['name']}.")
        return torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels)


def read_dataset(path):
    frame = pd.read_csv(path, dtype={"name": str, "labels": str}, keep_default_na=False)
    if not {"name", "labels"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Each dataset must be nonempty and contain name and labels columns.")
    if frame["name"].duplicated().any():
        raise ValueError("Protein names must be unique within each dataset.")
    return frame


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True, help="Training CSV.")
    parser.add_argument("--test_csv_path", type=Path, nargs="+", required=True, help="Independent test CSV(s).")
    parser.add_argument("--feature_dir", type=Path, required=True, help="Training residue .npy matrices.")
    parser.add_argument("--test_feature_dir", type=Path, nargs="+", help="One feature directory per test CSV; defaults to --feature_dir.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_knn", type=int, required=True)
    parser.add_argument("--CL_margin", type=float, required=True)
    parser.add_argument("--CL_noise_max", type=float, required=True)
    parser.add_argument("--CL_loss_weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--stop_epoch", type=int, default=5)
    parser.add_argument("--save_best_eval", choices=["loss", "acc"], default="loss")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    args = parser.parse_args()
    if args.test_feature_dir and len(args.test_feature_dir) != len(args.test_csv_path):
        parser.error("Provide one --test_feature_dir for each --test_csv_path.")
    if min(args.num_knn, args.hidden_channels, args.num_heads, args.num_layers, args.max_epochs,
           args.patience) < 1 or args.k_folds < 2:
        parser.error("Model/training counts must be positive; k_folds must be at least two.")
    if not 0 <= args.CL_noise_max <= 1 or min(args.CL_margin, args.CL_loss_weight) < 0:
        parser.error("CL_noise_max must be in [0, 1]; contrastive margin and weight must be nonnegative.")
    names = [path.stem for path in args.test_csv_path]
    if len(set(names)) != len(names):
        parser.error("Test CSV filenames must have distinct stems.")
    return args


def metric_row(fold, phase, dataset, epoch, metrics):
    return {"Fold": fold, "Phase": phase, "Dataset": dataset, "Epoch": epoch,
            "Loss": metrics["loss"], "Acc": metrics["binary_acc"], "Precision": metrics["precision"],
            "Recall": metrics["recall"], "F1": metrics["f1"], "AUC": metrics["AUC"],
            "AUPRC": metrics["AUPRC"], "MCC": metrics["mcc"], "Threshold": metrics["threshold"]}


def main():
    args = parse_args()
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    train = read_dataset(args.csv_path)
    if len(train) < args.k_folds:
        raise ValueError("The training dataset contains fewer proteins than k_folds.")
    feature_dirs = args.test_feature_dir or [args.feature_dir] * len(args.test_csv_path)
    test_loaders = []
    for path, directory in zip(args.test_csv_path, feature_dirs):
        test = read_dataset(path)
        if set(train["name"]).intersection(test["name"]):
            raise ValueError("Training and independent test protein names overlap.")
        test_loaders.append((path.stem, DataLoader(ProDataset(test, directory), batch_size=1, shuffle=False)))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    input_dim = ProDataset(train, args.feature_dir)[0][0].shape[1]
    rows = []
    folds = KFold(n_splits=args.k_folds, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(folds.split(train), start=1):
        fold_dir = args.results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(ProDataset(train.iloc[train_index], args.feature_dir), batch_size=1, shuffle=True)
        valid_loader = DataLoader(ProDataset(train.iloc[valid_index], args.feature_dir), batch_size=1, shuffle=False)
        options = OptInit(dim_in=input_dim, num_classes=2, num_knn=args.num_knn, drop_path_rate=0.0,
                          hidden_channels=args.hidden_channels, num_heads=args.num_heads, num_layers=args.num_layers,
                          CL_margin=args.CL_margin, CL_noise_max=args.CL_noise_max,
                          CL_loss_weight=args.CL_loss_weight, use_residual_cl=True)
        model = ProteinGCN(options).to(device)
        trainer = Protein_Trainer(model, device, fold_dir, args.lr, args.weight_decay)
        best_value = float("inf") if args.save_best_eval == "loss" else -1.0
        best_metrics, best_epoch, stale_epochs = None, 0, 0
        checkpoint = fold_dir / "best_model.pth"
        for epoch in range(1, args.max_epochs + 1):
            trainer.train_iteration(epoch, train_loader)
            metrics = trainer.eval_iteration(epoch, valid_loader, str_code="valid")
            value = metrics["loss"] if args.save_best_eval == "loss" else metrics["binary_acc"]
            improved = value < best_value if args.save_best_eval == "loss" else value > best_value
            if improved:
                best_value, best_metrics, best_epoch, stale_epochs = value, metrics, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience and epoch >= args.stop_epoch:
                break
        if best_metrics is None:
            raise RuntimeError("No finite validation result was obtained.")
        rows.append(metric_row(fold, "Valid", args.csv_path.stem, best_epoch, best_metrics))
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        for name, test_loader in test_loaders:
            metrics = trainer.eval_iteration(best_epoch, test_loader, str_code="test")
            rows.append(metric_row(fold, "Test", name, best_epoch, metrics))
        pd.DataFrame(rows).to_csv(args.results_dir / "fold_metrics.csv", index=False)

    frame = pd.DataFrame(rows)
    columns = ["Loss", "Acc", "Precision", "Recall", "F1", "AUC", "AUPRC", "MCC"]
    table = frame.loc[frame["Phase"] == "Test"].groupby("Dataset", sort=False)[columns].mean().reset_index()
    table.to_csv(args.results_dir / "main_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
