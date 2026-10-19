import os
import tqdm
import torch
import numpy as np
from sklearn import metrics


class Protein_Trainer:
    """
    Trainer for protein/RNA sequence-level binary classification.

    Expected input:
        data:  Tensor, shape [B, L, D]
        label: Tensor, shape [B, L] or [B, L, 1]

    Expected model output:
        logits: Tensor, shape [B, L, C]
    """


    def __init__(self, model, device, save_directory, lr=1e-4, weight_decay=1e-5):
        self.device = device
        self.model = model.to(device)
        self.save_directory = str(save_directory)
        self.logging_filename = "log.txt"
        self.with_cuda = device.type == "cuda"
        self.verbose_steps = 100
        self.save_model_flag = True
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optim_schedule = None

    def _prepare_batch(self, data, label):
        """
        Move batch data to the same device as model and normalize dtype.
        """
        data = data.to(self.device, non_blocking=True).float()
        label = label.to(self.device, non_blocking=True).long()

        if label.dim() == 3 and label.size(-1) == 1:
            label = label.squeeze(-1)

        return data, label

    def _unwrap_logits(self, output):
        """
        Some models may return tuple/list. We only use the first output as logits.
        """
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _get_bag_loss_fn(self):
        return self.criterion

    def _compute_loss(self, logits, label):
        loss_fn = self._get_bag_loss_fn()

        if logits.dim() == 2:
            return loss_fn(
                logits,
                label.contiguous().view(-1).long()
            )

        return loss_fn(
            logits.contiguous().view(-1, logits.size(-1)),
            label.contiguous().view(-1).long()
        )

    def _compute_batch_correct(self, logits, label):
        """
        Token-level accuracy.
        """
        pred = logits.argmax(dim=-1)
        correct = pred.eq(label).sum().item()
        element = label.numel()
        return correct, element

    def train_iteration(self, epoch, data_loader):
        """
        Training iteration loop with optional residual contrastive learning.
        """
        str_code = "train"
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (str_code, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        avg_loss = 0.0
        avg_cls_loss = 0.0
        avg_cl_loss = 0.0

        total_correct = 0
        total_element = 0

        self.model.train()

        for batch_idx, (data, label) in data_iter:
            data, label = self._prepare_batch(data, label)

            if data.size(1) != label.size(1):
                # 释放显存并跳过当前循环
                del data, label
                if getattr(self, "with_cuda", False):
                    torch.cuda.empty_cache()
                continue

            use_residual_cl = getattr(self.model, "use_residual_cl", False)

            if use_residual_cl:
                output = self.model(
                    data,
                    labels=None,
                    return_cl=True,
                    return_loss=False
                )

                logits = output["logits"]
                cl_loss = output.get("cl_loss", None)

                cls_loss = self._compute_loss(logits, label)

                if cl_loss is not None:
                    cl_weight = getattr(self.model, "CL_loss_weight", 0.05)
                    loss = cls_loss + cl_weight * cl_loss
                else:
                    loss = cls_loss

            else:
                output = self.model(data)
                logits = self._unwrap_logits(output)

                cls_loss = self._compute_loss(logits, label)
                cl_loss = None
                loss = cls_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if hasattr(self, "optim_schedule") and self.optim_schedule is not None:
                self.optim_schedule.step()

            avg_loss += loss.item()

            if cls_loss is not None:
                avg_cls_loss += cls_loss.item()

            if cl_loss is not None:
                avg_cl_loss += cl_loss.item()

            correct, element = self._compute_batch_correct(logits, label)
            total_correct += correct
            total_element += element

            if batch_idx % getattr(self, "verbose_steps", 100) == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": batch_idx,
                    "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                    "avg_cls_loss": "{:.4f}".format(avg_cls_loss / (batch_idx + 1)),
                    "avg_acc": "{:.4f}%".format(total_correct / max(total_element, 1) * 100),
                    "loss": "{:.4f}".format(loss.item()),
                    "acc": "{:.4f}%".format(correct / max(element, 1) * 100),
                }

                if cl_loss is not None:
                    post_fix["avg_cl_loss"] = "{:.4f}".format(avg_cl_loss / (batch_idx + 1))
                    post_fix["cl_loss"] = "{:.4f}".format(cl_loss.item())

                data_iter.write(str(post_fix))

            del data, label, logits, loss, output

            if "cls_loss" in locals():
                del cls_loss
            if "cl_loss" in locals():
                del cl_loss

            if getattr(self, "with_cuda", False):
                torch.cuda.empty_cache()

        avg_epoch_loss = avg_loss / max(len(data_iter), 1)
        avg_epoch_cls_loss = avg_cls_loss / max(len(data_iter), 1)
        avg_epoch_cl_loss = avg_cl_loss / max(len(data_iter), 1)
        avg_epoch_acc = total_correct * 100.0 / max(total_element, 1)

        print(
            "EP{}_{} | avg_loss={:.4f} | avg_cls_loss={:.4f} | avg_cl_loss={:.4f} | avg_acc={:.4f}%".format(
                epoch,
                str_code,
                avg_epoch_loss,
                avg_epoch_cls_loss,
                avg_epoch_cl_loss,
                avg_epoch_acc
            )
        )

        save_dir = getattr(self, "save_directory", "./logs")
        os.makedirs(save_dir, exist_ok=True)

        if getattr(self, "save_model_flag", False):
            log_txt_formatter = (
                "Train [Epoch] {epoch_str:03d} "
                "[Loss] {loss_str:.4f} "
                "[ClsLoss] {cls_loss_str:.4f} "
                "[CLLoss] {cl_loss_str:.4f} "
                "[Acc] {Acc_str:.4f}%\n"
            )

            log_path = os.path.join(
                save_dir,
                getattr(self, "logging_filename", "log.txt")
            )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    log_txt_formatter.format(
                        epoch_str=epoch,
                        loss_str=avg_epoch_loss,
                        cls_loss_str=avg_epoch_cls_loss,
                        cl_loss_str=avg_epoch_cl_loss,
                        Acc_str=avg_epoch_acc
                    )
                )


    def eval_iteration(self, epoch, data_loader, str_code="valid"):
        """
        Evaluation iteration loop.
        """
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (str_code, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        all_outputs = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, label) in data_iter:
                data, label = self._prepare_batch(data, label)

                if data.size(1) != label.size(1):
                    # 释放显存并跳过当前循环
                    del data, label
                    if getattr(self, "with_cuda", False):
                        torch.cuda.empty_cache()
                    continue

                output = self.model(data)
                logits = self._unwrap_logits(output)

                loss = self._compute_loss(logits, label)

                if logits.size(-1) == 1:
                    pos_probs = torch.sigmoid(logits).contiguous().view(-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    pos_probs = probs[..., 1].contiguous().view(-1)

                all_outputs.extend(pos_probs.detach().cpu().tolist())
                all_labels.extend(label.contiguous().view(-1).detach().cpu().tolist())

                avg_loss += loss.item()
                correct, element = self._compute_batch_correct(logits, label)
                total_correct += correct
                total_element += element

                if batch_idx % getattr(self, "verbose_steps", 100) == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": batch_idx,
                        "avg_loss": "{:.4f}".format(avg_loss / (batch_idx + 1)),
                        "avg_acc": "{:.4f}%".format(total_correct / max(total_element, 1) * 100),
                        "loss": "{:.4f}".format(loss.item()),
                        "acc": "{:.4f}%".format(correct / max(element, 1) * 100),
                    }
                    data_iter.write(str(post_fix))

                del data, label, logits, loss, output
                if "probs" in locals():
                    del probs
                del pos_probs

                if getattr(self, "with_cuda", False):
                    torch.cuda.empty_cache()

        avg_epoch_loss = avg_loss / max(len(data_iter), 1)
        avg_epoch_acc = total_correct * 100.0 / max(total_element, 1)

        results = self.calculate_scores(
            all_labels,
            all_outputs,
            autothresh = True
        )
        results["loss"] = avg_epoch_loss

        print("\nEP{}_{} Metrics:".format(epoch, str_code))
        print(
            " | Loss: {:.4f} | ACC: {:.4f}% | Pre: {:.4f} | Rec: {:.4f} | "
            "F1: {:.4f} | AUC: {:.4f} | AUPRC: {:.4f} | MCC: {:.4f} |".format(
                results["loss"],
                results["binary_acc"] * 100,
                results["precision"],
                results["recall"],
                results["f1"],
                results["AUC"],
                results["AUPRC"],
                results["mcc"]
            )
        )

        save_dir = getattr(self, "save_directory", "./logs")
        os.makedirs(save_dir, exist_ok=True)

        if getattr(self, "save_model_flag", False) and str_code == "valid":
            log_txt_formatter = (
                "Valid [Epoch] {epoch_str:03d} "
                "[Loss] {loss_str:.4f} "
                "[Acc] {acc_str:.4f}% "
                "[F1] {f1_str:.4f} "
                "[AUC] {auc_str:.4f} "
                "[AUPRC] {auprc_str:.4f}\n"
            )
            log_path = os.path.join(
                save_dir,
                getattr(self, "logging_filename", "log.txt")
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    log_txt_formatter.format(
                        epoch_str=epoch,
                        loss_str=results["loss"],
                        acc_str=results["binary_acc"] * 100.0,
                        f1_str=results["f1"],
                        auc_str=results["AUC"],
                        auprc_str=results["AUPRC"]
                    )
                )

        return results

    def calculate_scores(self, y_true, y_pred, threshold=0.5, autothresh=False):
        """
        Binary classification metrics.

        y_true: list of 0/1 labels
        y_pred: list of predicted probabilities for positive class
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(float)

        # Remove NaN/Inf if any.
        valid_mask = np.isfinite(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return {
                "binary_acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "AUC": 0.0,
                "AUPRC": 0.0,
                "mcc": 0.0,
                "threshold": threshold,
            }


        if autothresh and len(np.unique(y_true)) >= 2:
            p_curve, r_curve, t_curve = metrics.precision_recall_curve(y_true, y_pred)
            f1_scores = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-8)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(t_curve):
                threshold = float(t_curve[best_idx])
        # ==========================================

        binary_pred = (y_pred >= threshold).astype(int)

        binary_acc = metrics.accuracy_score(y_true, binary_pred)
        precision = metrics.precision_score(y_true, binary_pred, zero_division=0)
        recall = metrics.recall_score(y_true, binary_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, binary_pred, zero_division=0)
        mcc = metrics.matthews_corrcoef(y_true, binary_pred)

        # AUC/AUPRC need at least two classes.
        if len(np.unique(y_true)) < 2:
            AUC = 0.0
            AUPRC = 0.0
        else:
            try:
                AUC = metrics.roc_auc_score(y_true, y_pred)
            except ValueError:
                AUC = 0.0

            try:
                precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_pred)
                AUPRC = metrics.auc(recall_curve, precision_curve)
            except ValueError:
                AUPRC = 0.0

        results = {
            "binary_acc": binary_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "AUC": AUC,
            "AUPRC": AUPRC,
            "mcc": mcc,
            "threshold": threshold,
        }
        return results
