"""Train and evaluate DCST on the Indian Pines 10-shot split."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
import torch
from torch import nn

from dcst import DCST, build_deploy_model
from dcst.data import build_loaders, prepare_indian_pines


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "ip.json"
DEFAULT_OUTPUT = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCST Indian Pines training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seeds(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"data", "model", "training"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    return config


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    targets, predictions = [], []
    for (pca_patch, raw_patch), labels in loader:
        logits = model(
            pca_patch.to(device),
            raw_patch.to(device),
        )
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    labels = list(range(num_classes))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    denominator = confusion.sum(axis=1)
    class_accuracy = np.divide(
        np.diag(confusion),
        denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=denominator != 0,
    )
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(class_accuracy.mean()),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "class_accuracy": class_accuracy.tolist(),
        "confusion": confusion.tolist(),
    }


def cpu_state_dict(model: nn.Module) -> dict:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train_seed(
    seed: int,
    config: dict,
    prepared,
    device: torch.device,
    output_dir: Path,
) -> dict:
    training = config["training"]
    data = config["data"]
    set_seed(seed)
    train_loader, test_loader = build_loaders(
        prepared,
        train_batch_size=int(data["train_batch_size"]),
        eval_batch_size=int(data["eval_batch_size"]),
        seed=seed,
    )
    set_seed(seed)
    model = DCST(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = int(training["epochs"])
    evaluate_every = int(training["evaluate_every"])
    gradient_clip = float(training["gradient_clip"])

    best = None
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for (pca_patch, raw_patch), labels in train_loader:
            labels = labels.to(device)
            logits = model(
                pca_patch.to(device),
                raw_patch.to(device),
            )
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            count = int(labels.shape[0])
            loss_sum += float(loss.item()) * count
            sample_count += count

        mean_loss = loss_sum / max(sample_count, 1)
        print(
            f"[seed {seed}] epoch {epoch:03d}/{epochs:03d} "
            f"loss={mean_loss:.6f}",
            flush=True,
        )
        if epoch % evaluate_every == 0 or epoch == epochs:
            metrics = evaluate(
                model,
                test_loader,
                device,
                int(data["num_classes"]),
            )
            record = {"epoch": epoch, "loss": mean_loss, **metrics}
            history.append(record)
            print(
                f"[seed {seed}] epoch {epoch:03d} "
                f"OA={100 * metrics['OA']:.4f}% "
                f"AA={100 * metrics['AA']:.4f}% "
                f"Kappa={100 * metrics['Kappa']:.4f}%",
                flush=True,
            )
            if best is None or metrics["OA"] > best["OA"]:
                best = {"best_epoch": epoch, **metrics}
                best_state = cpu_state_dict(model)

    if best is None or best_state is None:
        raise RuntimeError("No checkpoint was evaluated")

    best_model = DCST(config).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    deploy_model = build_deploy_model(best_model).to(device)
    deploy_metrics = evaluate(
        deploy_model,
        test_loader,
        device,
        int(data["num_classes"]),
    )
    for metric in ("OA", "AA", "Kappa"):
        if not np.isclose(
            deploy_metrics[metric],
            best[metric],
            atol=1e-10,
            rtol=0,
        ):
            raise RuntimeError(f"QKR folding changed {metric}")

    checkpoint = {
        "model": "DCST",
        "deploy": True,
        "config": config,
        "seed": seed,
        "best_epoch": int(best["best_epoch"]),
        "state_dict": cpu_state_dict(deploy_model),
    }
    torch.save(checkpoint, output_dir / f"seed_{seed}_deploy.pth")
    result = {
        "seed": seed,
        **best,
        "history": history,
        "train_time_seconds": float(time.time() - started),
        "training_parameters": int(
            sum(parameter.numel() for parameter in best_model.parameters())
        ),
        "deployment_parameters": int(
            sum(parameter.numel() for parameter in deploy_model.parameters())
        ),
    }
    (output_dir / f"seed_{seed}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def summarize(records: list[dict]) -> dict:
    summary = {}
    for metric in ("OA", "AA", "Kappa"):
        values = np.asarray(
            [100.0 * record[metric] for record in records],
            dtype=np.float64,
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": values.tolist(),
        }
    classes = 100.0 * np.asarray(
        [record["class_accuracy"] for record in records],
        dtype=np.float64,
    )
    summary["class_accuracy"] = {
        "mean": classes.mean(axis=0).tolist(),
        "std": classes.std(axis=0, ddof=0).tolist(),
    }
    return summary


def write_table(output_dir: Path, summary: dict) -> None:
    rows = []
    for class_id, (mean, std) in enumerate(
        zip(
            summary["class_accuracy"]["mean"],
            summary["class_accuracy"]["std"],
        ),
        start=1,
    ):
        rows.append((str(class_id), mean, std))
    for metric in ("OA", "AA", "Kappa"):
        rows.append(
            (
                metric,
                summary[metric]["mean"],
                summary[metric]["std"],
            )
        )
    with (output_dir / "DCST_IP.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Class_or_metric", "Mean_percent", "Std_percent"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seeds = parse_seeds(args.seeds, config["training"]["seeds"])
    if args.smoke_test:
        seeds = [seeds[0]]
        config = copy.deepcopy(config)
        config["training"]["epochs"] = 1
        config["training"]["evaluate_every"] = 1
        config["data"]["train_batch_size"] = min(
            8,
            int(config["data"]["train_batch_size"]),
        )
        config["data"]["eval_batch_size"] = min(
            8,
            int(config["data"]["eval_batch_size"]),
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset_path = (ROOT / config["data"]["file"]).resolve()
    prepared = prepare_indian_pines(
        dataset_path,
        patch_size=int(config["data"]["patch_size"]),
        pca_components=int(config["data"]["pca_components"]),
        pca_whiten=bool(config["data"]["pca_whiten"]),
    )

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else DEFAULT_OUTPUT / time.strftime("run_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"Config:  {config_path}", flush=True)
    print(f"Data:    {dataset_path}", flush=True)
    print(f"Output:  {output_dir}", flush=True)
    print(f"Device:  {device}", flush=True)
    print(f"Seeds:   {seeds}", flush=True)

    records = []
    for seed in seeds:
        result_path = output_dir / f"seed_{seed}.json"
        if result_path.is_file() and not args.overwrite:
            records.append(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
            print(f"[seed {seed}] cached", flush=True)
            continue
        records.append(
            train_seed(seed, config, prepared, device, output_dir)
        )
        summary = summarize(records)
        payload = {
            "model": "DCST",
            "dataset": "Indian Pines",
            "data_audit": prepared.audit,
            "seeds": [record["seed"] for record in records],
            "per_seed": records,
            "summary_percent": summary,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        write_table(output_dir, summary)

    summary = summarize(records)
    print("\nCompleted DCST training", flush=True)
    for metric in ("OA", "AA", "Kappa"):
        print(
            f"{metric}: {summary[metric]['mean']:.4f} +/- "
            f"{summary[metric]['std']:.4f}",
            flush=True,
        )
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
