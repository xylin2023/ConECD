import argparse
import csv
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ResponseDataset, build_graphs, load_dataset
from metrics import dca_eca, empirical_behavior, epa, krc, prediction_metrics
from model import ConECD

ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for student, item, q_mask, label in loader:
        output = model(student.to(device), item.to(device), q_mask.to(device))
        predictions.append(output.cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def configure_logger(output):
    logger = logging.getLogger(f"conecd.{output}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(output / "train.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def fmt(metrics):
    return " ".join(f"{key}={metrics[key]:.6f}" for key in ("RMSE", "AUC", "ACC", "F1"))


def update_run_summary(run_root):
    rows = []
    for metrics_path in sorted(run_root.glob("seed_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(payload)
    if not rows:
        return
    pd.DataFrame(rows).sort_values("seed").to_csv(run_root / "per_seed_metrics.csv", index=False)
    numeric = ["RMSE", "AUC", "ACC", "F1", "KRC", "DCA", "ECA", "EPA", "seconds"]
    summary = {"dataset": rows[0]["dataset"], "run_tag": rows[0]["run_tag"], "completed_seeds": len(rows)}
    for column in numeric:
        values = pd.to_numeric(pd.DataFrame(rows).get(column), errors="coerce").dropna()
        if len(values):
            summary[f"{column}_mean"] = float(values.mean())
            summary[f"{column}_std"] = float(values.std(ddof=0))
    pd.DataFrame([summary]).to_csv(run_root / "summary_mean_std.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train ConECD")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--dataset", default="a0910")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", default="manual")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_logit_cap", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if "datasets" in raw_config:
        if args.dataset not in raw_config["datasets"]:
            raise ValueError(f"Unknown dataset {args.dataset!r}; choose one of: {', '.join(sorted(raw_config['datasets']))}")
        config = {**raw_config["defaults"], **raw_config["datasets"][args.dataset]}
    else:
        config = raw_config
    if args.seed is not None:
        config["seed"] = args.seed
    local_logit_cap = float(args.local_logit_cap if args.local_logit_cap is not None else config.get("local_logit_cap", 1.0))
    run_config = {**config, "config_dataset": args.dataset, "run_tag": args.run_tag, "local_logit_cap": local_logit_cap}
    output = Path(args.output) if args.output else ROOT / "results" / config["dataset"] / args.run_tag / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    run_root = output.parent
    logger = configure_logger(output)
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    set_seed(config["seed"])
    device = torch.device(args.device)
    logger.info("START dataset=%s seed=%s device=%s run_tag=%s", config["dataset"], config["seed"], device, args.run_tag)

    bundle = load_dataset(ROOT / "data" / config["dataset"])
    loaders = {
        "train": DataLoader(ResponseDataset(bundle.train, bundle.q_matrix), batch_size=config["batch_size"], shuffle=True),
        "valid": DataLoader(ResponseDataset(bundle.valid, bundle.q_matrix), batch_size=config["batch_size"], shuffle=False),
        "test": DataLoader(ResponseDataset(bundle.test, bundle.q_matrix), batch_size=config["batch_size"], shuffle=False),
    }
    graphs = build_graphs(bundle.train, bundle.q_matrix, bundle.student_num, bundle.exercise_num, config["response_self_loops"])
    model = ConECD(bundle.student_num, bundle.exercise_num, bundle.knowledge_num, latent_dim=config["latent_dim"], q_layers=config["q_layers"], response_layers=config["response_layers"], local_logit_cap=local_logit_cap).to(device)
    model.set_graphs(graphs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    checkpoint_path = output / "best_model.pt"
    history_path = output / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "avg_bce", "valid_RMSE", "valid_AUC", "valid_ACC", "valid_F1", "is_best"])
        writer.writeheader()
        best_rmse, bad_epochs = float("inf"), 0
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            losses = []
            for student, item, q_mask, label in loaders["train"]:
                prediction = model(student.to(device), item.to(device), q_mask.to(device))
                loss = F.binary_cross_entropy(prediction, label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            valid_y, valid_p = predict(model, loaders["valid"], device)
            valid = prediction_metrics(valid_y, valid_p)
            improved = valid["RMSE"] < best_rmse - config["min_delta"]
            writer.writerow({"epoch": epoch, "avg_bce": float(np.mean(losses)), **{f"valid_{k}": v for k, v in valid.items()}, "is_best": improved})
            history_file.flush()
            line = f"epoch={epoch:03d} bce={np.mean(losses):.6f} {fmt(valid)} best={improved}"
            logger.info(line)
            print(line)
            if improved:
                best_rmse, bad_epochs = valid["RMSE"], 0
                torch.save({"epoch": epoch, "valid": valid, "avg_loss": float(np.mean(losses)), "model": model.state_dict(), "config": run_config}, checkpoint_path)
            else:
                bad_epochs += 1
            if bad_epochs >= config["patience"]:
                logger.info("EARLY_STOP epoch=%s patience=%s", epoch, config["patience"])
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_y, test_p = predict(model, loaders["test"], device)
    mastery = model.mastery_matrix(batch_size=512).astype(np.float32, copy=False)
    behavior, counts = empirical_behavior(bundle.response, bundle.q_matrix, bundle.student_num, bundle.knowledge_num)
    profile = dca_eca(mastery, behavior, counts, tau=config["tau"], gamma=config["gamma"])
    alignment = epa(mastery, behavior, counts)
    finished_at = datetime.now(timezone.utc).isoformat()
    result = {"status": "completed", "dataset": config["dataset"], "config_dataset": args.dataset, "seed": config["seed"], "run_tag": args.run_tag, "device": str(device), "started_at": started_at, "finished_at": finished_at, **prediction_metrics(test_y, test_p), "KRC": krc(mastery, bundle.test, bundle.q_matrix), "DCA": profile["dca"], "ECA": profile["eca"], "EPA": alignment["epa"], "dca_eca_pairs": profile["pairs"], "epa_pairs": alignment["pairs"], "best_epoch": checkpoint["epoch"], "best_valid_rmse": checkpoint["valid"]["RMSE"], "seconds": time.time() - started, "best_checkpoint_path": str(checkpoint_path.resolve()), "metrics_path": str((output / "metrics.json").resolve()), "mastery_path": str((output / "mastery.npy").resolve())}
    np.save(output / "mastery.npy", mastery)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(output / "metrics.csv", index=False)
    summary = "\n".join(["ConECD final result", f"dataset: {result['dataset']}", f"seed: {result['seed']}", f"run_tag: {result['run_tag']}", f"device: {result['device']}", f"seconds: {result['seconds']:.2f}", f"best_epoch: {result['best_epoch']}", f"best_valid_rmse: {result['best_valid_rmse']:.6f}", fmt(result), f"KRC={result['KRC']:.6f}", f"checkpoint: {result['best_checkpoint_path']}", f"metrics: {result['metrics_path']}", f"mastery: {result['mastery_path']}", ""])
    (output / "final_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("FINAL %s", summary.replace("\n", " | "))
    print("\n" + summary)
    update_run_summary(run_root)


if __name__ == "__main__":
    main()

