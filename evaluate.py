import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import ResponseDataset, build_graphs, load_dataset
from metrics import dca_eca, empirical_behavior, epa, krc, prediction_metrics
from model import ConECD
from train import predict

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved ConECD checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Directory for re-evaluation metrics; defaults to checkpoint directory.")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve() if args.output else checkpoint_path.parent
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    bundle = load_dataset(ROOT / "data" / config["dataset"])
    model = ConECD(bundle.student_num, bundle.exercise_num, bundle.knowledge_num, latent_dim=config["latent_dim"], q_layers=config["q_layers"], response_layers=config["response_layers"], local_logit_cap=float(config.get("local_logit_cap", 1.0))).to(device)
    model.set_graphs(build_graphs(bundle.train, bundle.q_matrix, bundle.student_num, bundle.exercise_num, config["response_self_loops"]))
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(ResponseDataset(bundle.test, bundle.q_matrix), batch_size=config["batch_size"], shuffle=False)
    labels, predictions = predict(model, loader, device)
    mastery = model.mastery_matrix(512).astype(np.float32, copy=False)
    behavior, counts = empirical_behavior(bundle.response, bundle.q_matrix, bundle.student_num, bundle.knowledge_num)
    contrast = dca_eca(mastery, behavior, counts, tau=config["tau"], gamma=config["gamma"])
    alignment = epa(mastery, behavior, counts)
    result = {"status": "re_evaluated", "evaluated_at": datetime.now(timezone.utc).isoformat(), "dataset": config["dataset"], "seed": config.get("seed"), "run_tag": config.get("run_tag", "manual"), "checkpoint_path": str(checkpoint_path), **prediction_metrics(labels, predictions), "KRC": krc(mastery, bundle.test, bundle.q_matrix), "DCA": contrast["dca"], "ECA": contrast["eca"], "EPA": alignment["epa"], "dca_eca_pairs": contrast["pairs"], "epa_pairs": alignment["pairs"]}
    (output / "evaluation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(output / "evaluation_metrics.csv", index=False)
    np.save(output / "evaluation_mastery.npy", mastery)
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation metrics to: {(output / 'evaluation_metrics.csv').resolve()}")


if __name__ == "__main__":
    main()

