"""Training entry-point for the BST-GAT model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data import AirQualityDataset, adjacency_from_graph, collate_graph_batches
from .model import BayesianSpatioTemporalGAT, gaussian_nll


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: BayesianSpatioTemporalGAT,
    loader: DataLoader,
    adjacency: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_scale: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        features = batch["features"]  # (batch, history, stations, features)
        targets = batch["target"]  # (batch, horizon, stations)
        target_mask = batch["target_mask"].float()
        node_mask = batch["history_mask"][:, -1, :]

        pred = model(features, adjacency, mask=node_mask)
        nll = gaussian_nll(pred, targets, mask=target_mask)
        loss = nll + kl_scale * pred.kl_divergence / len(loader.dataset)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: BayesianSpatioTemporalGAT,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
    num_samples: int = 20,
) -> Dict[str, float]:
    model.eval()
    mse = 0.0
    mae = 0.0
    coverage = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            features = batch["features"]
            targets = batch["target"]
            mask = batch["target_mask"].float()
            node_mask = batch["history_mask"][:, -1, :]

            pred = model(features, adjacency, mask=node_mask)
            diff = (pred.mean - targets) * mask
            mse += torch.sum(diff.pow(2)).item()
            mae += torch.sum(diff.abs()).item()
            total += mask.sum().item()

            if num_samples > 1:
                samples = pred.predictive_samples(num_samples)
                lower = torch.quantile(samples, 0.025, dim=0)
                upper = torch.quantile(samples, 0.975, dim=0)
                in_interval = ((targets >= lower) & (targets <= upper)).float() * mask
                coverage += in_interval.sum().item()

    results = {
        "rmse": (mse / max(total, 1.0)) ** 0.5,
        "mae": mae / max(total, 1.0),
    }
    if num_samples > 1:
        results["p95_coverage"] = coverage / max(total, 1.0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Bayesian Spatio-Temporal GAT")
    parser.add_argument("--data", type=Path, required=True, help="Path to the air quality CSV dataset")
    parser.add_argument("--target", type=str, default="Ozone", help="Target column to forecast")
    parser.add_argument("--features", nargs="*", help="Feature columns to use")
    parser.add_argument("--history", type=int, default=24, help="History window length")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--k-neighbors", type=int, default=3, help="K for nearest-neighbor graph construction")
    parser.add_argument("--kl-scale", type=float, default=1.0, help="Scaling factor for KL divergence term")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of posterior samples for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, help="Optional path to save the trained model")
    args = parser.parse_args()

    dataset = AirQualityDataset(
        csv_path=args.data,
        history=args.history,
        horizon=args.horizon,
        target_column=args.target,
        feature_columns=args.features,
    )
    graph = dataset.build_graph(k=args.k_neighbors)
    adjacency = adjacency_from_graph(graph).to(args.device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_batches,
    )

    model = BayesianSpatioTemporalGAT(
        num_nodes=len(dataset.stations),
        feature_dim=len(dataset.feature_columns),
        history=args.history,
        horizon=args.horizon,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, adjacency, optimizer, args.device, args.kl_scale)
        metrics = evaluate(model, loader, adjacency, args.device, args.num_samples)
        metric_str = ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        print(f"Epoch {epoch}: loss={train_loss:.4f}, {metric_str}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "adjacency": adjacency.cpu(),
            "stations": dataset.stations,
            "feature_columns": dataset.feature_columns,
            "history": args.history,
            "horizon": args.horizon,
        }, args.save)
        print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
