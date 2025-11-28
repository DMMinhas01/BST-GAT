"""Data utilities for the BST-GAT project."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class StationMetadata:
    """Metadata describing a monitoring station."""

    station: str
    latitude: float
    longitude: float
    elevation: Optional[float]


class AirQualityDataset(Dataset):
    """Sliding-window dataset for multivariate spatio-temporal forecasting.

    The dataset expects a CSV file containing at least the following columns:

    ``Station``
        Station identifier. Each row must belong to one station.
    ``DateTime``
        Timestamp in ISO 8601 format.
    ``Latitude`` and ``Longitude``
        Coordinates used to compute the spatial graph.

    Any additional numeric columns are treated as features. The column provided
    via ``target_column`` is the value to forecast.
    """

    def __init__(
        self,
        csv_path: Path | str,
        history: int,
        horizon: int,
        target_column: str,
        feature_columns: Optional[Sequence[str]] = None,
        station_filter: Optional[Iterable[str]] = None,
        scaler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.history = history
        self.horizon = horizon
        self.target_column = target_column

        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        frame = pd.read_csv(self.csv_path)
        if station_filter is not None:
            stations = set(station_filter)
            frame = frame[frame["Station"].isin(stations)]
        frame["DateTime"] = pd.to_datetime(frame["DateTime"])
        frame = frame.sort_values(["DateTime", "Station"]).reset_index(drop=True)

        self.stations: List[str] = list(frame["Station"].unique())
        metadata_records = (
            frame.groupby("Station")[["Latitude", "Longitude", "Elevation"]]
            .first()
            .reset_index()
        )
        self.metadata: Dict[str, StationMetadata] = {
            row.Station: StationMetadata(
                station=row.Station,
                latitude=float(row.Latitude),
                longitude=float(row.Longitude),
                elevation=float(row.Elevation) if not math.isnan(row.Elevation) else None,
            )
            for row in metadata_records.itertuples()
        }

        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        if target_column not in numeric_cols:
            raise ValueError(f"Target column {target_column!r} is not numeric")

        if feature_columns is None:
            feature_columns = [col for col in numeric_cols if col != target_column]
        self.feature_columns: List[str] = list(feature_columns)
        missing = set(self.feature_columns + [target_column]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if scaler is not None:
            frame[self.feature_columns] = scaler(frame[self.feature_columns])

        self.timestamps: List[pd.Timestamp] = sorted(frame["DateTime"].unique())
        time_index = {ts: idx for idx, ts in enumerate(self.timestamps)}

        num_times = len(self.timestamps)
        num_stations = len(self.stations)
        num_features = len(self.feature_columns)

        features = np.zeros((num_times, num_stations, num_features), dtype=np.float32)
        targets = np.zeros((num_times, num_stations), dtype=np.float32)
        mask = np.zeros((num_times, num_stations), dtype=bool)

        station_index = {station: i for i, station in enumerate(self.stations)}
        for _, row in frame.iterrows():
            t_idx = time_index[row["DateTime"]]
            s_idx = station_index[row["Station"]]
            features[t_idx, s_idx] = row[self.feature_columns].to_numpy(dtype=np.float32)
            targets[t_idx, s_idx] = float(row[target_column])
            mask[t_idx, s_idx] = True

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.mask = torch.from_numpy(mask)

        self.index_pairs: List[int] = list(
            range(0, num_times - (history + horizon) + 1)
        )

    def __len__(self) -> int:
        return len(self.index_pairs)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        start = self.index_pairs[index]
        end_history = start + self.history
        end_total = end_history + self.horizon

        features = self.features[start:end_history]
        targets = self.targets[end_history:end_total]
        mask_hist = self.mask[start:end_history]
        mask_fut = self.mask[end_history:end_total]

        return {
            "features": features,  # (history, stations, features)
            "target": targets,  # (horizon, stations)
            "history_mask": mask_hist,  # (history, stations)
            "target_mask": mask_fut,  # (horizon, stations)
        }

    def build_graph(self, k: int = 3) -> nx.Graph:
        """Construct a k-nearest neighbor graph between stations."""
        if k < 1:
            raise ValueError("k must be >= 1")

        graph = nx.Graph()
        for station, meta in self.metadata.items():
            graph.add_node(
                station,
                latitude=meta.latitude,
                longitude=meta.longitude,
                elevation=meta.elevation,
            )

        stations = list(self.metadata.keys())
        coords = np.array(
            [
                (self.metadata[s].latitude, self.metadata[s].longitude)
                for s in stations
            ]
        )
        distance_matrix = np.zeros((len(stations), len(stations)))
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                dist = haversine_distance(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        for idx, station in enumerate(stations):
            neighbors = np.argsort(distance_matrix[idx])[: k + 1]
            for neighbor_idx in neighbors:
                if neighbor_idx == idx:
                    continue
                neighbor_station = stations[neighbor_idx]
                weight = math.exp(-distance_matrix[idx, neighbor_idx])
                graph.add_edge(station, neighbor_station, weight=weight)
        return graph


def collate_graph_batches(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Stack a batch of sliding windows."""
    features = torch.stack([item["features"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    history_mask = torch.stack([item["history_mask"] for item in batch], dim=0)
    target_mask = torch.stack([item["target_mask"] for item in batch], dim=0)
    return {
        "features": features,
        "target": targets,
        "history_mask": history_mask,
        "target_mask": target_mask,
    }


def adjacency_from_graph(graph: nx.Graph) -> Tensor:
    """Create a normalized adjacency matrix from a NetworkX graph."""
    stations = list(graph.nodes)
    idx_map = {station: i for i, station in enumerate(stations)}
    adjacency = torch.zeros(len(stations), len(stations))
    for src, dst, data in graph.edges(data=True):
        i, j = idx_map[src], idx_map[dst]
        weight = float(data.get("weight", 1.0))
        adjacency[i, j] = adjacency[j, i] = weight

    degrees = adjacency.sum(dim=1)
    deg_inv_sqrt = torch.pow(degrees + 1e-6, -0.5)
    normalization = deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
    adjacency = adjacency * normalization
    return adjacency


__all__ = [
    "AirQualityDataset",
    "StationMetadata",
    "adjacency_from_graph",
    "collate_graph_batches",
    "haversine_distance",
]
