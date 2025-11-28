# Bayesian Spatio-Temporal Graph Attention Network (BST-GAT)

This repository provides a reference implementation of a **Bayesian Spatio-Temporal Graph Attention Network (BST-GAT)** for probabilistic air-quality forecasting. The architecture jointly models spatial dependencies between monitoring stations and long-range temporal dynamics while producing calibrated predictive distributions.

## Key features

- **Graph-aware spatial encoding** powered by a graph attention layer that learns adaptive inter-station influence weights.
- **Temporal transformer encoder** capturing long-range patterns and seasonal effects.
- **Bayesian inference** via variational layers delivering predictive means, variances, and Monte Carlo samples for uncertainty quantification.
- **End-to-end training script** with sliding-window data preprocessing, graph construction, posterior sampling metrics, and model checkpointing.

## Project structure

```
BST-GAT/
├── bstgat/
│   ├── __init__.py
│   ├── data.py            # Dataset, graph utilities, and batching helpers
│   ├── model.py           # BST-GAT model definition and Gaussian NLL loss
│   └── train.py           # Training / evaluation entry-point
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and usage instructions
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code assumes **PyTorch ≥ 2.0** with CUDA support optional but recommended.

## Dataset

Place the Bucharest air-quality dataset (as shown in the prompt screenshot) under a known path, e.g. `data/bucharest_air_quality.csv`. The CSV file must contain at least:

- `Station`, `DateTime`, `Latitude`, `Longitude`, `Elevation`
- Target pollutant column (e.g. `Ozone`) plus additional numeric features (`PM2.5`, `NO2`, `WindDir`, `Temperature`, `Humidity`, etc.)

Each `(Station, DateTime)` pair should be unique. Missing observations are masked automatically.

## Training

Run the training script with the desired hyper-parameters. The example below uses a 48-hour history to forecast the next 12 hours of ozone concentration.

```bash
python -m bstgat.train \
  --data data/bucharest_air_quality.csv \
  --target Ozone \
  --features PM25 NO2 WindDir WindSpd Temperature Humidity Pressure \
  --history 48 \
  --horizon 12 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-3 \
  --k-neighbors 4 \
  --save checkpoints/bst_gat.pt
```

During training the script prints the variational loss, RMSE, MAE, and empirical 95% coverage of the predictive intervals estimated via Monte Carlo sampling.

## Probabilistic forecasting

After training, the saved checkpoint contains the model weights, normalized adjacency matrix, station metadata, and window configuration. To draw predictive samples for downstream analysis:

```python
import torch
from bstgat import BayesianSpatioTemporalGAT

checkpoint = torch.load("checkpoints/bst_gat.pt", map_location="cpu")
model = BayesianSpatioTemporalGAT(
    num_nodes=len(checkpoint["stations"]),
    feature_dim=len(checkpoint["feature_columns"]),
    history=checkpoint["history"],
    horizon=checkpoint["horizon"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    output = model(features, checkpoint["adjacency"])
    samples = output.predictive_samples(num_samples=500)
    mean_forecast = output.mean
    lower, upper = samples.quantile(0.025, dim=0), samples.quantile(0.975, dim=0)
```

## License

MIT License
