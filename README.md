# IR spectrum prediction benchmark

Code for *Benchmarking Molecular Representations and Model Architectures for Gas-Phase Infrared Spectrum Prediction*.

## Data

The spectra are not redistributed here. The pipeline expects a `Dataset/` folder next to the code, or one or two levels above it, containing

- `spectra_normalized.xlsx`, one row per compound, with five metadata columns (including `NIST ID` and `Up or downsampled?`) followed by one column per wavenumber
- `MorganFP_2048.xlsx`, two metadata columns (including `NIST ID`) followed by the 2,048 fingerprint bits, in the same compound order
- `SMILES.xlsx`, with `NIST ID` and `SMILES` columns

## Running

```bash
pip install -r requirements.txt
python benchmark.py
```

Settings are in `config.py`. Featurizations are cached in `cache/`. For every model and split, `checkpoints/` receives the tuned hyperparameters (`best_*.json`), the held-out predictions of each fold (`oof_*.npy`), predictions for all compounds and the trained weights from the tuning run (`predall_*.npy`, `model_*`), and the split indices, compound IDs and wavenumber axis.

The graph models need DGL and DGL-LifeSci, and the descriptor models need `mordredcommunity`. If either is missing, those models are skipped.

## Layout

| File | Contents |
|---|---|
| `config.py` | settings, seeds, data location |
| `data.py` | loading, filtering and normalization of the spectra and inputs |
| `splits.py` | random and Butina cluster splits |
| `metrics.py` | SID and SIS, used for model selection |
| `nets.py` | dense and sequence networks, training loop |
| `benchmark.py` | random forests, baselines, graph and descriptor models, tuning, and cross-validated training |
