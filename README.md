# DCST

Minimal research implementation of DCST for 10-shot hyperspectral image
classification on Indian Pines.

## Modules

- **Frequency Token Representation (FTR)** fuses a whitened PCA-domain
  convolutional path with a low-frequency raw-spectrum DCT path.
- **Foldable Query-Key Re-parameterization (QKR)** augments each query and key
  projection with gated full-rank and low-rank linear paths. These paths are
  analytically folded into one linear projection for deployment.
- **Center-Token Top-k Fusion (CTF)** uses low-frequency spectral similarity to
  select reliable spatial neighbors. It modifies only the center-query row in
  the first Transformer block.

## Directory

```text
DCST_release/
|-- configs/ip.json
|-- data/indian_pines_10shot.mat
|-- dcst/
|   |-- __init__.py
|   |-- data.py
|   `-- model.py
|-- requirements.txt
`-- train.py
```

The MATLAB data file contains:

- `input`: hyperspectral image with shape `145 x 145 x 200`
- `TR`: ten-shot training mask
- `TE`: test mask

## Environment

```bash
pip install -r requirements.txt
```

## Training

Five-seed experiment:

```bash
python -u train.py --device cuda:0
```

Quick pipeline check:

```bash
python -u train.py --device cuda:0 --smoke-test
```

Results are written to `results/run_TIMESTAMP/`. Each completed seed stores a
folded deployment checkpoint named `seed_N_deploy.pth`. The checkpoint contains
the configuration and a state dictionary for `DCST(config, deploy=True)`.

## Default IP Setting

- patch size: `17 x 17`
- PCA components: `200`
- DCT coefficients: `32`
- token dimension: `64`
- Transformer blocks: `3`
- attention heads: `20`
- dimension per head: `64`
- QKR rank: `8`
- CTF neighbors: `64`
- training and evaluation batch size: `64`
- optimizer: Adam, learning rate `1e-3`
- training epochs: `50`

## Release Checklist

The source code is provided under the MIT License. Replace the generic
copyright holder in `LICENSE` with the final author or institution name before
publication. The code license does not grant redistribution rights for the
dataset. Verify the data terms before uploading the MATLAB file, and add the
paper citation and dataset acknowledgment to the final repository.
