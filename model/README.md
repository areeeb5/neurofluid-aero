# model/

Stage 1 of NeuroFluid-Aero: the FNO trunk and Head A (surface Cp), trained on the XFOIL data.
Data contract: `data_spec.md` ("Stage 1 (Head A) training data contract"). Needs PyTorch (CUDA build was used) and the grids from `python data_generation/grid_encoding.py`.

| file | what it is |
|---|---|
| `fno.py` | `SpectralConv2d`, 4 Fourier layers (width 32, 12 modes), `HeadA`, `Stage1Net` (2,384,865 parameters) |
| `data_stage1.py` | Loads the converged cases, assembles the 6-channel input on the fly, batched CL-from-Cp |
| `test_fno.py` | Unit tests: shapes, translation equivariance, grid-sampling alignment vs numpy (with a known-bad), gradients, CL-from-Cp parity |
| `overfit_single_sample.py` | The required overfit-a-single-sample gate, with an lr = 0 control that must fail |
| `train_stage1.py` | Training. Train on `train`, early-stop on `val`, test splits touched once at the end. `--seed`, `--tag` |
| `active_selection_draft.py` | DRAFT ranking of the airfoils the model is most wrong about. Review only |

Run order:

```bash
python model/test_fno.py
python model/overfit_single_sample.py
python model/train_stage1.py --seed 42
```

Outputs go to `overnight/` (checkpoints are gitignored).
