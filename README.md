# Helmholtz-PINN with RAR

> A compact, reproducible implementation of a 1D Helmholtz solver using a Physics-Informed Neural Network (PINN) with Residual-based Adaptive Refinement (RAR).
>
> The code trains a small fully-connected network to satisfy the 1D Helmholtz ODE `d²p/dx² + k² p = 0` on `x ∈ [0, L]` with Dirichlet BCs `p(0)=A`, `p(L)=B`.

---

## Features

- Single-file PINN implementation.
- Residual-based Adaptive Refinement (RAR): candidate pool → evaluate residuals → replace worst collocation points by top residual candidates each cycle.
- L-BFGS inner solver per RAR cycle + optional final global L-BFGS pass.
- Trial solution enforces Dirichlet boundary conditions exactly.
- Pre/post final predictions and plotting of collocation evolution and predicted vs analytical solution.

---

## Quick start

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate        # Windows
```

2. Install minimal requirements:

```bash
pip install torch numpy matplotlib
```

3. Save the file (e.g. `helmholtz_pinn_rar.py`)

4. Run the example:

```bash
python helmholtz_pinn_rar.py
```

You should see printed progress for each RAR cycle and two matplotlib figures:
- collocation point distribution across cycles,
- pre/post PINN predictions vs the analytical Helmholtz solution.

---

## Important implementation notes

- **Domain mapping**: inputs `xh` are normalized to `[-1, 1]`. The physical coordinate `x_phys` is computed as `x_phys = (xh + 1) * (L/2)`. Because of this mapping, second derivative w.r.t. normalized `xh` is scaled by `(2/L)^2` — the code multiplies `d2p` by `(2.0 / L)**2` in the residual to recover the physical second derivative.

- **Trial solution and BCs**: `trial(xh)` constructs a solution that satisfies `p(0)=A` and `p(L)=B` exactly by adding a network-dependent correction `x*(L-x)*p_nn(x)`. This is a standard approach to embed Dirichlet BCs in PINNs (no loss term for boundary points required).

- **Residual computation**: uses PyTorch `autograd` to compute `dp/dxh` and `d2p/dxh2` with `create_graph=True`, then the residual is `scaled_d2p + k**2 * p`.

- **RAR loop**:
  - Keep `N_colloc` collocation points.
  - In each cycle, run an L-BFGS optimizer on the current collocation set (`iters_per_cycle` iterations).
  - Sample `cand_pool` random candidate points, evaluate residual magnitudes, pick top `replace_N = floor(refine_frac * N_colloc)` candidates, and replace the worst `replace_N` points from the current collocation set.
  - This greedily concentrates collocation points where residuals are largest (RAR-G style).

- **L-BFGS**:
  - The script uses PyTorch `LBFGS` optimizer (per-cycle and optionally a final global pass). You can tune `lr`, `max_iter` etc. via `lbfgs_options`.

- **Device**:
  - The script currently sets `device = torch.device('cpu')` — to use GPU, change to `torch.device('cuda')` and ensure your PyTorch build supports CUDA.

---

## Recommended hyperparameters / tips

- Network: `[1, 120, 120, 120, 1]` (yours). For harder Helmholtz / higher frequency `k`, increase width/depth or use positional/Fourier features.
- Collocation points: `N_colloc = 2000–5000` works for many 1D tests; increase for higher accuracy.
- Candidate pool: `cand_pool = 20000–100000` — larger pool yields better sampling at the cost of eval time.
- `refine_frac = 0.1–0.3`: fraction of collocation points replaced each cycle.
- `N_cycles = 3–10` depending on problem complexity.
- L-BFGS: the algorithm is sensitive to tolerances; keep `tolerance_grad` and `tolerance_change` small if you expect precise convergence.
- Activation: the script uses `sin` — good for wave/Helmholtz-like solutions. For other problems you might try `tanh` or periodic activations / Fourier feature encodings.

---

## Output files & plots

- The script returns a `result` dict with:
  - `colloc_history`: list of arrays (physical x) — collocation sets after each cycle.
  - `x_phys_plot`: `x` grid used for plotting (physical domain).
  - `p_pre`: PINN prediction before final global L-BFGS.
  - `p_post`: PINN prediction after final global L-BFGS.
  - `p_true`: analytical Helmholtz solution.
  - `k`: the wavenumber.

- Plots produced:
  1. Collocation-point distribution over cycles (each cycle plotted on a different y level).
  2. `p_true` vs `p_pre` vs `p_post`.

---

## Troubleshooting & common pitfalls

- **L-BFGS raises errors / stalls**: try lowering `max_iter` and then run multiple cycles; or switch to Adam pretraining followed by L-BFGS. L-BFGS in PyTorch expects closures that return the loss and compute gradients.
- **Poor accuracy for high k (many wavelengths)**: Helmholtz can be challenging for PINNs at high frequencies — increase network capacity, use sinusoidal activations or Fourier feature encodings, or use domain decomposition / multi-scale approaches.
- **Autograd errors**: ensure that all tensors used in gradient computations require `requires_grad=True` (the code sets this before each residual eval).
- **GPU support**: change `device` and `.to(device)` for tensors and model. Candidate pool `cand_pool` large → GPU memory may become a bottleneck.

---

## Files 

```
README.md                    # this file
helmholtz_pinn_rar.py        # main script (your code)
requirements.txt             # torch, numpy, matplotlib (optional)
examples/                    # small example configs / output figures
notebooks/                   # optional Jupyter notebooks for experiments
LICENSE                      # e.g. MIT
```

`requirements.txt` (example):

```
torch>=1.12
numpy
matplotlib
```

---

## Citations


**Repository citation (example):**

```bibtex
@software{yourname_helmholtz_pinn_rar_2025,
  author       = {Your Name},
  title        = {Helmholtz-PINN with Residual-based Adaptive Refinement (RAR)},
  year         = {2025},
  url          = {https://github.com/yourusername/helmholtz-pinn-rar},
  version      = {v0.1},
  note         = {Python code},
}
```

**Foundational PINN paper :**

```bibtex
@article{raissi2019physicsinformed,
  title   = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal = {Journal of Computational Physics},
  volume  = {378},
  pages   = {686--707},
  year    = {2019},
  doi     = {10.1016/j.jcp.2018.10.045}
}
```

**Gradient-enhanced PINNs :**

```bibtex
@article{yu2022gradient,
  title   = {Gradient-enhanced physics-informed neural networks for forward and inverse {PDE} problems},
  author  = {Yu, Jeremy and Lu, Lu and Meng, Xuhui and Karniadakis, George Em},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {393},
  pages   = {114823},
  year    = {2022},
  doi     = {10.1016/j.cma.2022.114823}
}
```

**Sampling / RAR study :**

```bibtex
@article{wu2023comprehensive,
  title   = {A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks},
  author  = {Wu, Chenxi and Zhu, Min and Tan, Qinyang and Kartha, Yadhu and Lu, Lu},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {403},
  pages   = {115671},
  year    = {2023},
  doi     = {10.1016/j.cma.2022.115671}
}
```

---



