import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def solve_helmholtz_pinn_rar_with_preds(
    k,
    A,
    B,
    layers,
    N_colloc=2000,            # you can scale up to 2000 for actual runs
    N_cycles=5,
    iters_per_cycle=100,
    cand_pool=30000,
    refine_frac=0.1,
    c=343.0,
    L=1.0,
    final_lbfgs=True,
    lbfgs_options=None
):
    t0 = time.perf_counter()
    device = torch.device('cpu')

    # --- Define the PINN ---
    class PINN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.act = torch.sin
            self.net = nn.ModuleList(
                nn.Linear(layers[i], layers[i+1])
                for i in range(len(layers)-1)
            )
            for lin in self.net:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

        def forward(self, x):
            y = x
            for lin in self.net[:-1]:
                y = self.act(lin(y))
            return self.net[-1](y)

    net = PINN(layers).to(device)

    # Trial solution that enforces boundary conditions
    def trial(xh):
        x_phys = (xh + 1) * (L / 2)  # map from [-1,1] → [0,L]
        p_nn = net(xh)
        return (1 - x_phys / L) * A + (x_phys / L) * B + x_phys * (L - x_phys) * p_nn

    # PDE residual: (d²p/dx² + k² p = 0)
    def residual(xh):
        p = trial(xh)
        dp  = torch.autograd.grad(p, xh,
                                  grad_outputs=torch.ones_like(p),
                                  create_graph=True)[0]
        d2p = torch.autograd.grad(dp, xh,
                                  grad_outputs=torch.ones_like(dp),
                                  create_graph=True)[0]
        return (2.0 / L)**2 * d2p + k**2 * p

    # Initialize collocation points in normalized [-1,1]
    x_colloc = torch.rand(N_colloc, 1, device=device) * 2 - 1
    replace_N = int(refine_frac * N_colloc)

    colloc_history = []  # Will hold physical x-coordinates each cycle

    # --- RAR Cycles (without final global L-BFGS) ---
    for cycle in range(1, N_cycles + 1):
        lbfgs_cycle = torch.optim.LBFGS(
            net.parameters(),
            lr=0.1,
            max_iter=iters_per_cycle,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50
        )

        def closure_cycle():
            lbfgs_cycle.zero_grad()
            x_req = x_colloc.clone().requires_grad_(True)
            res = residual(x_req)
            loss = torch.mean(res**2)
            loss.backward()
            return loss

        lbfgs_cycle.step(closure_cycle)

        # Record collocation set (in physical domain) after this cycle
        x_phys = ((x_colloc + 1) * (L / 2)).detach().cpu().numpy().flatten()
        colloc_history.append(x_phys)

        # Residual‐based refinement:
        x_cand = (torch.rand(cand_pool, 1, device=device) * 2 - 1).requires_grad_(True)
        r_cand = residual(x_cand).abs().flatten().detach()
        idx_top = torch.topk(r_cand, replace_N).indices
        x_add = x_cand[idx_top].detach()

        x_req2 = x_colloc.clone().detach().requires_grad_(True)
        r_coll = residual(x_req2).abs().flatten().detach()
        idx_keep = torch.topk(r_coll, N_colloc - replace_N).indices
        x_colloc = torch.cat([x_colloc[idx_keep], x_add], dim=0)

        print(f"Cycle {cycle:2d} completed – recorded collocation points.")

    # --- Before final global L-BFGS: record a “pre‐final” prediction ---
    # Build a fine grid in normalized space for plotting
    x_plot = torch.linspace(-1, 1, 800).unsqueeze(-1).to(device)  # 800 points in [-1,1]
    with torch.no_grad():
        p_pre = trial(x_plot).cpu().numpy().flatten()

    # Map x_plot → physical domain for both PINN and analytical
    x_phys_plot = ((x_plot + 1) * (L / 2)).cpu().numpy().flatten()

    # If requested, run a final global L-BFGS pass:
    if final_lbfgs:
        opts = dict(lr=0.1,
                    max_iter=2000,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-12,
                    history_size=50)
        if lbfgs_options:
            opts.update(lbfgs_options)

        lbfgs_final = torch.optim.LBFGS(net.parameters(), **opts)

        def closure_final():
            lbfgs_final.zero_grad()
            x_in = x_colloc.clone().detach().requires_grad_(True)
            res = residual(x_in)
            loss = torch.mean(res**2)
            loss.backward()
            return loss

        lbfgs_final.step(closure_final)
        print("Final global L-BFGS completed.")

    # Record a “post‐final” prediction:
    with torch.no_grad():
        p_post = trial(x_plot).cpu().numpy().flatten()

    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.1f}s")

    # Analytical (closed‐form) Helmholtz solution on [0,L]:
    x_np   = x_phys_plot
    p_true = (A * np.cos(k * x_np)
              + ((B - A * np.cos(k * L)) / np.sin(k * L)) * np.sin(k * x_np))

    return {
        "colloc_history": colloc_history,
        "x_phys_plot": x_np,
        "p_pre": p_pre,
        "p_post": p_post,
        "p_true": p_true,
        "k": k
    }


if __name__ == "__main__":
    # Example parameters (you can restore your original sizes once verified)
    freq  = 1529.0
    c_val = 340.0
    k_val = 2 * np.pi * freq / c_val
    A_val = 1.0
    B_val = -1.0

    result = solve_helmholtz_pinn_rar_with_preds(
        k=k_val,
        A=A_val,
        B=B_val,
        layers=[1, 120, 120, 120, 1],  # your original network
        N_colloc=5000,                  # or 2000 for full accuracy
        N_cycles=5,
        iters_per_cycle=2000,
        cand_pool=30000,
        refine_frac=0.2,
        L=1.0,
        final_lbfgs=True
    )

    # --- Plot 1: Collocation sets across cycles ---
    plt.figure(figsize=(10, 6))
    for idx, x_vals in enumerate(result["colloc_history"]):
        y_vals = np.full_like(x_vals, idx + 1)
        plt.scatter(x_vals, y_vals, s=10, alpha=0.6, label=f'Cycle {idx + 1}')
    plt.xlabel('Physical x-coordinate')
    plt.ylabel('RAR Cycle Number')
    plt.title('Collocation-Point Distribution Over RAR Cycles')
    plt.yticks(range(1, len(result["colloc_history"]) + 1))
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # --- Plot 2: Predictions before vs. after final L-BFGS (plus analytical) ---
    plt.figure(figsize=(10, 6))
    x_np   = result["x_phys_plot"]
    p_true = result["p_true"]
    p_pre  = result["p_pre"]
    p_post = result["p_post"]

    plt.plot(x_np, p_true,  label='Analytical', color='black', linewidth=2)
    plt.plot(x_np, p_pre,   '--', label='PINN solution without RAR', linewidth=2)
    plt.plot(x_np, p_post,  '-.', label='PINN solution with RAR', linewidth=2)
    plt.xlabel('x (physical domain)')
    plt.ylabel('p(x)')
    plt.title(f'Helmholtz PINN Predictions (k = {result["k"]:.2f})')
    plt.legend()
    plt.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.show()
