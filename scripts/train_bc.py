"""Train behaviour-cloning policies on the generated dataset (several seeds).

Split by EPISODE (whole unseen scenes in validation), standardise inputs with training-set statistics,
MSE loss on the movement, binary cross-entropy on the gripper.

Usage:  python scripts/train_bc.py clean_12 [--seeds 0 1 2] [--epochs 30] [--tag _time]
Output: models/bc_<clip>_seed<k>.pt, outputs/policy/bc_<clip>_training.png
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.policy.model import BCPolicy  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "policy"


def train_one(data, train_eps, val_eps, seed, epochs, batch, lr, grip_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ep = data["episode"]
    tr, va = np.isin(ep, train_eps), np.isin(ep, val_eps)
    X = torch.as_tensor(data["obs"], dtype=torch.float32)
    M = torch.as_tensor(data["act_move"], dtype=torch.float32)
    G = torch.as_tensor(data["act_grip"], dtype=torch.float32)

    model = BCPolicy(n_obs=X.shape[1])
    model.set_normalisation(X[tr].mean(0), X[tr].std(0))          # training statistics only
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    idx_tr = np.flatnonzero(tr)
    hist = {"train": [], "val": [], "val_grip_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(idx_tr)
        total = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            move, logit = model(X[b])
            loss = mse(move, M[b]) + grip_weight * bce(logit, G[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            move, logit = model(X[va])
            val = (mse(move, M[va]) + grip_weight * bce(logit, G[va])).item()
            acc = ((logit > 0).float() == G[va]).float().mean().item()
        hist["train"].append(total / len(idx_tr))
        hist["val"].append(val)
        hist["val_grip_acc"].append(acc)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    seed {seed} epoch {epoch:3d}  train {hist['train'][-1]:.4f}  val {val:.4f}  "
                  f"val gripper accuracy {100 * acc:.1f}%")
    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grip-weight", type=float, default=0.5, help="weight of the gripper loss")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--tag", default="", help="dataset/model variant, e.g. _time for the time-based expert")
    args = ap.parse_args()

    name = f"{args.clip}{args.tag}"
    d = np.load(PROCESSED / f"dataset_{name}.npz")
    data = {k: d[k] for k in ["obs", "act_move", "act_grip", "episode"]}
    episodes = np.unique(data["episode"])
    rng = np.random.default_rng(12345)                       # the SAME split for every seed
    val_eps = rng.choice(episodes, size=max(1, int(len(episodes) * args.val_fraction)), replace=False)
    train_eps = np.setdiff1d(episodes, val_eps)
    print(f"{args.clip}: {len(data['obs'])} steps, {len(episodes)} episodes -> "
          f"{len(train_eps)} train / {len(val_eps)} validation episodes")

    MODELS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed in args.seeds:
        t0 = time.time()
        model, hist = train_one(data, train_eps, val_eps, seed, args.epochs, args.batch, args.lr, args.grip_weight)
        out = MODELS / f"bc_{name}_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "n_obs": data["obs"].shape[1],
                    "config": vars(args), "seed": seed, "history": hist,
                    "val_episodes": val_eps.tolist()}, out)
        print(f"  seed {seed}: saved {out.name}  ({time.time() - t0:.0f} s)  "
              f"final val loss {hist['val'][-1]:.4f}, gripper accuracy {100 * hist['val_grip_acc'][-1]:.1f}%")
        axes[0].plot(hist["train"], label=f"train, seed {seed}")
        axes[0].plot(hist["val"], "--", label=f"val, seed {seed}")
        axes[1].plot(np.array(hist["val_grip_acc"]) * 100, label=f"seed {seed}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation gripper accuracy (%)")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"bc_{name}_training.png", dpi=110)
    print(f"\nTraining curves: {OUT_DIR / f'bc_{name}_training.png'}")
