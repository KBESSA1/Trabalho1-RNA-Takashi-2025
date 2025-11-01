#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# Trabalho 1 — Modelo 2: 100 Classes (CIFAR-100, fine)
# -------------------------------------------------------------
# Requisitos:
#  - split treino/val/test
#  - curvas de loss (treino/val) em PNG
#  - checkpoint do melhor modelo (val loss)
#  - early stopping
#  - classification_report no teste
#
# Compatibilidade:
#  - NÃO uso target_type; o CIFAR100 padrão já retorna fine labels em .targets
# -------------------------------------------------------------

import os
import copy
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def seed_everything(seed: int = 42):
    """Fixar seeds para reprodutibilidade."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------
# Dataset wrapper com rótulo fino (100 classes)
# (sem target_type; usa .data e .targets do torchvision)
# ----------------------------------------------------
class CIFAR100FineOnly(Dataset):
    """Retorna (imagem_transformada, fine_label[0..99])."""
    def __init__(self, root: str, train: bool, transform):
        self.transform = transform
        self.base = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=None
        )
        # fine labels vem em .targets
        if hasattr(self.base, "targets"):
            self.fine = list(self.base.targets)
        elif hasattr(self.base, "fine_targets"):
            self.fine = list(self.base.fine_targets)
        else:
            raise RuntimeError("CIFAR100 sem 'targets'/'fine_targets' — não consigo ler fine labels.")

    def __len__(self):
        return len(self.base.data)

    def __getitem__(self, idx):
        img = T.functional.to_pil_image(self.base.data[idx])
        if self.transform:
            img = self.transform(img)
        y = int(self.fine[idx])
        return img, y


# -----------------------------------
# Modelo: MobileNetV3-Small < 10M
# -----------------------------------
class Fine100Model(nn.Module):
    """MobileNetV3 Small com 100 saídas (classes finas)."""
    def __init__(self, num_fine: int = 100):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_fine)
        self.net = base
    def forward(self, x): return self.net(x)


# ---------------------------
# Laços de treino/val/test
# ---------------------------
def train_epoch(model, loader, opt, crit, device):
    model.train(); running = 0.0
    for x, y in loader:
        x=x.to(device); y=y.to(device)
        opt.zero_grad(); out=model(x)
        loss=crit(out, y); loss.backward(); opt.step()
        running += loss.item()*x.size(0)
    return running/len(loader.dataset)

def eval_epoch(model, loader, crit, device):
    model.eval(); running = 0.0
    with torch.no_grad():
        for x, y in loader:
            x=x.to(device); y=y.to(device)
            out=model(x); loss=crit(out, y)
            running += loss.item()*x.size(0)
    return running/len(loader.dataset)

def predict(model, loader, device):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            x=x.to(device); out=model(x)
            pred=out.argmax(1).cpu().tolist()
            yt.extend(y.tolist()); yp.extend(pred)
    return yt, yp


# -------------------------
# Plot das curvas de loss
# -------------------------
def plot_losses(train_losses: List[float], val_losses: List[float], out_png: str, title: str):
    plt.figure(figsize=(9,5))
    plt.plot(train_losses, label="Loss Treino")
    plt.plot(val_losses, label="Loss Validação")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()


# --------------
# Entrypoint
# --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir",  default="./outputs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    normalize = T.Normalize(mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761))
    tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    tf_test  = T.Compose([T.ToTensor(), normalize])

    # datasets / splits
    train_full = CIFAR100FineOnly(root=args.data_dir, train=True,  transform=tf_train)
    test_ds    = CIFAR100FineOnly(root=args.data_dir, train=False, transform=tf_test)

    n = len(train_full); val = int(args.val_split*n); train = n - val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_full, [train, val], generator=g)
    print(f"Tamanhos: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # modelo / otimizador / loss
    model = Fine100Model(num_fine=100).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {n_params:,}")
    assert n_params <= 10_000_000, "Modelo excede 10M de parâmetros!"

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr)

    ckpt = os.path.join(args.out_dir, "best_modelo_100_C.pth")
    png  = os.path.join(args.out_dir, "loss_modelo_100_C.png")
    repf = os.path.join(args.out_dir, "report_modelo_100_C.txt")

    best = float("inf"); best_w = copy.deepcopy(model.state_dict()); patience=0
    tr_hist, vl_hist = [], []

    print("\nIniciando treinamento (fine100)...")
    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, crit, device)
        vl = eval_epoch(model, val_loader, crit, device)
        tr_hist.append(tr); vl_hist.append(vl)
        print(f"Época {ep:03d} | train={tr:.4f} | val={vl:.4f}")

        if vl < best - 1e-6:
            best = vl; best_w = copy.deepcopy(model.state_dict())
            torch.save(best_w, ckpt); patience=0
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping ativado."); break

    # restaura melhor
    model.load_state_dict(best_w)
    plot_losses(tr_hist, vl_hist, png, "Curvas de Loss — fine100")
    print(f"Curvas de loss salvas em: {png}")

    # avaliação no teste
    print("Avaliando no conjunto de teste (fine100)...")
    yt, yp = predict(model, test_loader, device)
    target_names = [f"class_{i}" for i in range(100)]
    rep = classification_report(yt, yp, target_names=target_names, zero_division=0)
    with open(repf, "w", encoding="utf-8") as f: f.write(rep)
    print(rep)
    print(f"Relatório salvo em: {repf}")
    print(f"Checkpoint salvo em: {ckpt}")
    print("Concluído.")


if __name__ == "__main__":
    main()
