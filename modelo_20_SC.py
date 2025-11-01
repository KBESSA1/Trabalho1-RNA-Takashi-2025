#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# Trabalho 1 — Modelo 1: 20 Superclasses (CIFAR-100, coarse)
# -------------------------------------------------------------
# Requisitos:
#  - split treino/val/test
#  - curvas de loss (treino/val) em PNG
#  - checkpoint do melhor modelo (val loss)
#  - early stopping
#  - classification_report no teste
#
# Compatibilidade:
#  - NÃO uso target_type no CIFAR100; leio coarse_targets se existir.
#  - Se não existir, reconstruo o rótulo coarse via nomes oficiais (mapping 100→20).
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


# --------------------------------------------
# Dataset wrapper só com o rótulo "coarse"
# (com fallback caso não exista coarse_targets)
# --------------------------------------------
class CIFAR100CoarseOnly(Dataset):
    """
    Retorna (imagem_transformada, coarse_label[0..19]).
    Se a torchvision não tiver 'coarse_targets', reconstruo via nomes oficiais.
    """
    def __init__(self, root: str, train: bool, transform):
        self.transform = transform
        self.base = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=None
        )

        if hasattr(self.base, "coarse_targets"):
            self.coarse = list(self.base.coarse_targets)
        else:
            # 1) nomes finos e índice
            if hasattr(self.base, "classes") and hasattr(self.base, "class_to_idx"):
                fine_names = list(self.base.classes)       # e.g. 'apple', 'aquarium_fish', ...
                name_to_idx = dict(self.base.class_to_idx)
            else:
                raise RuntimeError("CIFAR100 sem 'classes'/'class_to_idx' — não consigo reconstruir coarse.")

            # 2) mapping canônico coarse → fines (nomes do CIFAR-100)
            coarse_to_fines = {
                "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
                "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
                "food containers": ["bottle", "bowl", "can", "cup", "plate"],
                "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
                "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
                "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
                "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
                "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
                "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
                "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
                "people": ["baby", "boy", "girl", "man", "woman"],
                "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                # <-- árvores: atenção ao sufixo "_tree" na taxonomia do torchvision
                "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }

            # 3) ordem canônica das 20 superclasses
            coarse_order = [
                "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
                "household electrical devices", "household furniture", "insects", "large carnivores",
                "large man-made outdoor things", "large natural outdoor scenes",
                "large omnivores and herbivores", "medium-sized mammals", "non-insect invertebrates",
                "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
            ]
            coarse_index = {name: i for i, name in enumerate(coarse_order)}

            # 4) tabela fine_idx -> coarse_id
            fine_to_coarse = [None] * len(fine_names)
            # alias para variações comuns
            alias = {
                # plurais e hifens
                "orchids": "orchid", "poppies": "poppy", "roses": "rose",
                "sunflowers": "sunflower", "tulips": "tulip",
                "sweet peppers": "sweet_pepper", "lawn-mower": "lawn_mower",
                "pickup truck": "pickup_truck",
                # caso anterior:
                "computer_keyboard": "keyboard",
                # árvores sem sufixo (caso apareçam assim em outra lista)
                "maple": "maple_tree", "oak": "oak_tree", "palm": "palm_tree",
                "pine": "pine_tree", "willow": "willow_tree",
            }

            for cname in coarse_order:
                cid = coarse_index[cname]
                for fname in coarse_to_fines[cname]:
                    if fname not in name_to_idx and fname in alias:
                        fname = alias[fname]
                    idx = name_to_idx[fname]
                    fine_to_coarse[idx] = cid

            if any(v is None for v in fine_to_coarse):
                missing = [i for i, v in enumerate(fine_to_coarse) if v is None]
                raise RuntimeError(f"Falha ao mapear alguns rótulos finos para coarse: {missing}")

            # 5) fine_target → coarse_id para cada amostra
            if hasattr(self.base, "targets"):
                fine_targets = list(self.base.targets)
            elif hasattr(self.base, "fine_targets"):
                fine_targets = list(self.base.fine_targets)
            else:
                raise RuntimeError("CIFAR100 sem 'targets'/'fine_targets' — não consigo reconstruir coarse.")

            self.coarse = [int(fine_to_coarse[f]) for f in fine_targets]

    def __len__(self):
        return len(self.base.data)

    def __getitem__(self, idx):
        img = T.functional.to_pil_image(self.base.data[idx])
        if self.transform:
            img = self.transform(img)
        y = int(self.coarse[idx])
        return img, y


class CoarseHeadModel(nn.Module):
    """MobileNetV3 Small com 20 saídas (superclasses)."""
    def __init__(self, num_coarse: int = 20):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_coarse)
        self.net = base
    def forward(self, x): return self.net(x)


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

def plot_losses(train_losses: List[float], val_losses: List[float], out_png: str, title: str):
    plt.figure(figsize=(9,5))
    plt.plot(train_losses, label="Loss Treino")
    plt.plot(val_losses, label="Loss Validação")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./outputs")
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

    train_full = CIFAR100CoarseOnly(root=args.data_dir, train=True,  transform=tf_train)
    test_ds    = CIFAR100CoarseOnly(root=args.data_dir, train=False, transform=tf_test)

    n = len(train_full); val = int(args.val_split*n); train = n - val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_full, [train, val], generator=g)
    print(f"Tamanhos: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CoarseHeadModel(num_coarse=20).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {n_params:,}")
    assert n_params <= 10_000_000, "Modelo excede 10M de parâmetros!"

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr)

    ckpt = os.path.join(args.out_dir, "best_modelo_20_SC.pth")
    png  = os.path.join(args.out_dir, "loss_modelo_20_SC.png")
    repf = os.path.join(args.out_dir, "report_modelo_20_SC.txt")

    best = float("inf"); best_w = copy.deepcopy(model.state_dict()); patience=0
    tr_hist, vl_hist = [], []

    print("\nIniciando treinamento (coarse20)...")
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

    model.load_state_dict(best_w)
    plot_losses(tr_hist, vl_hist, png, "Curvas de Loss — coarse20")
    print(f"Curvas de loss salvas em: {png}")

    print("Avaliando no conjunto de teste (coarse20)...")
    yt, yp = predict(model, test_loader, device)
    names = [f"super_{i}" for i in range(20)]
    rep = classification_report(yt, yp, target_names=names, zero_division=0)
    with open(repf, "w", encoding="utf-8") as f: f.write(rep)
    print(rep)
    print(f"Relatório salvo em: {repf}")
    print(f"Checkpoint salvo em: {ckpt}")
    print("Concluído.")


if __name__ == "__main__":
    main()
