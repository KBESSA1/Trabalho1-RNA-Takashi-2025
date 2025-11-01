#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# Trabalho 1 — Modelo 3: Multihead (100 classes + 20 superclasses)
# (opcional: --ten-superclasses-subset para 100 + 10 superclasses)
# -------------------------------------------------------------

import os, copy, argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def seed_everything(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TEN_COARSE_KEEP = set(range(10))  # 0..9

class CIFAR100FineAndCoarse(Dataset):
    """Retorna (img, fine[0..99], coarse[0..(K-1)]) sem usar target_type."""
    def __init__(self, root: str, train: bool, transform, ten_superclasses_subset: bool = False):
        self.transform = transform
        self.base = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=None)

        # fine targets
        if hasattr(self.base, "targets"):
            fine_targets = list(self.base.targets)
        elif hasattr(self.base, "fine_targets"):
            fine_targets = list(self.base.fine_targets)
        else:
            raise RuntimeError("CIFAR100 sem 'targets'/'fine_targets'.")

        # nomes finos e índice
        if not (hasattr(self.base, "classes") and hasattr(self.base, "class_to_idx")):
            raise RuntimeError("CIFAR100 sem 'classes'/'class_to_idx'.")
        name_to_idx = dict(self.base.class_to_idx)

        # coarse → fine (nomes canônicos)
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
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        }
        coarse_order = [
            "aquatic mammals","fish","flowers","food containers","fruit and vegetables",
            "household electrical devices","household furniture","insects","large carnivores",
            "large man-made outdoor things","large natural outdoor scenes","large omnivores and herbivores",
            "medium-sized mammals","non-insect invertebrates","people","reptiles","small mammals",
            "trees","vehicles 1","vehicles 2",
        ]
        coarse_index = {n:i for i,n in enumerate(coarse_order)}
        alias = {
            "sweet peppers":"sweet_pepper","lawn-mower":"lawn_mower","pickup truck":"pickup_truck",
            "maple":"maple_tree","oak":"oak_tree","palm":"palm_tree","pine":"pine_tree","willow":"willow_tree",
            "computer_keyboard":"keyboard",
        }

        fine_to_coarse = [None]*100
        for cname in coarse_order:
            cid = coarse_index[cname]
            for fname in coarse_to_fines[cname]:
                if fname not in name_to_idx and fname in alias:
                    fname = alias[fname]
                fine_to_coarse[name_to_idx[fname]] = cid
        if any(v is None for v in fine_to_coarse):
            miss = [i for i,v in enumerate(fine_to_coarse) if v is None]
            raise RuntimeError(f"Falha ao mapear fine→coarse: {miss}")

        coarse_all = [int(fine_to_coarse[f]) for f in fine_targets]
        self.ten_subset = ten_superclasses_subset
        if not self.ten_subset:
            self.indices = list(range(len(self.base)))
            self.coarse  = coarse_all
        else:
            keep = TEN_COARSE_KEEP
            self.indices = [i for i,c in enumerate(coarse_all) if c in keep]
            remap = {orig:new for new,orig in enumerate(sorted(keep))}
            self.coarse = [remap[coarse_all[i]] for i in self.indices]
        self.fine = fine_targets

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real = self.indices[idx]
        img = T.functional.to_pil_image(self.base.data[real])
        if self.transform: img = self.transform(img)
        y_fine = int(self.fine[real])
        y_coarse = int(self.coarse[idx]) if self.ten_subset else int(self.coarse[real])
        return img, y_fine, y_coarse

def make_splits(ds: Dataset, val_split: float, seed: int = 42) -> Tuple[Dataset, Dataset]:
    n=len(ds); val=int(val_split*n); train=n-val
    g=torch.Generator().manual_seed(seed)
    return random_split(ds,[train,val],generator=g)

class MultiHeadModel(nn.Module):
    """Backbone compartilhado + 2 cabeças (fine + coarse)."""
    def __init__(self, num_fine: int = 100, num_coarse: int = 20):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        # usamos somente o extrator de features + avgpool
        self.features = base.features
        self.avgpool  = base.avgpool
        self.flatten  = nn.Flatten(1)

        # Descobrir automaticamente a dimensionalidade após avgpool/flatten
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            f = self.features(dummy)
            f = self.avgpool(f)
            f = self.flatten(f)
            in_features = f.shape[1]  # tipicamente 576 para 32x32

        self.head_fine   = nn.Linear(in_features, num_fine)
        self.head_coarse = nn.Linear(in_features, num_coarse)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out_fine   = self.head_fine(x)
        out_coarse = self.head_coarse(x)
        return out_fine, out_coarse


def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_epoch(model, loader, opt, crit, device):
    model.train(); run=0.0
    for x,yf,yc in loader:
        x=x.to(device); yf=yf.to(device); yc=yc.to(device)
        opt.zero_grad()
        of, oc = model(x)
        loss = crit(of,yf) + crit(oc,yc)
        loss.backward(); opt.step()
        run += loss.item()*x.size(0)
    return run/len(loader.dataset)

def eval_epoch(model, loader, crit, device):
    model.eval(); run=0.0
    with torch.no_grad():
        for x,yf,yc in loader:
            x=x.to(device); yf=yf.to(device); yc=yc.to(device)
            of, oc = model(x)
            loss = crit(of,yf) + crit(oc,yc)
            run += loss.item()*x.size(0)
    return run/len(loader.dataset)

def predict(model, loader, device, head:str):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x,yf,yc in loader:
            x=x.to(device); of,oc = model(x)
            out = of if head=="fine" else oc
            y   = yf if head=="fine" else yc
            yp = out.argmax(1).cpu().tolist()
            yt.extend(y.cpu().tolist()); 
            # cuidado para não sobrescrever yp acumulado
            # vamos acumular corretamente:
    # Reprocessa para acumular corretamente:
    yt, yp = [], []
    with torch.no_grad():
        for x,yf,yc in loader:
            x=x.to(device); of,oc = model(x)
            out = of if head=="fine" else oc
            y   = yf if head=="fine" else yc
            pred = out.argmax(1).cpu().tolist()
            yt.extend(y.cpu().tolist()); yp.extend(pred)
    return yt, yp

def plot_losses(tr: List[float], vl: List[float], out_png: str, title: str):
    plt.figure(figsize=(9,5))
    plt.plot(tr, label="Loss Treino"); plt.plot(vl, label="Loss Validação")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir",  default="./outputs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ten-superclasses-subset", action="store_true",
                    help="Se ligar, coarse usa 10 superclasses (0..9). Fine continua 100.")
    args = ap.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    normalize = T.Normalize(mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761))
    tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32,padding=4), T.ToTensor(), normalize])
    tf_test  = T.Compose([T.ToTensor(), normalize])

    train_full = CIFAR100FineAndCoarse(root=args.data_dir, train=True,  transform=tf_train,
                                       ten_superclasses_subset=args.ten_superclasses_subset)
    test_ds    = CIFAR100FineAndCoarse(root=args.data_dir, train=False, transform=tf_test,
                                       ten_superclasses_subset=args.ten_superclasses_subset)
    train_ds, val_ds = make_splits(train_full, args.val_split, seed=42)
    print(f"Tamanhos: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_coarse = 10 if args.ten_superclasses_subset else 20
    model = MultiHeadModel(num_fine=100, num_coarse=num_coarse).to(device)
    n_params = count_trainable_params(model)
    print(f"Parâmetros treináveis: {n_params:,}")
    assert n_params <= 10_000_000, "Modelo excede 10M de parâmetros!"

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr)

    base = "modelo_multihead_10_SC_100_SC" + ("_coarse10" if args.ten_superclasses_subset else "")
    ckpt = os.path.join(args.out_dir, f"best_{base}.pth")
    png  = os.path.join(args.out_dir, f"loss_{base}.png")
    repf = os.path.join(args.out_dir, f"report_{base}.txt")

    best = float("inf"); best_w = copy.deepcopy(model.state_dict()); patience=0
    tr_hist, vl_hist = [], []

    print("\nIniciando treinamento (multihead)...")
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
    plot_losses(tr_hist, vl_hist, png, f"Curvas de Loss — {base}")
    print(f"Curvas de loss salvas em: {png}")

    print("Avaliando no conjunto de teste (multihead)...")
    yf_t, yf_p = predict(model, test_loader, device, head="fine")
    yc_t, yc_p = predict(model, test_loader, device, head="coarse")

    target_names_f = [f"class_{i}" for i in range(100)]
    target_names_c = [f"super_{i}" for i in range(num_coarse)]

    rep_f = classification_report(yf_t, yf_p, target_names=target_names_f, zero_division=0)
    rep_c = classification_report(yc_t, yc_p, target_names=target_names_c, zero_division=0)

    with open(repf, "w", encoding="utf-8") as f:
        f.write("=== Classification Report — 100 classes ===\n")
        f.write(rep_f + "\n\n")
        f.write(f"=== Classification Report — {num_coarse} superclasses ===\n")
        f.write(rep_c + "\n")

    print(rep_f); print(rep_c)
    print(f"Relatório salvo em: {repf}")
    print(f"Checkpoint salvo em: {ckpt}")
    print("Concluído.")

if __name__ == "__main__":
    main()
