#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho de Redes Neurais — CIFAR-100
Modelos:
  1) Classificação por 20 superclasses (coarse)
  2) Classificação por 100 classes (fine)
  3) Multihead: duas cabeças (100 classes + 20 superclasses)

Extra (opcional, para compatibilidade com enunciados que pedem 10 superclasses):
  • --ten-superclasses-subset: filtra o dataset para as 10 primeiras superclasses
    do enunciado e executa:
      - model_type=coarse20   → vira coarse10 (com 50 classes no total)
      - model_type=multihead  → vira 100 + 10 (cabeça coarse com 10)

Requisitos atendidos:
  • Split treino/val/test (val é holdout do treino)
  • Curvas de loss (treino e validação) salvas em PNG
  • Checkpoint a cada melhor época (pela loss de validação)
  • Early stopping
  • Classification report (sklearn) ao final no conjunto de teste

Backbone: torchvision.models.mobilenet_v3_small (≈ 2.5M parâmetros < 10M)

Como usar (exemplos):
  # 20 superclasses
  python train_cifar100.py --model-type coarse20

  # 100 classes
  python train_cifar100.py --model-type fine100

  # Multihead (100 + 20 superclasses)
  python train_cifar100.py --model-type multihead

  # Versão "10 superclasses" (subset das 10 primeiras superclasses do enunciado)
  python train_cifar100.py --model-type multihead --ten-superclasses-subset
  python train_cifar100.py --model-type coarse20 --ten-superclasses-subset

"""
import os
import copy
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset

import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ------------------------------
# Utilidades
# ------------------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class HParams:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 12
    val_split: float = 0.2
    num_workers: int = 2
    data_dir: str = "./data"
    out_dir: str = "./outputs"


# ------------------------------
# Dataset helpers (CIFAR-100)
# ------------------------------

TEN_COARSE_KEEP = set(list(range(0, 10)))  # manter 0..9 (as 10 primeiras do enunciado)


TEN_COARSE_KEEP = set(range(10))  # 0..9 (as 10 primeiras superclasses)


TEN_COARSE_KEEP = set(range(10))  # 0..9 (as 10 primeiras superclasses)

class CIFAR100DualLabels(Dataset):
    """
    Retorna (imagem, fine_label[0..99], coarse_label[0..19 ou 0..9]).
    Compatível com torchvision antigo (sem target_type): reconstrói coarse por nomes.
    Se ten_superclasses_subset=True, filtra coarse em {0..9} e reindexa para [0..9].
    """
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

        # nomes finos oficiais e índice
        if hasattr(self.base, "classes") and hasattr(self.base, "class_to_idx"):
            fine_names = list(self.base.classes)
            name_to_idx = dict(self.base.class_to_idx)
        else:
            raise RuntimeError("CIFAR100 sem 'classes'/'class_to_idx'.")

        # mapping canônico coarse → fines (nomes do torchvision)
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
            "large man-made outdoor things","large natural outdoor scenes",
            "large omnivores and herbivores","medium-sized mammals","non-insect invertebrates",
            "people","reptiles","small mammals","trees","vehicles 1","vehicles 2",
        ]
        coarse_index = {name: i for i, name in enumerate(coarse_order)}

        # aliases simples (se necessário)
        alias = {
            "orchids":"orchid","poppies":"poppy","roses":"rose","sunflowers":"sunflower","tulips":"tulip",
            "sweet peppers":"sweet_pepper","lawn-mower":"lawn_mower","pickup truck":"pickup_truck",
            "computer_keyboard":"keyboard",
            "maple":"maple_tree","oak":"oak_tree","palm":"palm_tree","pine":"pine_tree","willow":"willow_tree",
        }

        # fine_idx → coarse_id
        fine_to_coarse = [None] * len(fine_names)
        for cname in coarse_order:
            cid = coarse_index[cname]
            for fname in coarse_to_fines[cname]:
                if fname not in name_to_idx and fname in alias:
                    fname = alias[fname]
                idx = name_to_idx[fname]
                fine_to_coarse[idx] = cid
        if any(v is None for v in fine_to_coarse):
            missing = [i for i, v in enumerate(fine_to_coarse) if v is None]
            raise RuntimeError(f"Falha ao mapear rótulos finos para coarse: {missing}")

        # coarse por amostra
        self.ten_subset = ten_superclasses_subset
        coarse_all = [int(fine_to_coarse[f]) for f in fine_targets]

        if not self.ten_subset:
            self.indices = list(range(len(self.base)))
            self.coarse = coarse_all
        else:
            keep = TEN_COARSE_KEEP
            self.indices = [i for i, c in enumerate(coarse_all) if c in keep]
            remap = {orig: new for new, orig in enumerate(sorted(keep))}
            self.coarse = [remap[coarse_all[i]] for i in self.indices]

        self.fine = fine_targets  # sempre 0..99

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = T.functional.to_pil_image(self.base.data[real_idx])
        if self.transform:
            img = self.transform(img)
        y_fine = int(self.fine[real_idx])
        y_coarse = int(self.coarse[idx]) if self.ten_subset else int(self.coarse[real_idx])
        return img, y_fine, y_coarse


def make_splits(dataset: Dataset, val_split: float, seed: int = 42) -> Tuple[Subset, Subset]:
    n = len(dataset)
    val_size = int(val_split * n)
    train_size = n - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    return train_ds, val_ds


# ------------------------------
# Modelos
# ------------------------------



class CoarseHeadModel(nn.Module):
    """Classificação nas superclasses (20 por padrão, 10 no modo subset)."""
    def __init__(self, num_coarse: int):
        super().__init__()
        base = mobilenet_v3_small(weights=None)  # sem pesos pré-treinados
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_coarse)
        self.net = base
    def forward(self, x):
        return self.net(x)



class FineHeadModel(nn.Module):
    """Classificação nas 100 classes finas."""
    def __init__(self, num_fine: int = 100):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_fine)
        self.net = base
    def forward(self, x):
        return self.net(x)


class MultiHeadModel(nn.Module):
    """Backbone compartilhado + 2 cabeças (fine + coarse)."""
    def __init__(self, num_fine: int = 100, num_coarse: int = 20):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.flatten  = nn.Flatten(1)

        # Inferir in_features de forma segura (BatchNorm reclama com batch=1 em modo train)
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 32, 32)  # batch=2 evita erro do BN
            f = self.features(dummy)
            f = self.avgpool(f)
            f = self.flatten(f)
            in_features = f.shape[1]
        if was_training:
            self.features.train()

        self.head_fine   = nn.Linear(in_features, num_fine)
        self.head_coarse = nn.Linear(in_features, num_coarse)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out_fine   = self.head_fine(x)
        out_coarse = self.head_coarse(x)
        return out_fine, out_coarse


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out_fine = self.head_fine(x)
        out_coarse = self.head_coarse(x)
        return out_fine, out_coarse


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------
# Treino / Validação / Teste
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, multihead: bool = False):
    model.train()
    running = 0.0
    for imgs, y_fine, y_coarse in loader:
        imgs = imgs.to(device)
        y_fine = y_fine.to(device)
        y_coarse = y_coarse.to(device)
        optimizer.zero_grad()
        if multihead:
            out_fine, out_coarse = model(imgs)
            loss = criterion(out_fine, y_fine) + criterion(out_coarse, y_coarse)
        else:
            out = model(imgs)
            loss = criterion(out, y_fine)  # pode ser sobrescrito p/ coarse
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device, multihead: bool = False):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for imgs, y_fine, y_coarse in loader:
            imgs = imgs.to(device)
            y_fine = y_fine.to(device)
            y_coarse = y_coarse.to(device)
            if multihead:
                out_fine, out_coarse = model(imgs)
                loss = criterion(out_fine, y_fine) + criterion(out_coarse, y_coarse)
            else:
                out = model(imgs)
                loss = criterion(out, y_fine)
            running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def predict_loader(model, loader, device, head: str = "fine"):
    """Retorna y_true, y_pred para a head especificada: 'fine' ou 'coarse'."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, y_fine, y_coarse in loader:
            imgs = imgs.to(device)
            if isinstance(model, MultiHeadModel):
                out_fine, out_coarse = model(imgs)
                if head == "coarse":
                    out = out_coarse
                    y = y_coarse
                else:
                    out = out_fine
                    y = y_fine
            elif isinstance(model, CoarseHeadModel):
                out = model(imgs)
                y = y_coarse
            else:  # FineHeadModel
                out = model(imgs)
                y = y_fine
            pred = out.argmax(dim=1).cpu().numpy().tolist()
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred)
    return y_true, y_pred


def plot_losses(train_losses: List[float], val_losses: List[float], out_png: str, title: str):
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Loss Treino")
    plt.plot(val_losses, label="Loss Validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["coarse20", "fine100", "multihead"], required=True,
                        help="Tipo de modelo: coarse20 (20 superclasses), fine100 (100 classes), multihead (100+20).")
    parser.add_argument("--batch-size", type=int, default=HParams.batch_size)
    parser.add_argument("--lr", type=float, default=HParams.lr)
    parser.add_argument("--epochs", type=int, default=HParams.epochs)
    parser.add_argument("--patience", type=int, default=HParams.patience)
    parser.add_argument("--val-split", type=float, default=HParams.val_split)
    parser.add_argument("--data-dir", type=str, default=HParams.data_dir)
    parser.add_argument("--out-dir", type=str, default=HParams.out_dir)
    parser.add_argument("--num-workers", type=int, default=HParams.num_workers)
    parser.add_argument("--ten-superclasses-subset", action="store_true",
                        help="Usa apenas as 10 primeiras superclasses (subset). Coarse vira 10 e multihead vira 100+10.")
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Transforms
    normalize = T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    tf_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        normalize,
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Datasets
    train_full = CIFAR100DualLabels(root=args.data_dir, train=True, transform=tf_train,
                                    ten_superclasses_subset=args.ten_superclasses_subset)
    test_ds = CIFAR100DualLabels(root=args.data_dir, train=False, transform=tf_test,
                                 ten_superclasses_subset=args.ten_superclasses_subset)

    train_ds, val_ds = make_splits(train_full, args.val_split, seed=42)

    print(f"Tamanhos: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Modelos
    if args.model_type == "coarse20":
        num_coarse = 10 if args.ten_superclasses_subset else 20
        model = CoarseHeadModel(num_coarse=num_coarse)
        criterion = nn.CrossEntropyLoss()
        head_name = f"coarse{num_coarse}"
        multihead = False
    elif args.model_type == "fine100":
        model = FineHeadModel(num_fine=100)
        criterion = nn.CrossEntropyLoss()
        head_name = "fine100"
        multihead = False
    else:  # multihead
        num_coarse = 10 if args.ten_superclasses_subset else 20
        model = MultiHeadModel(num_fine=100, num_coarse=num_coarse)
        criterion = nn.CrossEntropyLoss()
        head_name = f"multi_100+coarse{num_coarse}"
        multihead = True

    model = model.to(device)
    n_params = count_trainable_params(model)
    print(f"Parâmetros treináveis: {n_params:,}")
    assert n_params <= 10_000_000, "Modelo excede 10M de parâmetros!"

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.out_dir, f"best_{head_name}.pth")
    loss_png = os.path.join(args.out_dir, f"loss_{head_name}.png")
    report_txt = os.path.join(args.out_dir, f"report_{head_name}.txt")

    best_val = float("inf")
    best_w = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses, val_losses = [], []

    print("\nIniciando treinamento...")
    for epoch in range(1, args.epochs + 1):
        if not multihead:
            def _criterion_single(out, y_fine, y_coarse):
                return criterion(out, y_coarse) if "coarse" in head_name else criterion(out, y_fine)

            def train_single(model, loader, optimizer, device):
                model.train(); running=0.0
                for imgs, yf, yc in loader:
                    imgs=imgs.to(device); yf=yf.to(device); yc=yc.to(device)
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = _criterion_single(out, yf, yc)
                    loss.backward(); optimizer.step()
                    running += loss.item()*imgs.size(0)
                return running/len(loader.dataset)

            def eval_single(model, loader, device):
                model.eval(); running=0.0
                with torch.no_grad():
                    for imgs, yf, yc in loader:
                        imgs=imgs.to(device); yf=yf.to(device); yc=yc.to(device)
                        out = model(imgs)
                        loss = _criterion_single(out, yf, yc)
                        running += loss.item()*imgs.size(0)
                return running/len(loader.dataset)

            tr_loss = train_single(model, train_loader, optimizer, device)
            vl_loss = eval_single(model, val_loader, device)
        else:
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, multihead=True)
            vl_loss = eval_one_epoch(model, val_loader, criterion, device, multihead=True)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"Época {epoch:03d} | train={tr_loss:.4f} | val={vl_loss:.4f}")

        if vl_loss < best_val - 1e-6:
            best_val = vl_loss
            best_w = copy.deepcopy(model.state_dict())
            torch.save(best_w, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping ativado.")
                break

    model.load_state_dict(best_w)

    plot_losses(train_losses, val_losses, out_png=loss_png,
                title=f"Curvas de Loss — {head_name}")
    print(f"Curvas de loss salvas em: {loss_png}")

    print("Avaliando no conjunto de teste...")
    if isinstance(model, MultiHeadModel):
        y_true_f, y_pred_f = predict_loader(model, test_loader, device, head="fine")
        y_true_c, y_pred_c = predict_loader(model, test_loader, device, head="coarse")
        target_names_f = [f"class_{i}" for i in range(100)]
        target_names_c = [f"super_{i}" for i in range(10 if args.ten_superclasses_subset else 20)]
        rep_f = classification_report(y_true_f, y_pred_f, labels=sorted(set(y_true_f)), zero_division=0)
        rep_c = classification_report(y_true_c, y_pred_c, labels=sorted(set(y_true_c)), zero_division=0)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write("=== Classification Report — 100 classes ===\n")
            f.write(rep_f + "\n\n")
            f.write(f"=== Classification Report — {len(target_names_c)} superclasses ===\n")
            f.write(rep_c + "\n")
        print(rep_f)
        print(rep_c)
    else:
        y_true, y_pred = predict_loader(model, test_loader, device,
                                        head="coarse" if "coarse" in head_name else "fine")
        target_names = [f"super_{i}" for i in range(10 if args.ten_superclasses_subset else 20)] \
                        if "coarse" in head_name else [f"class_{i}" for i in range(100)]
        rep = classification_report(y_true, y_pred, labels=sorted(set(y_true)), zero_division=0)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(rep)
        print(rep)

    print(f"Relatórios salvos em: {report_txt}")
    print(f"Checkpoint salvo em: {ckpt_path}")
    print("Concluído.")


if __name__ == "__main__":
    main()
