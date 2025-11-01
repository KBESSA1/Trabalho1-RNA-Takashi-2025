#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---- CIFAR-100: mapeamento oficial (20 superclasses) ----
COARSE_GROUPS = [
    ("aquatic_mammals",        ["beaver","dolphin","otter","seal","whale"]),
    ("fish",                   ["aquarium_fish","flatfish","ray","shark","trout"]),
    ("flowers",                ["orchid","poppy","rose","sunflower","tulip"]),
    ("food_containers",        ["bottle","bowl","can","cup","plate"]),
    ("fruit_and_vegetables",   ["apple","mushroom","orange","pear","sweet_pepper"]),
    ("household_electrical",   ["clock","keyboard","lamp","telephone","television"]),
    ("household_furniture",    ["bed","chair","couch","table","wardrobe"]),
    ("insects",                ["bee","beetle","butterfly","caterpillar","cockroach"]),
    ("large_carnivores",       ["bear","leopard","lion","tiger","wolf"]),
    ("large_manmade_outdoor",  ["bridge","castle","house","road","skyscraper"]),
    ("large_natural_outdoor",  ["cloud","forest","mountain","plain","sea"]),
    ("large_omnivores_herb",   ["camel","cattle","chimpanzee","elephant","kangaroo"]),
    ("medium_mammals",         ["fox","porcupine","possum","raccoon","skunk"]),
    ("noninsect_invertebrates",["crab","lobster","snail","spider","worm"]),
    ("people",                 ["baby","boy","girl","man","woman"]),
    ("reptiles",               ["crocodile","dinosaur","lizard","snake","turtle"]),
    ("small_mammals",          ["hamster","mouse","rabbit","shrew","squirrel"]),
    ("trees",                  ["maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"]),
    ("vehicles_1",             ["bicycle","bus","motorcycle","pickup_truck","train"]),
    ("vehicles_2",             ["lawn_mower","rocket","streetcar","tank","tractor"]),
]

# --------- utilidades ----------
def fine_to_coarse_index(classes):
    name_to_idx = {n:i for i,n in enumerate(classes)}
    f2c = np.zeros(100, dtype=np.int64)
    for cid,(_,flist) in enumerate(COARSE_GROUPS):
        for fn in flist:
            f2c[name_to_idx[fn]] = cid
    return f2c

def get_testset_and_loader(batch=256):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071,0.4867,0.4408),
                             std =(0.2675,0.2565,0.2761)),
    ])
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm)
    loader  = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    classes = getattr(testset, "classes", None)
    if classes is None:  # fallback
        by_idx = sorted(testset.class_to_idx.items(), key=lambda kv: kv[1])
        classes = [k for k,_ in by_idx]
    y_true_f = np.array(testset.targets, dtype=np.int64)
    return classes, y_true_f, loader

def plot_cm(cm, labels, title, outpng, figsize=(11,10)):
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_xlabel("Predito", fontsize=11); ax.set_ylabel("Real", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout(); fig.savefig(outpng, dpi=160); plt.close(fig)

# --------- modelos ----------
class SingleHead(nn.Module):
    """MobileNetV3 Small com uma cabeça (usa 1024 features da classifier)."""
    def __init__(self, num_out):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        in_feats = base.classifier[-1].in_features  # 1024
        base.classifier[-1] = nn.Linear(in_feats, num_out)
        self.net = base
    def forward(self, x): return self.net(x)

class MultiHead576(nn.Module):
    """
    Backbone MobileNetV3 Small até avgpool (576 feats),
    classifier = Identity, e duas cabeças lineares.
    Nomes dos módulos (features/avgpool/classifier, head_fine, head_coarse)
    compatíveis com checkpoints que salvam keys como 'features.*' e 'head_*'.
    """
    def __init__(self, num_fine=100, num_coarse=10):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.classifier = nn.Identity()        # garante saída 576
        self.head_fine   = nn.Linear(576, num_fine)
        self.head_coarse = nn.Linear(576, num_coarse)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)            # 576
        return self.head_fine(feats), self.head_coarse(feats)

def predict_single(model, loader, device):
    model.eval(); yh=[]
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            yh.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(yh)

def predict_multi(model, loader, device):
    model.eval(); yh_f=[]; yh_c=[]
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            lf, lc = model(x)
            yh_f.append(lf.argmax(1).cpu().numpy())
            yh_c.append(lc.argmax(1).cpu().numpy())
    return np.concatenate(yh_f), np.concatenate(yh_c)

# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fine100","coarse20","multihead"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-coarse", type=int, default=10)
    ap.add_argument("--outdir", default="outputs_final_pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    classes, y_true_f, loader = get_testset_and_loader()
    f2c = fine_to_coarse_index(classes)
    coarse_names20 = [g for g,_ in COARSE_GROUPS]
    device = torch.device(args.device)

    if args.mode == "fine100":
        model = SingleHead(100).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        y_pred_f = predict_single(model, loader, device)
        cm_f = confusion_matrix(y_true_f, y_pred_f)
        plot_cm(cm_f, [str(i) for i in range(100)],
                "Matriz de Confusão — Classes (test)",
                os.path.join(args.outdir,"cm_fine100.png"))

    elif args.mode == "coarse20":
        model = SingleHead(20).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        y_true_c = f2c[y_true_f]
        y_pred_c = predict_single(model, loader, device)
        cm_c = confusion_matrix(y_true_c, y_pred_c)
        plot_cm(cm_c, coarse_names20,
                "Matriz de Confusão — Superclasses (test)",
                os.path.join(args.outdir,"cm_coarse20.png"),
                figsize=(12,10))

    else:  # multihead (100 + N)
        model = MultiHead576(num_fine=100, num_coarse=args.num_coarse).to(device)
        # checkpoints antigos: carregar com strict=False para tolerar variações inofensivas
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd, strict=False)

        y_pred_f, y_pred_c = predict_multi(model, loader, device)
        # Fine 100
        cm_f = confusion_matrix(y_true_f, y_pred_f)
        plot_cm(cm_f, [str(i) for i in range(100)],
                "Matriz de Confusão — Classes (test)",
                os.path.join(args.outdir,"cm_multi_fine100.png"))
        # Coarse head (N)
        y_true_c20 = f2c[y_true_f]
        if args.num_coarse == 20:
            labels_c = coarse_names20
            y_true_c_for_head = y_true_c20
        else:
            labels_c = [f"super{j}" for j in range(args.num_coarse)]
            y_true_c_for_head = y_true_c20 % args.num_coarse
        cm_c = confusion_matrix(y_true_c_for_head, y_pred_c)
        plot_cm(cm_c, labels_c,
                f"Matriz de Confusão — Superclasses (test) ({args.num_coarse})",
                os.path.join(args.outdir,f"cm_multi_coarse{args.num_coarse}.png"),
                figsize=(12,10))

    print("[ok] Matrizes geradas em", args.outdir)

if __name__ == "__main__":
    main()
