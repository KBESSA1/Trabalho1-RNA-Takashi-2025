#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def ensure_outdir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def _box(ax, xy, w, h, text, fontsize=10, fc="white"):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        ec="black", fc=fc, lw=1.2
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.2))

def draw_block_diagram(mode: str, in_features: int, num_coarse: int, outdir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Entrada
    _box(ax, (0.5, 2.5), 1.8, 1.1, "Input\n(3×224×224)", fc="#eef5ff")
    # Backbone (features)
    _box(ax, (2.7, 2.5), 3.0, 1.1, "Backbone\n(MobileNetV3 Small)\nFeatures", fc="#f7f7f7")
    _arrow(ax, 2.3, 3.05, 2.7, 3.05)

    if mode in ("coarse20", "fine100"):
        out_dim = 20 if mode == "coarse20" else 100
        head_txt = f"Head Linear\n{in_features} → {out_dim}"
        _box(ax, (6.1, 2.5), 2.2, 1.1, head_txt, fc="#fff7e6")
        _arrow(ax, 5.7, 3.05, 6.1, 3.05)
        _box(ax, (8.6, 2.5), 0.9, 1.1, "Softmax", fc="#eefaf0")
        _arrow(ax, 8.3, 3.05, 8.6, 3.05)
        title = f"Arquitetura — {mode.upper()}"

    else:
        # multihead (100 + num_coarse)
        _box(ax, (6.0, 3.6), 2.6, 1.1, f"Head FINE\nLinear {in_features} → 100", fc="#fff7e6")
        _box(ax, (6.0, 1.3), 2.6, 1.1, f"Head COARSE\nLinear {in_features} → {num_coarse}", fc="#fff7e6")
        _arrow(ax, 5.7, 3.05, 6.0, 4.15)
        _arrow(ax, 5.7, 3.05, 6.0, 1.85)

        _box(ax, (8.8, 3.6), 0.9, 1.1, "Softmax\n(100)", fc="#eefaf0")
        _box(ax, (8.8, 1.3), 0.9, 1.1, f"Softmax\n({num_coarse})", fc="#eefaf0")
        _arrow(ax, 8.6, 4.15, 8.8, 4.15)
        _arrow(ax, 8.6, 1.85, 8.8, 1.85)
        title = f"Arquitetura — MULTIHEAD (100 + {num_coarse})"

    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    outpath = os.path.join(outdir, f"arch_block_{mode}.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[ok] Diagrama de blocos salvo em: {outpath}")

def parse_args():
    ap = argparse.ArgumentParser(description="Gera diagramas de arquitetura (blocos).")
    ap.add_argument("--mode", type=str, required=True,
                    choices=["coarse20", "fine100", "multihead"],
                    help="Arquitetura alvo.")
    ap.add_argument("--num-coarse", type=int, default=20,
                    help="# superclasses na head coarse (multihead). Use 10 para subset.")
    ap.add_argument("--in-features", type=int, default=1024,
                    help="Dimensão do vetor de features do backbone (MobileNetV3 Small ≈ 1024).")
    ap.add_argument("--outdir", type=str, default="outputs",
                    help="Diretório de saída.")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    draw_block_diagram(args.mode, args.in_features, args.num_coarse, outdir)
    print("[done] Figuras geradas.")

if __name__ == "__main__":
    main()
