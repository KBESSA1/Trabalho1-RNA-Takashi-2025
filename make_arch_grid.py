#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def ensure_outdir(d="outputs"):
    os.makedirs(d, exist_ok=True); return d

def load_or_none(path):
    return mpimg.imread(path) if os.path.exists(path) else None

def draw_grid(imgs, titles, outpath):
    cols = len(imgs)
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, im, tt in zip(axes, imgs, titles):
        ax.axis("off")
        if im is not None:
            ax.imshow(im)
        ax.set_title(tt, fontsize=12, pad=6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[ok] Painel salvo em: {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="", help="Sufixo opcional no nome do arquivo final")
    # nomes padrão gerados pelo seu script
    ap.add_argument("--fine", default="outputs/arch_block_fine100.png")
    ap.add_argument("--coarse", default="outputs/arch_block_coarse20.png")
    ap.add_argument("--multi", default="outputs/arch_block_multihead.png")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    imgs = [
        load_or_none(args.fine),
        load_or_none(args.coarse),
        load_or_none(args.multi),
    ]
    titles = [
        "FINE — 100 classes (head 100)",
        "COARSE — 20 superclasses (head 20)",
        "MULTIHEAD — 100 + N superclasses",
    ]
    outname = "arch_panel"
    if args.tag:
        outname += f"_{args.tag}"
    outpath = os.path.join(args.outdir, f"{outname}.png")
    draw_grid(imgs, titles, outpath)

if __name__ == "__main__":
    main()
