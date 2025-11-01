Trabalho – CIFAR-100 (coarse20, fine100 e multihead 100+10)

Resumo
- Treinei três variações usando MobileNetV3 Small (torchvision, <10M params):
  1) 20 superclasses (coarse20)
  2) 100 classes (fine100)
  3) Multihead: duas cabeças após o flatten (100 classes e 10 superclasses), loss = CE_fine + CE_coarse.
- Dataset dividido em treino/validação/teste (40k/10k/10k).
- Curvas de loss por época (train/val).
- Checkpoint automático no melhor val_loss + early stop.
- Classification report (sklearn) ao final.

Como rodar (exemplos)
# 100 classes
python train_cifar100.py --model-type fine100  --epochs 60 --patience 8 --batch-size 128

# 20 superclasses
python train_cifar100.py --model-type coarse20 --epochs 60 --patience 8 --batch-size 128

# Multihead (100 + 10 superclasses)
python modelo_multihead_10_SC_100_SC.py --epochs 60 --patience 8 --batch-size 128

Saídas principais (ver pasta outputs/ ou outputs_final_pt/)
- Curvas: loss_fine100.png, loss_coarse20.png, loss_multi_100+coarse10.png
- Relatórios: report_fine100.txt, report_coarse20.txt, report_multi_100+coarse10.txt
- Checkpoints: best_fine100.pth, best_coarse20.pth, best_multi_100+coarse10.pth
- Matrizes de confusão: cm_fine100.png, cm_coarse20.png, cm_multi_fine100.png, cm_multi_coarse10.png

Observações
- Multihead usa MobileNetV3 Small com duas MLPs paralelas no topo (fine=100, coarse=10).
- Otimizador: Adam, CE loss; scheduler opcional.
- Treino feito em CUDA quando disponível.

Autor
Rodrigo KBESSA
