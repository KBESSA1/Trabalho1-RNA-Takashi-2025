CIFAR-100 — coarse20, fine100 e multihead (100 + 10)

Trabalho da disciplina usando MobileNetV3 Small (torchvision, < 10M parâmetros).
Treinei três variações:
- coarse20: 20 superclasses
- fine100: 100 classes
- multihead: duas cabeças após o flatten (100 classes e 10 superclasses). A loss final soma as duas cross-entropies.

Como rodar (exemplos):
- 100 classes:        python train_cifar100.py --model-type fine100 --epochs 60 --patience 8 --batch-size 128
- 20 superclasses:    python train_cifar100.py --model-type coarse20 --epochs 60 --patience 8 --batch-size 128
- Multihead 100+10:   python train_cifar100.py --model-type multihead --ten-superclasses-subset --epochs 60 --patience 8 --batch-size 128

Saídas (organizadas na entrega):
- entrega_coarse20_pt/:     arch_block_coarse20.png, loss_coarse20.png, cm_coarse20.png, best_coarse20.pth, report_coarse20.txt
- entrega_fine100_pt/:      arch_block_fine100.png, loss_fine100.png, cm_fine100.png, best_fine100.pth, report_fine100.txt
- entrega_multi_100+10_pt/: arch_block_multihead_100+10.png, loss_multi_100+coarse10.png, cm_multi_fine100.png, cm_multi_coarse10.png, best_multi_100+coarse10.pth, report_multi_100+coarse10.txt

Atendido no enunciado:
- split treino/val/teste
- curvas de loss (treino/val)
- checkpoint no melhor val-loss
- early stopping
- classification_report final
- multihead 100+10 somando duas losses
- backbone < 10M parâmetros

Visualizar rápido (opcional):
- python -m http.server 8000
- abrir no navegador: /entrega_coarse20_pt/, /entrega_fine100_pt/, /entrega_multi_100+10_pt/

Autor: Rodrigo KBESSA
