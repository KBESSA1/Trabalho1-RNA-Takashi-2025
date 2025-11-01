CIFAR-100 — coarse20, fine100 e multihead (100 + 10)

Resumo
Projeto da disciplina usando MobileNetV3 Small (torchvision, menos de 10 milhões de parâmetros).
Treinei três variações: coarse20 (20 superclasses), fine100 (100 classes) e multihead (duas cabeças após o flatten: 100 classes e 10 superclasses, com duas cross-entropies somadas).

Como rodar (exemplos):
python train_cifar100.py --model-type fine100 --epochs 60 --patience 8 --batch-size 128
python train_cifar100.py --model-type coarse20 --epochs 60 --patience 8 --batch-size 128
python train_cifar100.py --model-type multihead --ten-superclasses-subset --epochs 60 --patience 8 --batch-size 128

Saídas:
outputs/loss_fine100.png
outputs/report_fine100.txt
outputs/best_fine100.pth
outputs/loss_coarse20.png
outputs/report_coarse20.txt
outputs/best_coarse20.pth
outputs/loss_multi_100+coarse10.png
outputs/report_multi_100+coarse10.txt
outputs/best_multi_100+coarse10.pth
outputs_final_pt/cm_fine100.png
outputs_final_pt/cm_coarse20.png
outputs_final_pt/cm_multi_fine100.png
outputs_final_pt/cm_multi_coarse10.png

O QUE FOI PEDIDO:
treinar um modelo com 20 superclasses
treinar um modelo com 100 classes
treinar um modelo multihead (10 superclasses e 100 classes)
usar arquitetura do torchvision ou timm com menos de 10 milhões de parâmetros
dividir o dataset em treino, validação e teste
mostrar curvas de loss de treino e validação por época
salvar checkpoint a cada melhor época pela loss de validação
usar early stopping
apresentar classification report do sklearn ao final

Arquiteturas
MobileNetV3 Small como backbone.
Single-head: uma camada final para 100 ou para 20.
Multihead: duas MLPs após o flatten, uma para 100 e outra para 10 superclasses. As losses são somadas.

VISUALIZAÇÃO EASY
python -m http.server 8000
abrir http://localhost:8000/outputs_final_pt/
ver as figuras e relatórios diretamente pelo navegador

Autor
Rodrigo KBESSA
