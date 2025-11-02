CIFAR-100: modelos para 20 superclasses, 100 classes e multihead (100 classes + 10 superclasses)

Este repositório reúne o código e os artefatos que desenvolvi para o trabalho da disciplina de Redes Neurais Artificiais (2025/2), sob orientação do professor Takashi. Trabalhei com o CIFAR‑100, que tem 100 classes finas agrupadas em 20 superclasses. O multihead foi adaptado para usar 10 superclasses (um subset coerente das 20 oficiais), exatamente para atender ao enunciado sem inventar rótulos que não existem no dataset.

A arquitetura base é a MobileNetV3‑Small do torchvision, que fica confortavelmente abaixo de 10 milhões de parâmetros. Treinei três configurações:

20 superclasses

100 classes

multihead (100 classes + 10 superclasses), com duas cabeças independentes após o flatten; a loss final é a soma das duas cross‑entropies.

Ambiente

Instale as dependências:

pip install -r requirements.txt


Usei Python 3.10. Em GPU (CUDA) o treino vai bem mais rápido; em CPU também roda, só leva mais tempo.

Como rodar

Os comandos abaixo fazem o treinamento de cada variação. O CIFAR‑100 é baixado automaticamente pelo torchvision. O split interno fica assim:

full: 40.000 treino / 10.000 validação / 10.000 teste

subset 10 superclasses (multihead): 20.000 treino / 5.000 validação / 5.000 teste
(subset = 10 superclasses ⇒ 50 classes ⇒ 25.000 imagens de treino+val; o script separa 20k/5k)

100 classes (single‑head)

python train_cifar100.py --model-type fine100 --epochs 60 --patience 8 --batch-size 128


20 superclasses (single‑head)

python train_cifar100.py --model-type coarse20 --epochs 60 --patience 8 --batch-size 128


multihead (100 classes + 10 superclasses)

python modelo_multihead_10_SC_100_SC.py --ten-superclasses-subset --epochs 60 --patience 8 --batch-size 128


Como funciona o pipeline
Otimizador Adam com lr=1e-3 (há scheduler no código), batch size 128, early stopping guiado pela loss de validação e checkpoint automático na melhor época. Ao final, o script salva o classification_report (sklearn) no teste. A augmentação é leve (random crop + horizontal flip), com normalização padrão do CIFAR‑100.

Saídas

Curvas de perda

outputs/loss_fine100.png

outputs/loss_coarse20.png

outputs_final_pt/loss_multi_100+coarse10.png

Relatórios (classification_report)

outputs/report_fine100.txt

outputs/report_coarse20.txt

outputs_final_pt/report_multi_100+coarse10.txt

Checkpoints

outputs/best_fine100.pth

outputs/best_coarse20.pth

outputs_final_pt/best_multi_100+coarse10.pth

Matrizes de confusão

outputs_final_pt/cm_fine100.png

outputs_final_pt/cm_coarse20.png

outputs_final_pt/cm_multi_fine100.png

outputs_final_pt/cm_multi_coarse10.png

Pacotes prontos

outputs_final_pt/PESOS_v1.1.tgz (pesos dos três modelos)

outputs_final_pt/FIGURAS_v1.1.tgz (todas as figuras)

Se precisar regerar as matrizes a partir dos pesos:

# 100 classes
python make_confusions_pt.py --mode fine100  --ckpt outputs/best_fine100.pth

# 20 superclasses
python make_confusions_pt.py --mode coarse20 --ckpt outputs/best_coarse20.pth

# multihead (100 + 10)
python make_confusions_pt.py --mode multihead --num-coarse 10 --ckpt outputs_final_pt/best_multi_100+coarse10.pth

O que o enunciado pediu (e está coberto)

Modelo para 20 superclasses

Modelo para 100 classes

Multihead com duas cabeças (100 e 10) e loss = CE(100) + CE(10)

Arquitetura do torchvision/timm com ≤ 10M parâmetros

Split treino/validação/teste

Curvas de loss (treino/validação por época)

Checkpoint na melhor época (menor val_loss)

Early stopping

classification_report no teste

Arquiteturas

Mantive o MobileNetV3‑Small como backbone em tudo para uma comparação justa.
Nos single‑head, troco apenas a camada final (100 ou 20 saídas).
No multihead, após o flatten saem duas MLPs paralelas: uma para as 100 classes, outra para as 10 superclasses. Cada cabeça calcula sua cross‑entropy e a loss final é a soma. Na prática, a cabeça “grossa” (10 SC) age como regularizador supervisionado: organiza o espaço de features por família e facilita o refinamento fino da cabeça de 100.

Visualização “easy”

Se quiser abrir tudo no navegador:

python -m http.server 8000
# depois, no navegador:
# http://localhost:8000/outputs_final_pt/


Autor: Rodrigo Luiz Campos (Rodrigo KBESSA)
