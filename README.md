CIFAR-100: modelos para 20 superclasses, 100 classes e multihead (100 classes + 10 superclasses)

Este repositório reúne o código e os artefatos que desenvolvi para o trabalho da disciplina de Redes Neurais Artificiais (2025/2), sob orientação do professor Takashi. Trabalhei com o CIFAR-100, que possui 100 classes finas organizadas em 20 superclasses. Para atender ao enunciado, o modelo multihead foi adaptado para usar 10 superclasses. Esse ajuste foi feito por meio de um subconjunto coerente das superclasses oficiais, com remapeamento consistente dos rótulos e divisão limpa entre treino, validação e teste.

Arquitetura
Usei a MobileNetV3-Small da biblioteca torchvision como backbone para todas as variações. Essa arquitetura se mantém bem abaixo do limite de 10 milhões de parâmetros exigido no enunciado. Nos modelos single-head, apenas a última camada muda para 100 classes ou 20 superclasses. No multihead, após o flatten, há duas cabeças independentes: uma para 100 classes e outra para 10 superclasses. A função de perda total é a soma das duas entropias cruzadas.

Ambiente
As dependências estão descritas em requirements.txt. Utilizei Python 3.10. O treinamento em GPU é recomendado pela agilidade, mas o código também roda em CPU, com tempo maior de execução. Os arquivos VERSOES.txt e SHA256SUMS.txt registram, respectivamente, as versões das bibliotecas usadas e os hashes dos principais artefatos para reprodutibilidade.

Como executar
O dataset é baixado automaticamente via torchvision. O particionamento padrão é de 40.000 imagens para treino, 10.000 para validação e 10.000 para teste. No caso do multihead com 10 superclasses, o subconjunto reduz proporcionalmente e o script realiza a separação para aproximadamente 20.000 imagens de treino, 5.000 de validação e 5.000 de teste.

Treinamento de 100 classes
python train_cifar100.py --model-type fine100 --epochs 60 --patience 8 --batch-size 128

Treinamento de 20 superclasses
python train_cifar100.py --model-type coarse20 --epochs 60 --patience 8 --batch-size 128

Treinamento multihead (100 classes + 10 superclasses)
python modelo_multihead_10_SC_100_SC.py --ten-superclasses-subset --epochs 60 --patience 8 --batch-size 128

Detalhes do pipeline
Otimizador Adam, taxa de aprendizado inicial igual a 1e-3, scheduler de decaimento, batch size 128, early stopping com base na perda de validação e checkpoint automático na melhor época. Ao final, o script gera o classification report do scikit-learn no conjunto de teste. A preparação inclui normalização padrão do CIFAR-100 e aumento leve de dados com recorte aleatório e espelhamento horizontal.

Artefatos gerados
Curvas de perda: outputs/loss_fine100.png; outputs/loss_coarse20.png; outputs_final_pt/loss_multi_100+coarse10.png.
Relatórios de classificação: outputs/report_fine100.txt; outputs/report_coarse20.txt; outputs_final_pt/report_multi_100+coarse10.txt.
Pesos dos modelos: outputs/best_fine100.pth; outputs/best_coarse20.pth; outputs_final_pt/best_multi_100+coarse10.pth.
Matrizes de confusão: outputs_final_pt/cm_fine100.png; outputs_final_pt/cm_coarse20.png; outputs_final_pt/cm_multi_fine100.png; outputs_final_pt/cm_multi_coarse10.png.
Pacotes de conveniência: outputs_final_pt/PESOS_v1.1.tgz (pesos); outputs_final_pt/FIGURAS_v1.1.tgz (figuras).

Como regerar as matrizes a partir dos pesos
100 classes
python make_confusions_pt.py --mode fine100 --ckpt outputs/best_fine100.pth

20 superclasses
python make_confusions_pt.py --mode coarse20 --ckpt outputs/best_coarse20.pth

Multihead (100 + 10)
python make_confusions_pt.py --mode multihead --num-coarse 10 --ckpt outputs_final_pt/best_multi_100+coarse10.pth

Observação sobre o subset de 10 superclasses
O CIFAR-100 possui hierarquia oficial de 20 superclasses. Para cumprir o pedido de 10 superclasses no multihead sem criar rótulos inexistentes, utilizei um subconjunto coerente de 10 superclasses, com remapeamento consistente no carregamento. O código documenta o filtro aplicado e mantém a rastreabilidade do processo. Caso seja necessário, posso explicitar a lista exata de superclasses mantidas.

Conformidade com o enunciado
Modelo para 20 superclasses.
Modelo para 100 classes.
Modelo multihead com duas cabeças, uma para 100 classes e outra para 10 superclasses, com perda total igual à soma das duas entropias cruzadas.
Arquitetura do torchvision sob o limite de 10 milhões de parâmetros.
Divisão em treino, validação e teste.
Curvas de perda por época para treino e validação.
Checkpoint na melhor época com base na perda de validação.
Early stopping.
Relatório de classificação no conjunto de teste.

Reprodutibilidade e organização
As versões de bibliotecas estão registradas em VERSOES.txt. Os hashes dos artefatos principais constam em SHA256SUMS.txt. Os scripts de treino e de geração de matrizes estão na raiz do projeto. A tag v1.2 marca a entrega final.

Autor
Rodrigo Luiz Campos (Rodrigo KBESSA)
