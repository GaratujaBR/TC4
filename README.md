# Análise de Vídeo com IA

Este projeto implementa um sistema de análise de vídeo que combina três funcionalidades principais usando Inteligência Artificial:
1. Reconhecimento Facial
2. Detecção de Emoções
3. Reconhecimento de Atividades

## Funcionalidades

### Reconhecimento Facial
- Detecta faces em cada frame do vídeo
- Identifica pessoas conhecidas comparando com imagens de referência
- Marca as faces detectadas com retângulos verdes
- Exibe o nome da pessoa identificada ou "Não Identificado"

### Detecção de Emoções
- Analisa a emoção predominante em cada face detectada
- Suporta as seguintes emoções:
  - Feliz
  - Triste
  - Irritado
  - Medo
  - Surpreso
  - Neutro
  - Nojo

### Reconhecimento de Atividades
- Identifica a atividade sendo realizada no vídeo
- Suporta diversas atividades como:
  - Dançando
  - Dormindo
  - Aplaudindo
  - Bebendo
  - Comendo
  - E muitas outras...
- Exibe a atividade detectada com nível de confiança

## Requisitos

### Dependências

- Python 3.x
- OpenCV
- DeepFace
- Transformers
- PyTorch
- Torch
- tqdm
- numpy
- tensorflow

### Modelos Pré-treinados
O sistema baixa automaticamente os seguintes modelos:
- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel
- openface.nn4.small2.v1.t7

## Estrutura do Projeto 

- projeto_analise_video/
  - face_emotion_activities.py
  - images/
  - output_combined.mp4
  - README.md


## Como Usar

1. **Preparação**
   - Coloque o vídeo a ser analisado como 'video.mp4' na pasta do projeto
   - Adicione fotos das pessoas a serem reconhecidas na pasta 'images'
   - Nomeie as fotos com o nome da pessoa (ex: "João.jpg")

2. **Execução**
   ```bash
   python face_emotion_activities.py
   ```

3. **Saída**
   - O sistema gerará um arquivo 'output_combined.mp4'
   - O vídeo de saída mostrará:
     - Faces detectadas com nomes
     - Emoções identificadas
     - Atividade atual com nível de confiança

## Configurações

O sistema possui alguns parâmetros configuráveis:
- `confidence_threshold`: 0.7 (limiar para detecção facial)
- `face_size`: 96 (tamanho padrão para processamento facial)
- `buffer_size`: 5 (frames para suavização temporal)

## Limitações

- Requer boa iluminação para melhor detecção facial
- O desempenho pode variar dependendo da qualidade do vídeo
- Necessita de recursos computacionais adequados para processamento em tempo real

## Tecnologias Utilizadas

- **PyTorch**: Para modelos de deep learning
- **OpenCV**: Para processamento de imagem e vídeo
- **DeepFace**: Para análise de emoções
- **Transformers**: Para reconhecimento de atividades
- **CUDA**: Suporte opcional para aceleração por GPU

## Contribuição

Sinta-se à vontade para contribuir com o projeto através de:
- Relatórios de bugs
- Sugestões de melhorias
- Pull requests

## Licença

Este projeto está sob a licença MIT.  
