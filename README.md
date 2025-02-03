# Análise de Vídeo Inteligente

Framework para análise de vídeo com reconhecimento de ações humanas, detecção facial e análise de emoções.

## 📌 Funcionalidades Principais

- **Reconhecimento de Ações Humanas**  
  Identifica atividades em tempo real usando modelo Vision Transformer (ViT)
  - 600+ classes de ações humanas
  - Suavização temporal de predições
  - Exibição de confiança em tempo real

- **Reconhecimento Facial Avançado**
  - Detecção de faces com SSD MobileNet
  - Reconhecimento facial com OpenFace
  - Banco de faces conhecidas
  - Detecção de emoções com DeepFace

- **Análise de Emoções**  
  Identifica emoções dominantes frame a frame:
  - Alegria, Tristeza, Raiva, Neutro, etc
  - Bounding boxes com labels

## ⚙️ Pré-requisitos

- Python 3.8+
- Nvidia GPU (recomendado)
- 4GB+ RAM
- Espaço em disco: 2GB+

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/GaratujaBR/projeto_analise_video.git
cd projeto_analise_video
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Baixe os modelos necessários:
```python
python -c "from face_recognition import download_models; download_models()"
```

## 🎬 Como Usar

1. Coloque o vídeo de entrada na pasta raiz com o nome `video.mp4`

2. Executar análise de ações:
```bash
python detect_activities.py
```

3. Executar reconhecimento facial:
```bash
python face_recognition.py
```

4. Análise de emoções:
```bash
python detect_expression_video.py
```

**Saídas Geradas:**
- `output_vit_actions.mp4` - Vídeo com ações detectadas
- `output_video_enhanced.mp4` - Vídeo com reconhecimento facial
- `output_video_emotions.mp4` - Vídeo com análise de emoções

## 📂 Estrutura de Arquivos
```
projeto_analise_video/
├── detect_activities.py       # Reconhecimento de ações
├── face_recognition.py        # Reconhecimento facial
├── detect_expression_video.py # Análise de emoções
├── images/                    # Faces conhecidas para treino
├── models/                    # Modelos pré-treinados
└── video.mp4                  # Arquivo de vídeo de entrada
```

## ⚡ Configuração
- **Para melhor performance:**  
  Ajuste os parâmetros no código:
  ```python
  # Em detect_activities.py
  self.buffer_size = 5  # Tamanho do buffer de suavização
  confidence_threshold=0.5  # Limiar de confiança
  
  # Em face_recognition.py
  self.confidence_threshold = 0.7  # Limiar de detecção facial
  ```

## 💡 Dicas
- Use vídeos com resolução 720p para melhor performance
- Adicione fotos de rostos conhecidos na pasta `/images`
- Para CPU: reduza o tamanho do buffer nos códigos

## 🤝 Contribuição
Contribuições são bem-vindas! Siga estes passos:
1. Faça um fork do projeto
2. Crie sua branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença
MIT License - Consulte o arquivo LICENSE para mais detalhes
```
