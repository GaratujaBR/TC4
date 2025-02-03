import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import numpy as np
from tqdm import tqdm
import os

class ActionRecognizer:
    def __init__(self):
        # Carregar modelo e processador
        self.model_name = "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
        print(f"Carregando modelo: {self.model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando device: {self.device}")
        
        # Carregar modelo e processador
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        # Mover modelo para GPU se disponível
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Buffer para suavização temporal
        self.prediction_buffer = []
        self.buffer_size = 5  # Número de frames para suavização
        
    def process_frame(self, frame):
        """Processa um único frame."""
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preparar imagem para o modelo
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Fazer predição
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Pegar a classe com maior probabilidade
        confidence, prediction = torch.max(probs, 1)
        
        return (
            self.model.config.id2label[prediction.item()],
            confidence.item()
        )
        
    def smooth_predictions(self, prediction, confidence):
        """Aplica suavização temporal às predições."""
        self.prediction_buffer.append((prediction, confidence))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
            
        # Contar ocorrências de cada predição
        prediction_counts = {}
        total_confidence = {}
        
        for pred, conf in self.prediction_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                total_confidence[pred] = 0
            prediction_counts[pred] += 1
            total_confidence[pred] += conf
            
        # Encontrar a predição mais frequente
        max_count = 0
        smoothed_prediction = prediction
        smoothed_confidence = confidence
        
        for pred, count in prediction_counts.items():
            if count > max_count:
                max_count = count
                smoothed_prediction = pred
                smoothed_confidence = total_confidence[pred] / count
                
        return smoothed_prediction, smoothed_confidence
        
    def process_video(self, input_path, output_path, confidence_threshold=0.5):
        """Processa o vídeo completo."""
        print(f"Processando vídeo: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {input_path}")
            return
            
        # Configurar vídeo de saída
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Dimensões do vídeo: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total de frames: {total_frames}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_action = "Unknown"
        current_confidence = 0.0
        
        with tqdm(total=total_frames, desc="Processando frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Processar frame
                action, confidence = self.process_frame(frame)
                
                # Aplicar suavização temporal
                smoothed_action, smoothed_confidence = self.smooth_predictions(
                    action, confidence)
                
                # Atualizar ação atual se confiança for alta o suficiente
                if smoothed_confidence > confidence_threshold:
                    current_action = smoothed_action
                    current_confidence = smoothed_confidence
                
                # Desenhar resultados no frame
                self.draw_predictions(frame, current_action, current_confidence)
                
                out.write(frame)
                pbar.update(1)
        
        cap.release()
        out.release()
        print(f"Processamento concluído: {output_path}")
        
    def draw_predictions(self, frame, action, confidence):
        """Desenha as predições no frame."""
        # Configurações de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        padding = 10
        
        # Texto principal
        text = f"Action: {action}"
        conf_text = f"Confidence: {confidence:.2f}"
        
        # Cor baseada na confiança
        color = (
            int(255 * (1 - confidence)),  # B
            int(255 * confidence),        # G
            0                            # R
        )
        
        # Posição do texto
        y_pos = 30
        cv2.putText(frame, text, (padding, y_pos),
                   font, font_scale, color, thickness)
        
        y_pos += 40
        cv2.putText(frame, conf_text, (padding, y_pos),
                   font, font_scale, color, thickness)

def main():
    # Configurar caminhos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'video.mp4')
    output_path = os.path.join(script_dir, 'output_vit_actions.mp4')
    
    # Verificar arquivo de entrada
    if not os.path.exists(input_path):
        print(f"Erro: Vídeo não encontrado em {input_path}")
        return
        
    print(f"Arquivo de vídeo encontrado: {input_path}")
    
    # Criar e usar detector
    recognizer = ActionRecognizer()
    recognizer.process_video(input_path, output_path, confidence_threshold=0.5)

if __name__ == "__main__":
    main()