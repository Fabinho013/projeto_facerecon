import cv2
import mediapipe as mp

# Inicializar os módulos de detecção de face e desenho
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo (0 para webcam padrão)
cap = cv2.VideoCapture(0)

# Configurar o modelo de detecção de rosto
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o quadro. Encerrando...")
            break

        # Converter a imagem de BGR para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar a detecção de rosto
        results = face_detection.process(image_rgb)

        # Se rostos forem detectados
        if results.detections:
            for detection in results.detections:
                # Desenhar as anotações da face detectada no frame
                mp_drawing.draw_detection(frame, detection)

        # Exibir o frame com a detecção
        cv2.imshow('Detecção Facial', frame)

        # Pressionar 'q' para sair do loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Liberar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()