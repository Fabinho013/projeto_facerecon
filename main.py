import cv2
import face_recognition

# Acessar a webcam (0 é o índice padrão para a câmera principal)
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame da webcam
    ret, frame = video_capture.read()

    # Reduzir o tamanho do frame para processar mais rápido (opcional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converter o frame de BGR (usado pelo OpenCV) para RGB (usado pelo face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detectar os rostos no frame atual
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Percorrer todas as localizações de rostos encontradas
    for top, right, bottom, left in face_locations:
        # Redimensionar as coordenadas para o tamanho original do frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Exibir o frame com o rosto detectado
    cv2.imshow('Video - Reconhecimento Facial', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
