import cv2

# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture('output.avi')  # Substitua 'seu_video.mp4' pelo caminho do seu vídeo

# Inicialize o objeto de subtração de fundo MOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplica a subtração de fundo no quadro atual
    fgmask = fgbg.apply(frame)

    # Realiza operações de limiarização e erosão/dilatação para remover ruídos
    fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Encontre os contornos dos objetos em movimento
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhe retângulos em torno dos objetos em movimento
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtre pequenos ruídos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exiba o vídeo com a detecção de movimento
    cv2.imshow('Detecção de Movimento', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Pressione Esc para sair
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()
