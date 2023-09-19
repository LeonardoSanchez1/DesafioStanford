import cv2
import numpy as np

# Carregar as imagens das placas
placa_sem_defeito = cv2.imread('pcbCroppedTranslated.png')
placa_com_defeito = cv2.imread('pcbCroppedTranslatedDefected.png')

# Verificar se as imagens foram carregadas com sucesso
if placa_sem_defeito is None or placa_com_defeito is None:
    print("Erro ao carregar as imagens.")
    exit()

# Converter as imagens para escala de cinza (grayscale)
placa_sem_defeito_gray = cv2.cvtColor(placa_sem_defeito, cv2.COLOR_BGR2GRAY)
placa_com_defeito_gray = cv2.cvtColor(placa_com_defeito, cv2.COLOR_BGR2GRAY)

# Calcular a diferença entre as duas imagens
diferenca = cv2.absdiff(placa_sem_defeito_gray, placa_com_defeito_gray)

# Threshold para destacar as áreas de diferença
limiar = 30  # Você pode ajustar esse valor de acordo com suas necessidades
_, mascara = cv2.threshold(diferenca, limiar, 255, cv2.THRESH_BINARY)

# Encontrar os contornos das áreas destacadas
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar os contornos nas imagens originais
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    cv2.rectangle(placa_com_defeito, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Redimensionar a janela para que caiba na tela
cv2.namedWindow('Placa com Defeito', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Placa com Defeito', 800, 600)

# Exibir as imagens com os contornos dos defeitos
cv2.imshow('Placa com Defeito', placa_com_defeito)
cv2.waitKey(0)
cv2.destroyAllWindows()
