import cv2
import numpy as np

# Carregue as imagens da placa sem defeito e com defeito
placa_sem_defeito = cv2.imread('pcbCroppedTranslated.png')
placa_com_defeito = cv2.imread('pcbCroppedTranslatedDefected.png')

# Verifique se as imagens foram carregadas corretamente
if placa_sem_defeito is None or placa_com_defeito is None:
    print("Erro ao carregar as imagens.")
    exit()

# Converta as imagens para tons de cinza
placa_sem_defeito_gray = cv2.cvtColor(placa_sem_defeito, cv2.COLOR_BGR2GRAY)
placa_com_defeito_gray = cv2.cvtColor(placa_com_defeito, cv2.COLOR_BGR2GRAY)

# Realize a subtração entre as duas imagens
diferenca = cv2.absdiff(placa_sem_defeito_gray, placa_com_defeito_gray)

# Aplique um limiar para destacar as diferenças (defeitos)
limiar = 30  # Ajuste conforme necessário
_, diferenca_binaria = cv2.threshold(diferenca, limiar, 255, cv2.THRESH_BINARY)

# Encontre os contornos das áreas destacadas
contornos, _ = cv2.findContours(diferenca_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhe os contornos na imagem original da placa com defeito
placa_com_defeito_com_contornos = placa_com_defeito.copy()
cv2.drawContours(placa_com_defeito_com_contornos, contornos, -1, (0, 0, 255), 2)

# Redimensione as imagens e janelas
scale_percent = 40  # Porcentagem de redimensionamento
width = int(placa_sem_defeito.shape[1] * scale_percent / 100)
height = int(placa_sem_defeito.shape[0] * scale_percent / 100)
dim = (width, height)

placa_sem_defeito_resized = cv2.resize(placa_sem_defeito, dim, interpolation=cv2.INTER_AREA)
placa_com_defeito_resized = cv2.resize(placa_com_defeito_com_contornos, dim, interpolation=cv2.INTER_AREA)
diferenca_binaria_resized = cv2.resize(diferenca_binaria, dim, interpolation=cv2.INTER_AREA)

# Exiba as imagens redimensionadas
cv2.imshow('Placa Sem Defeito', placa_sem_defeito_resized)
cv2.imshow('Placa Com Defeito', placa_com_defeito_resized)
cv2.imshow('Diferenca', diferenca_binaria_resized)

# Aguarde até que uma tecla seja pressionada e feche as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
