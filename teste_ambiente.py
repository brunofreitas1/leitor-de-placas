import cv2
import pytesseract
import numpy as np

# NO WINDOWS: Se o Tesseract não estiver nas variáveis de ambiente, 
# descomente a linha abaixo e aponte para o caminho correto do seu executável:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("--- Testando Ambiente de Visão Computacional ---")
print(f"Versão do OpenCV: {cv2.__version__}")
print(f"Versão do Numpy: {np.__version__}")

try:
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"Versão do Tesseract OCR: {tesseract_version}")
    print("\n[OK] Ambiente configurado com sucesso!")
except Exception as e:
    print("\n[ERRO] Tesseract não foi encontrado ou não está configurado no PATH.")
    print(f"Detalhes do erro: {e}")