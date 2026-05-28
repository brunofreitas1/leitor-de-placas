import cv2

def aplicar_filtros_e_localizar(caminho_imagem):
    # 1. Carregar a imagem original
    img = cv2.imread(caminho_imagem)
    
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{caminho_imagem}'.")
        return None, None 

    # Redimensionar a imagem
    img = cv2.resize(img, (800, 600))
    altura_img, largura_img = img.shape[:2] # Adicione esta linha

    # 2. Escala de Cinza (Grayscale)
    # Remove as cores para facilitar o processamento
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Filtro Bilateral
    # Reduz o ruído (como texturas de telhado) mantendo as bordas principais afiadas. 
    img_desfoque = cv2.bilateralFilter(img_cinza, 11, 17, 17)

    # 4. Detecção de Bordas (Método de Canny)
    img_bordas = cv2.Canny(img_desfoque, 50, 150)

    # 4.5 Dilatação (FECHAMENTO DE BURACOS)
    # Engrossa as linhas brancas para conectar partes "quebradas" do retângulo
    elemento_estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bordas = cv2.dilate(img_bordas, elemento_estruturante, iterations=1)

    # 5. Encontrar e Filtrar Contornos
    contornos, _ = cv2.findContours(img_bordas.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # AUMENTO DA BUSCA: Pegando os 30 maiores contornos em vez de apenas 10
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:30]
    
    contorno_placa = None
    coordenadas_retangulo = None 
    
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        
        proporcao = w / float(h)
        area = w * h
        limite_y = altura_img * 0.3 
        
        # O pulo do gato: adicionamos "area < 30000" para ignorar objetos gigantes como o tapete
        if 2.0 <= proporcao <= 5.0 and 1000 < area < 30000 and y > limite_y:
            contorno_placa = c
            coordenadas_retangulo = (x, y, w, h)
            break 

    # 6. Validação e Feedback Visual
    if coordenadas_retangulo is not None:
        x, y, w, h = coordenadas_retangulo
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print("[SUCESSO] Placa localizada e destacada em verde!")
    else:
        print("[AVISO] Não foi possível encontrar a placa nesta imagem.")

    # --- Exibindo os resultados para visualização ---
    cv2.imshow("1. Imagem Original com Deteccao", img)
    cv2.imshow("4. Bordas Limpas (Canny)", img_bordas)

    # Aguarda o usuário apertar qualquer tecla para fechar as janelas
    print("Pressione qualquer tecla nas janelas de imagem para fechá-las...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Agora a função retorna a imagem marcada e as coordenadas exatas da placa
    return img, contorno_placa

# --- Testando o código ---
# Certifique-se de que 'carro.jpg' está na mesma pasta do script
imagem_processada, coordenadas_placa = aplicar_filtros_e_localizar('carro.jpg')