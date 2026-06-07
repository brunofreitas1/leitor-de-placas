import cv2
import re
import csv
import os
import time
import easyocr
import numpy as np
from datetime import datetime
from acesso import carregar_autorizados, verificar_acesso, registrar_log_txt, registrar_log_csv

REGEX_PLACA = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$|^[A-Z]{3}[0-9]{4}$')


def preprocessar_imagem(caminho_imagem, largura_max=1000):
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {caminho_imagem}")

    altura, largura = img.shape[:2]
    if largura > largura_max:
        escala = largura_max / largura
        img = cv2.resize(
            img,
            (int(largura * escala), int(altura * escala)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def encontrar_candidatos(img):
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cinza = clahe.apply(img_cinza)

    img_filtrada = cv2.bilateralFilter(img_cinza, 11, 17, 17)
    sobelx = cv2.Sobel(img_filtrada, cv2.CV_8U, 1, 0, ksize=3)

    _, limiar = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    img_fechada = cv2.morphologyEx(limiar, cv2.MORPH_CLOSE, kernel)
    img_fechada = cv2.dilate(img_fechada, None, iterations=2)

    contornos, _ = cv2.findContours(
        img_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / float(h)
        area = w * h
        if 2.0 <= proporcao <= 5.5 and area > 2000:
            candidatos.append((x, y, w, h))

    candidatos.sort(key=lambda c: c[2] * c[3], reverse=True)
    return candidatos, img_cinza


def corrigir_ocr(texto):
    if not texto:
        return texto
    if len(texto) < 7:
        return texto
    if REGEX_PLACA.match(texto):
        return texto

    if len(texto) > 7:
        texto = texto[-7:]

    if REGEX_PLACA.match(texto):
        return texto

    corrigido = list(texto)
    if len(corrigido) == 7:
        for i in [5, 6]:
            if corrigido[i] in 'GODQC':
                corrigido[i] = '0'
            elif corrigido[i] == 'I':
                corrigido[i] = '1'
            elif corrigido[i] == 'B':
                corrigido[i] = '8'

        if corrigido[3].isalpha():
            if corrigido[3] == 'Z':
                corrigido[3] = '2'
            elif corrigido[3] == 'O':
                corrigido[3] = '0'
            elif corrigido[3] == 'I':
                corrigido[3] = '1'
            elif corrigido[3] == 'B':
                corrigido[3] = '8'

    resultado = ''.join(corrigido)
    if REGEX_PLACA.match(resultado):
        return resultado

    if len(resultado) == 7 and resultado[0] == 'D':
        resultado_tentativa = 'G' + resultado[1:]
        if REGEX_PLACA.match(resultado_tentativa):
            return resultado_tentativa

    return texto


def ocr_placa(img_color, x, y, w, h, reader, idx=0):
    placa_crop = img_color[y:y + h, x:x + w]
    placa_gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)

    h_c, w_c = placa_gray.shape

    candidatos_validos = []
    todos_textos = []

    for angulo in [5, 6, 7, 8, 9]:
        center = (w_c // 2, h_c // 2)
        M = cv2.getRotationMatrix2D(center, angulo, 1.0)
        placa_rot = cv2.warpAffine(
            placa_gray, M, (w_c, h_c),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        placa_clahe = clahe.apply(placa_rot)

        placa_enhanced = cv2.convertScaleAbs(placa_clahe, alpha=2.0, beta=0)

        placa_ampliada = cv2.resize(
            placa_enhanced, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
        )

        img_borda = cv2.copyMakeBorder(
            placa_ampliada, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255]
        )

        if angulo == 7:
            cv2.imwrite(f'tests/debug_ocr_c{idx}.png', img_borda)
            img_invertida = cv2.bitwise_not(img_borda)
            cv2.imwrite(f'tests/debug_ocr_inv_c{idx}.png', img_invertida)
            variacoes = [('normal', img_borda), ('invertida', img_invertida)]
        else:
            variacoes = [('normal', img_borda)]

        for nome, img_tentativa in variacoes:
            resultados = reader.readtext(img_tentativa, detail=1, paragraph=False)
            for (bbox, texto, conf) in resultados:
                texto_limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())
                texto_corrigido = corrigir_ocr(texto_limpo)
                print(f"    ang={angulo} {nome}: {texto_limpo!r} (conf={conf:.2f}) -> {texto_corrigido!r}")
                todos_textos.append((texto_corrigido, conf, img_borda))
                if REGEX_PLACA.match(texto_corrigido) and len(texto_corrigido) == 7:
                    candidatos_validos.append((texto_corrigido, conf, img_borda))

    if candidatos_validos:
        candidatos_validos.sort(key=lambda x: x[1], reverse=True)
        return candidatos_validos[0][0], candidatos_validos[0][2]

    if todos_textos:
        todos_textos.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
        return todos_textos[0][0], todos_textos[0][2]

    return '', img_borda


def registrar_placa(placa):
    arquivo = 'registro_placas.csv'
    existe = os.path.isfile(arquivo)
    with open(arquivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['Data_Hora', 'Placa'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), placa])
    print(f"[LOG] Placa {placa} registrada em {arquivo}")


def processar_imagem(caminho, reader, autorizados, output_dir='tests', verbose=False):
    img = preprocessar_imagem(caminho)
    candidatos, _ = encontrar_candidatos(img)

    placa_encontrada = None
    coords_placa = None
    for idx, (x, y, w, h) in enumerate(candidatos):
        if verbose:
            print(f"  --- Candidato {idx}: ({x},{y},{w},{h}) ---")
        texto, img_ocr = ocr_placa(img, x, y, w, h, reader, idx=idx)
        if verbose:
            print(f"    Melhor: {texto!r}")
        if REGEX_PLACA.match(texto) and len(texto) == 7:
            placa_encontrada = texto
            coords_placa = (x, y, w, h)
            break

    resultado = {
        'caminho': caminho,
        'placa': placa_encontrada,
        'coords': coords_placa,
        'liberado': False,
        'status': 'NAO_DETECTADA',
        'morador': None,
        'img': img,
    }

    if placa_encontrada:
        liberado, morador = verificar_acesso(placa_encontrada, autorizados)
        resultado['liberado'] = liberado
        resultado['morador'] = morador
        resultado['status'] = 'LIBERADO' if liberado else 'BLOQUEADO'

        x, y, w, h = coords_placa
        cor = (0, 255, 0) if liberado else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), cor, 3)
        label = f"{placa_encontrada} - {resultado['status']}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        if liberado and morador:
            info = f"{morador['nome']} - Apto {morador['apartamento']}"
            cv2.putText(img, info, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        os.makedirs(output_dir, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        cv2.imwrite(os.path.join(output_dir, f'resultado_{nome_base}.png'), img)
        resultado['img_path'] = os.path.join(output_dir, f'resultado_{nome_base}.png')

    return resultado


def main():
    caminho = 'carro.jpg'
    inicio = time.time()

    os.makedirs('tests', exist_ok=True)

    print("=" * 60)
    print("SISTEMA DE CONTROLE DE ACESSO - LEITOR DE PLACAS")
    print("=" * 60)

    print("\n[0/5] Carregando banco de moradores...")
    autorizados = carregar_autorizados('moradores.json')
    print(f"      {len(autorizados)} moradores cadastrados.")

    print("\n[1/5] Carregando modelo EasyOCR...")
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print(f"      Modelo carregado em {time.time() - t0:.1f}s")

    print(f"\n[2/5] Carregando imagem: {caminho}")
    img = preprocessar_imagem(caminho)

    print("[3/5] Detectando candidatos a placa...")
    candidatos, _ = encontrar_candidatos(img)
    print(f"      Encontrados {len(candidatos)} candidatos.")

    print("\n[4/5] Executando OCR nos candidatos...")
    placa_encontrada = None
    coords_placa = None
    for idx, (x, y, w, h) in enumerate(candidatos):
        print(f"\n--- Candidato {idx}: ({x},{y},{w},{h}) ---")
        texto, img_ocr = ocr_placa(img, x, y, w, h, reader, idx=idx)
        print(f"  Melhor resultado: {texto!r}")

        if REGEX_PLACA.match(texto) and len(texto) == 7:
            placa_encontrada = texto
            coords_placa = (x, y, w, h)
            break

    print("\n[5/5] Verificando acesso...")
    if placa_encontrada:
        x, y, w, h = coords_placa
        liberado, morador = verificar_acesso(placa_encontrada, autorizados)

        if liberado:
            cor = (0, 255, 0)
            status = "LIBERADO"
            info = f"{morador['nome']} - Apto {morador['apartamento']}"
            print(f"\n[OK] ACESSO {status}: {placa_encontrada}")
            print(f"     Morador: {info}")
            print(f"     Veiculo: {morador.get('veiculo', 'N/A')}")
        else:
            cor = (0, 0, 255)
            status = "BLOQUEADO"
            print(f"\n[BLOQUEADO] ACESSO {status}: {placa_encontrada}")
            print(f"     Motivo: Placa nao cadastrada no sistema.")

        cv2.rectangle(img, (x, y), (x + w, y + h), cor, 3)
        label = f"{placa_encontrada} - {status}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        if liberado:
            info_text = f"{morador['nome']} - Apto {morador['apartamento']}"
            cv2.putText(img, info_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        log_txt = registrar_log_txt(placa_encontrada, liberado, morador)
        registrar_log_csv(placa_encontrada, liberado, morador)
        print(f"     Log: {log_txt}")

        registrar_placa(placa_encontrada)
        cv2.imwrite('tests/debug_resultado.png', img)
    else:
        print("\n[AVISO] Nenhuma placa valida encontrada para verificar acesso.")
        cv2.imwrite('tests/debug_resultado.png', img)

    print(f"\nTempo total: {time.time() - inicio:.1f}s")
    return placa_encontrada


if __name__ == "__main__":
    main()
