import cv2
import re
import os
import time
import easyocr
import numpy as np
from datetime import datetime
from banco import listar_moradores, buscar_morador_por_placa, registrar_acesso

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


def encontrar_candidatos_haar(img, cascade_path=None, min_size=(40, 15)):
    cascatas = [
        cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml',
        cv2.data.haarcascades + 'haarcascade_license_plate_rus_16stages.xml',
    ]

    if cascade_path:
        cascatas = [cascade_path] + [c for c in cascatas if c != cascade_path]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    max_area = (w_img * h_img) * 0.5

    todos_candidatos = []
    for c_path in cascatas:
        cascade = cv2.CascadeClassifier(c_path)
        if cascade.empty():
            continue

        deteccoes = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=min_size,
        )

        for (x, y, w, h) in deteccoes:
            proporcao = w / float(h)
            area = w * h
            if 2.0 <= proporcao <= 5.5 and area >= 1500 and area <= max_area:
                todos_candidatos.append((x, y, w, h))

    unicos = []
    for c in todos_candidatos:
        x, y, w, h = c
        ja_existe = False
        for ux, uy, uw, uh in unicos:
            if abs(x - ux) < 20 and abs(y - uy) < 20 and abs(w - uw) < 20 and abs(h - uh) < 20:
                ja_existe = True
                break
        if not ja_existe:
            unicos.append(c)

    unicos.sort(key=lambda c: c[2] * c[3], reverse=True)
    return unicos


def encontrar_candidatos_sobel(img):
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
    return candidatos


def encontrar_candidatos(img):
    candidatos = encontrar_candidatos_haar(img)

    if not candidatos:
        candidatos = encontrar_candidatos_sobel(img)

    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return candidatos, img_cinza


def corrigir_ocr(texto):
    if not texto:
        return texto, False
    if len(texto) < 6:
        return texto, False
    if REGEX_PLACA.match(texto):
        return texto, False

    if len(texto) > 7:
        texto = texto[-7:]

    if REGEX_PLACA.match(texto):
        return texto, False

    corrigido = list(texto)
    n = len(corrigido)

    if n >= 7:
        alvo = corrigido[:7]
    else:
        alvo = corrigido + ['?'] * (7 - n)

    for i in [0, 1, 2]:
        if i < n:
            if alvo[i] in '0OQD':
                alvo[i] = 'O'
            elif alvo[i] in '1I':
                alvo[i] = 'I'
            elif alvo[i] in '2Z':
                alvo[i] = 'Z'
            elif alvo[i] in '5S':
                alvo[i] = 'S'
            elif alvo[i] in '6G':
                alvo[i] = 'G'

    if 3 < n and alvo[3].isalpha():
        if alvo[3] in 'Z':
            alvo[3] = '2'
        elif alvo[3] in 'OQD':
            alvo[3] = '0'
        elif alvo[3] in 'I':
            alvo[3] = '1'
        elif alvo[3] in 'B':
            alvo[3] = '8'
        elif alvo[3] in 'S':
            alvo[3] = '5'
        elif alvo[3] in 'G':
            alvo[3] = '6'

    for i in [5, 6]:
        if i < n:
            if alvo[i] in 'GODQC6':
                alvo[i] = '0'
            elif alvo[i] in 'I':
                alvo[i] = '1'
            elif alvo[i] in 'B':
                alvo[i] = '8'
            elif alvo[i] in 'Z':
                alvo[i] = '2'
            elif alvo[i] in 'S':
                alvo[i] = '5'

    if 4 < n:
        if alvo[4] in '0OQD':
            alvo[4] = 'O'
        elif alvo[4] in '1I':
            alvo[4] = 'I'
        elif alvo[4] in '2Z':
            alvo[4] = 'Z'
        elif alvo[4] in '5S':
            alvo[4] = 'S'
        elif alvo[4] in '6G':
            alvo[4] = 'G'

    inferido = False
    if n == 6:
        prefixo = ''.join(alvo[:6])
        for digito in '0123456789':
            tentativa = prefixo + digito
            if REGEX_PLACA.match(tentativa):
                return tentativa, True
        inferido = True

    resultado = ''.join(alvo[:7])
    if REGEX_PLACA.match(resultado):
        return resultado, inferido

    if len(resultado) == 7 and resultado[0] == 'D':
        resultado_tentativa = 'G' + resultado[1:]
        if REGEX_PLACA.match(resultado_tentativa):
            return resultado_tentativa, inferido

    return texto, False


def corrigir_por_levenshtein(texto, placas_autorizadas=None):
    texto = texto.upper().strip()
    n = len(texto)

    if not texto or n < 6:
        return texto, False

    if REGEX_PLACA.match(texto) and n == 7:
        return texto, False

    if placas_autorizadas:
        for placa in placas_autorizadas:
            if len(texto) != len(placa):
                continue
            diferencas = sum(1 for a, b in zip(texto, placa) if a != b)
            if diferencas == 1:
                return placa, True

    ALFANUM = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    if n == 7:
        for i in range(7):
            original = texto[i]
            for c in ALFANUM:
                if c == original:
                    continue
                tentativa = texto[:i] + c + texto[i+1:]
                if REGEX_PLACA.match(tentativa):
                    return tentativa, True

    if n == 6:
        for i in range(7):
            for c in ALFANUM:
                tentativa = texto[:i] + c + texto[i:]
                if REGEX_PLACA.match(tentativa):
                    return tentativa, True

    return texto, False


def gerar_variantes_binarias(img_borda):
    """Gera variantes de binarização a partir da imagem com borda."""
    gray = img_borda if len(img_borda.shape) == 2 else cv2.cvtColor(img_borda, cv2.COLOR_BGR2GRAY)
    variantes = []

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(('otsu', otsu))

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variantes.append(('otsu_inv', otsu_inv))

    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
    variantes.append(('adaptive', adapt))

    return variantes


def ocr_placa(img_color, x, y, w, h, reader, idx=0, is_closeup=False, verbose=False):
    margem = 15 if is_closeup else 5
    y1 = max(0, y - margem)
    y2 = min(img_color.shape[0], y + h + margem)
    x1 = max(0, x - margem)
    x2 = min(img_color.shape[1], x + w + margem)
    placa_crop = img_color[y1:y2, x1:x2]
    placa_gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)

    h_c, w_c = placa_gray.shape

    angulos = [0, 2, 4, 6, 7, 8] if is_closeup else [0, 5, 6, 7, 8, 9]

    candidatos_validos = []
    todos_textos = []
    melhor_img = None

    def processar_uma_variante(img_tentativa, nome, angulo):
        nonlocal melhor_img
        resultados = reader.readtext(img_tentativa, detail=1, paragraph=False)
        variante_ok = False
        for (bbox, texto, conf) in resultados:
            texto_limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())
            texto_corrigido, inferido = corrigir_ocr(texto_limpo)
            conf_ajustada = conf * 0.3 if inferido else conf
            print(f"    ang={angulo} {nome}: {texto_limpo!r} (conf={conf:.2f}) -> {texto_corrigido!r} {'[INFERIDO]' if inferido else ''}")
            todos_textos.append((texto_corrigido, conf_ajustada, img_borda))
            if REGEX_PLACA.match(texto_corrigido) and len(texto_corrigido) == 7:
                candidatos_validos.append((texto_corrigido, conf_ajustada, img_borda))
                if conf_ajustada >= 0.5:
                    variante_ok = True
        if variante_ok:
            melhor_img = img_borda
        return variante_ok

    def preparar_angulo(angulo):
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
        return cv2.copyMakeBorder(
            placa_ampliada, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255]
        )

    # Fase 1: CLAHE + invertida em todos os angulos
    # Rastreia por angulo se produziu resultado util (>= 6 chars)
    angulos_uteis = set()
    n_antes = 0
    img_bordas = {}
    for angulo in angulos:
        img_borda = preparar_angulo(angulo)
        img_bordas[angulo] = img_borda
        if processar_uma_variante(img_borda, 'clahe', angulo):
            continue
        if angulo == 7 and not is_closeup:
            cv2.imwrite(f'tests/debug_ocr_c{idx}.png', img_borda)
            img_invertida = cv2.bitwise_not(img_borda)
            cv2.imwrite(f'tests/debug_ocr_inv_c{idx}.png', img_invertida)
            processar_uma_variante(img_invertida, 'invertida', angulo)
        elif is_closeup and angulo in (4, 7):
            img_invertida = cv2.bitwise_not(img_borda)
            processar_uma_variante(img_invertida, 'invertida', angulo)
        n_depois = sum(1 for t, _, _ in todos_textos if len(t) >= 6)
        if n_depois > n_antes:
            angulos_uteis.add(angulo)
        n_antes = n_depois

    # Fase 2: binarizacao nos angulos uteis (mas para se ja achou placa com conf >= 0.5)
    for angulo in angulos_uteis:
        ja_tem_placa = any(c[1] >= 0.5 for c in candidatos_validos)
        if ja_tem_placa:
            if verbose:
                print(f"    ang={angulo}: pulando binarizacao (ja tem placa com conf >= 0.5)")
            break
        img_borda = img_bordas[angulo]
        for nome, img_bin in gerar_variantes_binarias(img_borda):
            if processar_uma_variante(img_bin, nome, angulo):
                break

    if candidatos_validos:
        votos = {}
        for texto, conf, img in candidatos_validos:
            if texto not in votos:
                votos[texto] = {'total': 0, 'melhor_conf': 0, 'melhor_img': None}
            votos[texto]['total'] += conf
            if conf > votos[texto]['melhor_conf']:
                votos[texto]['melhor_conf'] = conf
                votos[texto]['melhor_img'] = img
        ganhador = max(votos, key=lambda t: (votos[t]['total'], votos[t]['melhor_conf']))
        info = votos[ganhador]
        if verbose:
            for texto, v in sorted(votos.items(), key=lambda x: x[1]['total'], reverse=True):
                print(f"    [ENSEMBLE] {texto}: total={v['total']:.2f}, max={v['melhor_conf']:.2f}, votos={sum(1 for t,_,_ in candidatos_validos if t==texto)}")
            print(f"    [VENCEDOR] {ganhador} (total={info['total']:.2f})")
        return ganhador, info['total'], info['melhor_img']

    if todos_textos:
        todos_textos.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
        img_ret = melhor_img if melhor_img is not None else todos_textos[0][2]
        return todos_textos[0][0], todos_textos[0][1], img_ret

    return '', 0.0, melhor_img


def refinar_crop_por_contorno(img_gray, x, y, w, h, margem=20, min_area_ratio=0.5):
    h_img, w_img = img_gray.shape
    y1 = max(0, y - margem)
    y2 = min(h_img, y + h + margem)
    x1 = max(0, x - margem)
    x2 = min(w_img, x + w + margem)
    regiao = img_gray[y1:y2, x1:x2]
    if regiao.size == 0:
        return x, y, w, h
    blur = cv2.bilateralFilter(regiao, 9, 50, 50)
    bordas = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=2)
    contornos, _ = cv2.findContours(
        bordas_dilatadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return x, y, w, h
    area_original = w * h
    candidatos_ref = []
    for c in contornos:
        rx, ry, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        proporcao = rw / float(rh) if rh > 0 else 0
        if area >= area_original * min_area_ratio and 1.5 <= proporcao <= 6.0:
            candidatos_ref.append((rx, ry, rw, rh, area))
    if not candidatos_ref:
        return x, y, w, h
    candidatos_ref.sort(key=lambda c: c[4], reverse=True)
    rx, ry, rw, rh, _ = candidatos_ref[0]
    return max(0, x1 + rx), max(0, y1 + ry), rw, rh


def processar_imagem(caminho, reader, autorizados, output_dir='tests', verbose=False):
    img = preprocessar_imagem(caminho)
    h_img, w_img = img.shape[:2]

    candidatos, img_cinza = encontrar_candidatos(img)

    if not candidatos:
        ratio = w_img / float(h_img)
        if 2.0 <= ratio <= 5.5:
            if verbose:
                print(f"  [FALLBACK] Nenhum candidato, usando imagem inteira ({w_img}x{h_img}, ratio={ratio:.2f})")
            candidatos = [(0, 0, w_img, h_img)]

    candidatos_refinados = []
    for (cx, cy, cw, ch) in candidatos:
        if cx == 0 and cy == 0 and cw == w_img and ch == h_img:
            candidatos_refinados.append((cx, cy, cw, ch))
        else:
            candidatos_refinados.append(refinar_crop_por_contorno(img_cinza, cx, cy, cw, ch))
    candidatos = candidatos_refinados

    melhores = []
    for idx, (x, y, w, h) in enumerate(candidatos):
        is_closeup = (x == 0 and y == 0 and w == w_img and h == h_img)
        if verbose:
            print(f"  --- Candidato {idx}: ({x},{y},{w},{h}) closeup={is_closeup} ---")
        texto, conf, img_ocr = ocr_placa(img, x, y, w, h, reader, idx=idx, is_closeup=is_closeup, verbose=verbose)
        if verbose:
            print(f"    Resultado: {texto!r} (conf={conf:.2f})")
        if REGEX_PLACA.match(texto) and len(texto) == 7:
            melhores.append((texto, conf, img_ocr, (x, y, w, h)))
        else:
            placas_db = [m['placa'] for m in autorizados]
            texto_lv, foi_corrigido = corrigir_por_levenshtein(texto, placas_db)
            if foi_corrigido:
                conf_lv = conf * 0.5
                if verbose:
                    print(f"    [LEVENSHTEIN] {texto!r} -> {texto_lv!r} (conf_lv={conf_lv:.2f})")
                melhores.append((texto_lv, conf_lv, img_ocr, (x, y, w, h)))

    if not melhores:
        resultado = {
            'caminho': caminho,
            'placa': None,
            'coords': None,
            'liberado': False,
            'status': 'NAO_DETECTADA',
            'morador': None,
            'img': img,
        }
        os.makedirs(output_dir, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        cv2.imwrite(os.path.join(output_dir, f'resultado_{nome_base}.png'), img)
        resultado['img_path'] = os.path.join(output_dir, f'resultado_{nome_base}.png')
        return resultado

    melhores.sort(key=lambda m: m[1], reverse=True)
    placa_encontrada, conf, img_ocr, coords_placa = melhores[0]
    if verbose:
        print(f"  [ESCOLHIDO] {placa_encontrada!r} (conf={conf:.2f})")

    morador = buscar_morador_por_placa(placa_encontrada)
    liberado = morador is not None

    x, y, w, h = coords_placa
    cor = (0, 255, 0) if liberado else (0, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, 3)
    label = f"{placa_encontrada} - {'LIBERADO' if liberado else 'BLOQUEADO'}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    if liberado and morador:
        info = f"{morador['nome']} - Apto {morador['apartamento']}"
        cv2.putText(img, info, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    os.makedirs(output_dir, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho))[0]
    cv2.imwrite(os.path.join(output_dir, f'resultado_{nome_base}.png'), img)

    return {
        'caminho': caminho,
        'placa': placa_encontrada,
        'coords': coords_placa,
        'liberado': liberado,
        'status': 'LIBERADO' if liberado else 'BLOQUEADO',
        'morador': morador,
        'img': img,
        'img_path': os.path.join(output_dir, f'resultado_{nome_base}.png'),
    }


def main():
    caminho = 'carro.jpg'
    inicio = time.time()

    os.makedirs('tests', exist_ok=True)

    print("=" * 60)
    print("SISTEMA DE CONTROLE DE ACESSO - LEITOR DE PLACAS")
    print("=" * 60)

    print("\n[0/5] Carregando banco de moradores...")
    autorizados = listar_moradores()
    print(f"      {len(autorizados)} moradores cadastrados.")

    print("\n[1/5] Carregando modelo EasyOCR...")
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print(f"      Modelo carregado em {time.time() - t0:.1f}s")

    print(f"\n[2/5] Processando imagem: {caminho}")
    resultado = processar_imagem(caminho, reader, autorizados, output_dir='tests', verbose=True)

    print("\n[3/5] Verificando acesso...")
    if resultado['placa']:
        print(f"\n[OK] ACESSO {resultado['status']}: {resultado['placa']}")
        if resultado['morador']:
            print(f"     Morador: {resultado['morador']['nome']} - Apto {resultado['morador']['apartamento']}")
            print(f"     Veiculo: {resultado['morador'].get('veiculo', 'N/A')}")
        morador_id = resultado['morador']['id'] if resultado['morador'] else None
        registrar_acesso(
            placa=resultado['placa'],
            status=resultado['status'],
            morador_id=morador_id,
            imagem_path=resultado.get('img_path'),
        )
        print(f"     Log registrado no banco SQLite")
    else:
        print("\n[AVISO] Nenhuma placa valida encontrada para verificar acesso.")

    print(f"\nTempo total: {time.time() - inicio:.1f}s")
    return resultado['placa']


if __name__ == "__main__":
    main()
