import json
import os
import time
import easyocr
from pre_processamento import processar_imagem, REGEX_PLACA
from acesso import carregar_autorizados


CONFIG_PATH = 'testes.json'
IMAGES_DIR = 'test_images'
RESULTS_DIR = 'tests'


def carregar_config(caminho):
    if not os.path.exists(caminho):
        print(f"[ERRO] Arquivo de config nao encontrado: {caminho}")
        print(f"       Crie o arquivo com a lista de imagens e placas esperadas.")
        return None
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def comparar(ocr, esperado):
    return ocr == esperado


def executar_testes():
    inicio = time.time()

    print("=" * 60)
    print("TESTES EM LOTE - LEITOR DE PLACAS")
    print("=" * 60)

    config = carregar_config(CONFIG_PATH)
    if not config:
        return

    imagens_cfg = config.get('imagens', [])
    if not imagens_cfg:
        print(f"[ERRO] Nenhuma imagem listada em {CONFIG_PATH}.")
        return

    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERRO] Pasta {IMAGES_DIR}/ nao encontrada.")
        print(f"       Crie a pasta e coloque as imagens de teste nela.")
        return

    print(f"\n[0/4] Carregando EasyOCR e banco de moradores...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    autorizados = carregar_autorizados('moradores.json')
    print(f"      {len(autorizados)} moradores cadastrados.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    resultados = []
    total = len(imagens_cfg)
    corretos_placa = 0
    corretos_status = 0
    detectados = 0

    for i, item in enumerate(imagens_cfg, 1):
        arquivo = item['arquivo']
        placa_esperada = item['placa_esperada'].upper()
        deve_liberar = item.get('deve_liberar', True)

        caminho = os.path.join(IMAGES_DIR, arquivo)
        if not os.path.isfile(caminho):
            print(f"\n[{i}/{total}] [PULANDO] {arquivo} nao encontrado em {IMAGES_DIR}/")
            resultados.append({
                'arquivo': arquivo,
                'placa_ocr': None,
                'placa_esperada': placa_esperada,
                'status_ocr': 'ARQUIVO_AUSENTE',
                'status_esperado': 'LIBERADO' if deve_liberar else 'BLOQUEADO',
                'acertou_placa': False,
                'acertou_status': False,
            })
            continue

        print(f"\n[{i}/{total}] Processando {arquivo}...")
        t0 = time.time()
        resultado = processar_imagem(caminho, reader, autorizados, output_dir=RESULTS_DIR, verbose=False)
        duracao = time.time() - t0

        placa_ocr = resultado['placa']
        status_ocr = resultado['status']
        status_esperado = 'LIBERADO' if deve_liberar else 'BLOQUEADO'

        acertou_placa = comparar(placa_ocr, placa_esperada)
        acertou_status = comparar(status_ocr, status_esperado)

        if placa_ocr:
            detectados += 1
        if acertou_placa:
            corretos_placa += 1
        if acertou_status:
            corretos_status += 1

        marca_placa = '[OK]' if acertou_placa else '[X]'
        marca_status = '[OK]' if acertou_status else '[X]'
        print(f"  Placa: OCR={placa_ocr!r} Esperada={placa_esperada!r} {marca_placa}")
        print(f"  Status: OCR={status_ocr!r} Esperado={status_esperado!r} {marca_status}")
        print(f"  Tempo: {duracao:.1f}s")

        resultados.append({
            'arquivo': arquivo,
            'placa_ocr': placa_ocr,
            'placa_esperada': placa_esperada,
            'status_ocr': status_ocr,
            'status_esperado': status_esperado,
            'acertou_placa': acertou_placa,
            'acertou_status': acertou_status,
        })

    processados = sum(1 for r in resultados if r['status_ocr'] != 'ARQUIVO_AUSENTE')

    print("\n" + "=" * 60)
    print("RELATORIO FINAL")
    print("=" * 60)
    print(f"Imagens processadas:    {processados}/{total}")
    print(f"Placas detectadas:      {detectados}/{processados}  ({100 * detectados / max(processados, 1):.1f}%)")
    print(f"Placas corretas:        {corretos_placa}/{processados}  ({100 * corretos_placa / max(processados, 1):.1f}%)")
    print(f"Status corretos:        {corretos_status}/{processados}  ({100 * corretos_status / max(processados, 1):.1f}%)")
    print(f"Tempo total:            {time.time() - inicio:.1f}s")

    relatorio_path = os.path.join(RESULTS_DIR, 'relatorio_testes.csv')
    with open(relatorio_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Arquivo', 'Placa_OCR', 'Placa_Esperada', 'Status_OCR', 'Status_Esperado', 'Acertou_Placa', 'Acertou_Status'])
        for r in resultados:
            writer.writerow([
                r['arquivo'], r['placa_ocr'] or '', r['placa_esperada'],
                r['status_ocr'], r['status_esperado'],
                'SIM' if r['acertou_placa'] else 'NAO',
                'SIM' if r['acertou_status'] else 'NAO',
            ])
    print(f"\nRelatorio detalhado salvo em: {relatorio_path}")


if __name__ == "__main__":
    executar_testes()
