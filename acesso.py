import json
import os
import csv
from datetime import datetime


def carregar_autorizados(caminho_arquivo='moradores.json'):
    if not os.path.exists(caminho_arquivo):
        print(f"[AVISO] Arquivo {caminho_arquivo} nao encontrado.")
        return []

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    return dados.get('moradores', [])


def verificar_acesso(placa, autorizados):
    placa = placa.upper().strip()
    for morador in autorizados:
        if morador['placa'].upper() == placa:
            return True, morador
    return False, None


def registrar_log_txt(placa, liberado, morador, arquivo='acessos.txt'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if liberado:
        info = f"{morador['nome']} (apto {morador['apartamento']}) - {morador.get('veiculo', 'N/A')}"
        status = "LIBERADO"
        linha = f"[{timestamp}] PLACA: {placa} | STATUS: {status} | MORADOR: {info}\n"
    else:
        status = "BLOQUEADO"
        linha = f"[{timestamp}] PLACA: {placa} | STATUS: {status} | MOTIVO: Placa nao cadastrada\n"

    with open(arquivo, 'a', encoding='utf-8') as f:
        f.write(linha)
    return linha.strip()


def registrar_log_csv(placa, liberado, morador, arquivo='acessos.csv'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    existe = os.path.isfile(arquivo)
    with open(arquivo, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['Data_Hora', 'Placa', 'Status', 'Nome', 'Apartamento', 'Veiculo'])
        if liberado:
            writer.writerow([
                timestamp, placa, 'LIBERADO',
                morador['nome'], morador['apartamento'],
                morador.get('veiculo', 'N/A')
            ])
        else:
            writer.writerow([timestamp, placa, 'BLOQUEADO', '-', '-', '-'])
