import json
import os
from banco import adicionar_morador, limpar_banco, listar_moradores


def migrar_para_sqlite():
    caminho_json = os.path.join(os.path.dirname(__file__), 'moradores.json')

    if not os.path.exists(caminho_json):
        print("[init_db] moradores.json nao encontrado. Banco vazio criado.")
        return

    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    moradores = dados.get('moradores', [])
    if not moradores:
        print("[init_db] Nenhum morador encontrado no JSON.")
        return

    for m in moradores:
        sucesso, msg = adicionar_morador(
            nome=m.get('nome', 'Desconhecido'),
            placa=m.get('placa', ''),
            apartamento=m.get('apartamento'),
            veiculo=m.get('veiculo')
        )
        tag = "[OK]" if sucesso else "[IGNORADO]"
        print(f"{tag} {m.get('nome', '?')} ({m.get('placa', '?')}): {msg}")

    print(f"\n[init_db] Migracao concluida. {len(moradores)} morador(es) processado(s).")


def resetar_banco():
    sucesso, msg = limpar_banco()
    print(f"[init_db] {msg}")
    migrar_para_sqlite()


if __name__ == '__main__':
    import sys
    if '--reset' in sys.argv:
        resetar_banco()
    else:
        migrar_para_sqlite()
