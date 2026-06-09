import sqlite3
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'leitor_placas.db'


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS moradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            placa TEXT UNIQUE NOT NULL,
            apartamento TEXT,
            veiculo TEXT,
            ativo INTEGER DEFAULT 1,
            criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS acessos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT NOT NULL,
            data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT CHECK(status IN ('LIBERADO','BLOQUEADO')),
            morador_id INTEGER,
            imagem_path TEXT,
            confianca REAL,
            observacao TEXT,
            FOREIGN KEY (morador_id) REFERENCES moradores(id)
        );

        CREATE INDEX IF NOT EXISTS idx_acessos_placa ON acessos(placa);
        CREATE INDEX IF NOT EXISTS idx_acessos_data ON acessos(data_hora DESC);
        CREATE INDEX IF NOT EXISTS idx_moradores_placa ON moradores(placa);
    """)
    conn.commit()
    conn.close()


# --- Moradores CRUD ---

def listar_moradores(apenas_ativos=True):
    conn = get_connection()
    if apenas_ativos:
        rows = conn.execute(
            "SELECT * FROM moradores WHERE ativo = 1 ORDER BY nome"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM moradores ORDER BY nome"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def buscar_morador_por_placa(placa):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM moradores WHERE placa = ? AND ativo = 1",
        (placa.upper().strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def adicionar_morador(nome, placa, apartamento=None, veiculo=None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO moradores (nome, placa, apartamento, veiculo) VALUES (?, ?, ?, ?)",
            (nome.strip(), placa.upper().strip(), apartamento, veiculo)
        )
        conn.commit()
        return True, "Morador cadastrado com sucesso."
    except sqlite3.IntegrityError:
        return False, "Placa já cadastrada."
    finally:
        conn.close()


def editar_morador(morador_id, nome=None, placa=None, apartamento=None, veiculo=None, ativo=None):
    conn = get_connection()
    updates = []
    params = []
    if nome is not None:
        updates.append("nome = ?")
        params.append(nome.strip())
    if placa is not None:
        updates.append("placa = ?")
        params.append(placa.upper().strip())
    if apartamento is not None:
        updates.append("apartamento = ?")
        params.append(apartamento)
    if veiculo is not None:
        updates.append("veiculo = ?")
        params.append(veiculo)
    if ativo is not None:
        updates.append("ativo = ?")
        params.append(1 if ativo else 0)
    if not updates:
        conn.close()
        return False, "Nenhum campo para alterar."
    params.append(morador_id)
    try:
        conn.execute(
            f"UPDATE moradores SET {', '.join(updates)} WHERE id = ?",
            params
        )
        conn.commit()
        conn.close()
        return True, "Morador atualizado com sucesso."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Placa já cadastrada por outro morador."


def remover_morador(morador_id):
    conn = get_connection()
    conn.execute("UPDATE moradores SET ativo = 0 WHERE id = ?", (morador_id,))
    conn.commit()
    conn.close()
    return True, "Morador desativado com sucesso."


# --- Acessos ---

def registrar_acesso(placa, status, morador_id=None, imagem_path=None, confianca=None, observacao=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO acessos (placa, status, morador_id, imagem_path, confianca, observacao)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (placa.upper().strip(), status, morador_id, imagem_path, confianca, observacao)
    )
    conn.commit()
    conn.close()


def historico_acessos(limite=50):
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.*, m.nome as morador_nome, m.apartamento
           FROM acessos a
           LEFT JOIN moradores m ON a.morador_id = m.id
           ORDER BY a.data_hora DESC
           LIMIT ?""",
        (limite,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def buscar_acessos_por_placa(placa, limite=20):
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.*, m.nome as morador_nome, m.apartamento
           FROM acessos a
           LEFT JOIN moradores m ON a.morador_id = m.id
           WHERE a.placa = ?
           ORDER BY a.data_hora DESC
           LIMIT ?""",
        (placa.upper().strip(), limite)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def estatisticas():
    conn = get_connection()
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'LIBERADO' THEN 1 ELSE 0 END) as liberados,
            SUM(CASE WHEN status = 'BLOQUEADO' THEN 1 ELSE 0 END) as bloqueados
        FROM acessos
    """).fetchone()
    total_moradores = conn.execute(
        "SELECT COUNT(*) as total FROM moradores WHERE ativo = 1"
    ).fetchone()
    conn.close()
    return {
        'total_acessos': stats['total'],
        'liberados': stats['liberados'] or 0,
        'bloqueados': stats['bloqueados'] or 0,
        'total_moradores': total_moradores['total']
    }


def limpar_banco():
    conn = get_connection()
    conn.executescript("""
        DELETE FROM acessos;
        DELETE FROM moradores;
    """)
    conn.commit()
    conn.close()
    return True, "Banco limpo com sucesso."


init_db()
