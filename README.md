# Leitor de Placas Veiculares

Sistema de controle de acesso baseado em reconhecimento de placas veiculares utilizando visão computacional e OCR.

## Arquitetura

```
leitor-de-placas/
├── pre_processamento.py   # Pipeline principal: Haar Cascade + EasyOCR + correção
├── banco.py               # Banco SQLite (moradores, acessos)
├── app.py                 # Interface Streamlit (Dashboard + Portaria + Cadastro)
├── init_db.py             # Migração de moradores.json → SQLite
├── acesso.py              # Módulo legado (JSON/CSV) — preservado
├── testes_batch.py        # Testes automatizados em lote
├── test_images/           # 13 imagens de teste
├── carro.jpg              # Imagem demo (Hyundai HB20 - GEP2C00)
├── moradores.json         # Dados seed (4 moradores)
├── requirements.txt       # Dependências
└── leitor_placas.db       # Banco SQLite (criado automaticamente)
```

## Pipeline de Detecção

1. **Pré-processamento**: redimensionamento para max 1000px largura
2. **Detecção de candidatos**:
   - Haar Cascade (`haarcascade_russian_plate_number.xml`) — detecção primária
   - Sobel + fechamento morfológico — fallback quando Haar não encontra nada
   - Fallback para imagem inteira quando proporção (2.0–5.5) sugere close-up
3. **OCR Multi-ângulo**: rotações de 5° a 9° (normal) ou 0° a 8° (close-up), com inversão de cores nos ângulos 7 e 4
4. **Correção posicional**: regras específicas para cada posição da placa (Mercosul e formato antigo)
5. **Verificação de acesso**: busca no SQLite por morador com a placa detectada

## Instalação

```bash
# 1. Criar ambiente virtual
python -m venv env

# 2. Ativar (Windows)
env\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Migrar dados (opcional)
python init_db.py
```

## Uso

### Interface Web (recomendado)
```bash
streamlit run app.py
```

### Linha de comando (teste rápido)
```bash
python pre_processamento.py
```

### Testes em lote
```bash
python testes_batch.py
```

## Resultados

| Métrica | Valor |
|---------|-------|
| Imagens de teste | 13 |
| Placas detectadas | 11/13 (84.6%) |
| Placas lidas corretamente | 3/13 (23.1%) |
| Tempo médio (1ª execução) | ~40s |
| Tempo médio (execuções seguintes) | ~35s |

## Tecnologias

- **Python 3.13**
- **OpenCV** (Haar Cascade, processamento de imagem)
- **EasyOCR** (OCR baseado em deep learning)
- **SQLite** (persistência)
- **Streamlit** (interface web)
- **NumPy / Pandas**

## Autores

Projeto acadêmico — Controle de Acesso Veicular com Visão Computacional.
