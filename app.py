import os
import sys
import tempfile
import time
import streamlit as st
import pandas as pd
from PIL import Image
import easyocr

from banco import (
    listar_moradores, buscar_morador_por_placa, adicionar_morador,
    editar_morador, remover_morador, historico_acessos,
    buscar_acessos_por_placa, registrar_acesso, estatisticas
)
from pre_processamento import processar_imagem

st.set_page_config(
    page_title="Controle de Acesso - Leitor de Placas",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def carregar_reader():
    if 'reader' not in st.session_state:
        with st.spinner("Carregando modelo EasyOCR..."):
            st.session_state.reader = easyocr.Reader(
                ['en'], gpu=False, verbose=False
            )
    return st.session_state.reader


def pagina_dashboard():
    st.title("📊 Dashboard")

    stats = estatisticas()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Acessos", stats['total_acessos'])
    col2.metric("Liberados", stats['liberados'], delta_color="normal")
    col3.metric("Bloqueados", stats['bloqueados'], delta_color="inverse")
    col4.metric("Moradores Cadastrados", stats['total_moradores'])

    st.divider()

    st.subheader("Últimos Acessos")
    acessos = historico_acessos(limite=50)
    if acessos:
        df = pd.DataFrame(acessos)
        df['data_hora'] = pd.to_datetime(df['data_hora']).dt.strftime('%d/%m/%Y %H:%M:%S')
        df['status'] = df['status'].apply(
            lambda x: f"✅ {x}" if x == 'LIBERADO' else f"❌ {x}"
        )
        df['morador'] = df.apply(
            lambda r: f"{r['morador_nome'] or '-'} ({r['apartamento'] or '-'})"
            if r['morador_nome'] else '-',
            axis=1
        )
        display = df[['data_hora', 'placa', 'status', 'morador']]
        display.columns = ['Data/Hora', 'Placa', 'Status', 'Morador']
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum acesso registrado ainda.")

    st.divider()
    st.subheader("Buscar por Placa")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        placa_busca = st.text_input("Placa", placeholder="Ex: GEP2C00").upper()
    with col_b:
        st.write("")
        st.write("")
        if st.button("Buscar", use_container_width=True) and placa_busca:
            resultados = buscar_acessos_por_placa(placa_busca)
            if resultados:
                df2 = pd.DataFrame(resultados)
                df2['data_hora'] = pd.to_datetime(df2['data_hora']).dt.strftime('%d/%m/%Y %H:%M:%S')
                df2['status'] = df2['status'].apply(
                    lambda x: f"✅ {x}" if x == 'LIBERADO' else f"❌ {x}"
                )
                st.dataframe(
                    df2[['data_hora', 'placa', 'status']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Nenhum registro encontrado para esta placa.")

    st.divider()
    st.subheader("Moradores Cadastrados")
    moradores = listar_moradores()
    if moradores:
        df3 = pd.DataFrame(moradores)
        df3 = df3[['nome', 'placa', 'apartamento', 'veiculo']]
        df3.columns = ['Nome', 'Placa', 'Apartamento', 'Veículo']
        st.dataframe(df3, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum morador cadastrado.")

    if st.button("🔄 Atualizar Dados"):
        st.rerun()


def pagina_portaria():
    st.title("🚧 Portaria - Controle de Acesso")

    reader = carregar_reader()
    moradores = listar_moradores()

    tab_upload, tab_webcam = st.tabs(["📁 Upload de Imagem", "📷 Webcam"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Selecione uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            col_img, col_result = st.columns([1, 1])

            with col_img:
                imagem = Image.open(uploaded_file)
                st.image(imagem, caption="Imagem enviada", use_container_width=True)

            with col_result:
                with st.spinner("Processando imagem..."):
                    t0 = time.time()
                    resultado = processar_imagem(
                        tmp_path, reader, moradores,
                        output_dir='tests', verbose=False
                    )
                    elapsed = time.time() - t0

                st.metric("Tempo de processamento", f"{elapsed:.1f}s")

                if resultado['placa']:
                    placa = resultado['placa']
                    status = resultado['status']
                    liberado = resultado['liberado']
                    morador = resultado['morador']

                    if liberado:
                        st.success(f"✅ ACESSO LIBERADO")
                    else:
                        st.error(f"❌ ACESSO BLOQUEADO")

                    st.markdown(f"### Placa: `{placa}`")

                    if morador:
                        st.markdown(f"**Morador:** {morador['nome']}")
                        st.markdown(f"**Apartamento:** {morador['apartamento']}")
                        st.markdown(f"**Veículo:** {morador.get('veiculo', 'N/A')}")
                    else:
                        st.warning("Placa não cadastrada no sistema.")

                    morador_id = morador['id'] if morador else None
                    registrar_acesso(
                        placa=placa,
                        status=status,
                        morador_id=morador_id,
                        imagem_path=resultado.get('img_path'),
                    )

                    if os.path.exists(resultado.get('img_path', '')):
                        img_result = Image.open(resultado['img_path'])
                        st.image(img_result, caption="Resultado", use_container_width=True)
                else:
                    st.warning("Nenhuma placa válida detectada na imagem.")

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    with tab_webcam:
        st.info(
            "📸 Para usar a webcam, clique no botão abaixo. "
            "A captura será processada automaticamente."
        )
        camera_image = st.camera_input("Tirar foto")
        if camera_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(camera_image.getvalue())
                tmp_path = tmp.name

            with st.spinner("Processando imagem da webcam..."):
                t0 = time.time()
                resultado = processar_imagem(
                    tmp_path, reader, moradores,
                    output_dir='tests', verbose=False
                )
                elapsed = time.time() - t0

            st.metric("Tempo de processamento", f"{elapsed:.1f}s")

            if resultado['placa']:
                placa = resultado['placa']
                liberado = resultado['liberado']
                morador = resultado['morador']

                if liberado:
                    st.success(f"✅ ACESSO LIBERADO - Placa: `{placa}`")
                    if morador:
                        st.markdown(f"{morador['nome']} - Apto {morador['apartamento']}")
                else:
                    st.error(f"❌ ACESSO BLOQUEADO - Placa: `{placa}`")

                morador_id = morador['id'] if morador else None
                registrar_acesso(
                    placa=placa,
                    status=resultado['status'],
                    morador_id=morador_id,
                    imagem_path=resultado.get('img_path'),
                )

                if os.path.exists(resultado.get('img_path', '')):
                    st.image(resultado['img_path'], caption="Resultado", use_container_width=True)
            else:
                st.warning("Nenhuma placa válida detectada.")

            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def pagina_cadastro():
    st.title("👤 Cadastro de Moradores")

    tab_lista, tab_novo = st.tabs(["📋 Moradores Cadastrados", "➕ Novo Morador"])

    with tab_lista:
        moradores = listar_moradores(apenas_ativos=True)
        if moradores:
            df = pd.DataFrame(moradores)
            df_exibir = df[['id', 'nome', 'placa', 'apartamento', 'veiculo']].copy()
            df_exibir.columns = ['ID', 'Nome', 'Placa', 'Apt', 'Veículo']
            st.dataframe(df_exibir, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Editar / Desativar Morador")

            opcoes = {m['id']: f"{m['nome']} ({m['placa']})" for m in moradores}
            selected_id = st.selectbox(
                "Selecione um morador",
                options=list(opcoes.keys()),
                format_func=lambda x: opcoes[x],
                key="edit_select"
            )

            if selected_id:
                m = next((m for m in moradores if m['id'] == selected_id), None)
                if m:
                    with st.form("form_editar"):
                        novo_nome = st.text_input("Nome", value=m['nome'])
                        nova_placa = st.text_input("Placa", value=m['placa']).upper()
                        novo_apto = st.text_input("Apartamento", value=m['apartamento'] or '')
                        novo_veiculo = st.text_input("Veículo", value=m['veiculo'] or '')

                        col_s, col_d = st.columns(2)
                        with col_s:
                            salvar = st.form_submit_button("💾 Salvar Alterações", use_container_width=True)
                        with col_d:
                            desativar = st.form_submit_button("🗑️ Desativar Morador", use_container_width=True)

                        if salvar:
                            ok, msg = editar_morador(
                                selected_id,
                                nome=novo_nome,
                                placa=nova_placa,
                                apartamento=novo_apto or None,
                                veiculo=novo_veiculo or None,
                            )
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

                        if desativar:
                            ok, msg = remover_morador(selected_id)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
        else:
            st.info("Nenhum morador cadastrado. Use a aba '➕ Novo Morador' para adicionar.")

    with tab_novo:
        with st.form("form_novo_morador"):
            nome = st.text_input("Nome completo *")
            placa = st.text_input("Placa do veículo *", placeholder="Ex: GEP2C00").upper()
            apto = st.text_input("Apartamento", placeholder="Ex: 101")
            veiculo = st.text_input("Veículo", placeholder="Ex: Hyundai HB20")
            submit = st.form_submit_button("✅ Cadastrar", use_container_width=True)

            if submit:
                erros = []
                if not nome.strip():
                    erros.append("Nome é obrigatório.")
                if not placa.strip():
                    erros.append("Placa é obrigatória.")
                if erros:
                    for e in erros:
                        st.error(e)
                else:
                    ok, msg = adicionar_morador(
                        nome=nome, placa=placa,
                        apartamento=apto or None,
                        veiculo=veiculo or None,
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


PAGES = {
    "📊 Dashboard": pagina_dashboard,
    "🚧 Portaria": pagina_portaria,
    "👤 Cadastro": pagina_cadastro,
}

st.sidebar.title("🚗 Leitor de Placas")
st.sidebar.markdown("---")
selected_page = st.sidebar.radio("Navegação", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Sistema de Controle de Acesso")
st.sidebar.caption("v1.0 - Projeto Acadêmico")

PAGES[selected_page]()
