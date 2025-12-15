# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:29:39 2025

@author: joses
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu


# -------- MENU LATERAL --------
with st.sidebar:
    escolha = option_menu(
        "Menu",
        ["Home", "Setor Real", "Setor Fiscal", "Setor Monetário", "Setor Externo", "Principais Rácios Macroeconómicos"],
        menu_icon="cast",
        icons=["house","building", "cash-stack", "bank", "globe", "speedometer2"],
        default_index=0
    )

# -------- FUNÇÃO PARA CARREGAR EXCEL --------
@st.cache_data
def carregar_excel(ficheiro, folha=None):
    return pd.read_excel(ficheiro, sheet_name=folha)

# -------- CAMINHO DO FICHEIRO EXCEL --------
FICHEIRO_EXCEL = r"C:\Users\joses\Documents\PYTHON\Indicadores_Macro.xlsx"
FICHEIRO_EXCEL = "Indicadores_Macro.xlsx"

st.set_page_config(
    page_title="Dashboard Macroeconómico de Cabo Verde",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)


if escolha == "Home":
    st.title("📊 Dashboard Macroeconómico de Cabo Verde")
    st.markdown("""
    O dashboard macroeconómico de Cabo Verde é uma ferramenta interativa que permite acompanhar 
    e analisar os principais indicadores económicos do país. Ele integra dados dos setores **real**, 
    **fiscal**, **monetário**, **externo** e principais **rácios macroeconómicos**, apresentando-os 
    em gráficos dinâmicos, tabelas e métricas-chave.

    Esta ferramenta facilita a visualização de **tendências económicas**, comparações anuais e a 
    identificação de padrões, apoiando a **tomada de decisões**, o **planeamento estratégico** e a 
    comunicação clara de informações económicas complexas em Cabo Verde.
    """)

if escolha == "Setor Real":
    st.title("📊 Setor Real")
    df = carregar_excel(FICHEIRO_EXCEL, folha="PIB")

    # Slider para selecionar o período
    intervalo = st.slider("Selecione o período",
        int(df["Ano"].min()),
        int(df["Ano"].max()),
        (int(df["Ano"].min()), int(df["Ano"].max()))
    )

    # Filtrar dados pelo intervalo
    df_f = df[(df["Ano"] >= intervalo[0]) & (df["Ano"] <= intervalo[1])]
    ultimo = df_f.iloc[-1]
    st.markdown(f"**Dados referentes ao ano :** {int(ultimo['Ano'])}")

    # Métricas principais
    col1, col2, col3 = st.columns(3)
    col1.metric("PIB_real (milhões de CVE)", f"{ultimo['PIB_real']:,.0f}")
    col2.metric("Crescimento PIB", f"{ultimo['Crescimento']:.1f}%")
    col3.metric("Inflação", f"{ultimo['Inflacao']:.1f}%")

    # Layout: tabela à esquerda, gráfico à direita
    col_table, col_chart = st.columns([1.3, 1.7])

    # -------- TABELA BONITA COM PLOTLY --------
    with col_table:
        import plotly.graph_objects as go

        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=["Ano", "PIB Real (milhões CVE)", "Crescimento PIB (%)", "Inflação (%)"],
                fill_color='lightblue',
                align='center',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=[
                    df_f["Ano"],
                    df_f["PIB_real"].apply(lambda x: f"{x:,.0f}"),
                    df_f["Crescimento"].apply(lambda x: f"{x:.1f}%"),
                    df_f["Inflacao"].apply(lambda x: f"{x:.1f}%")
                ],
                fill_color='white',
                align='center',
                font=dict(color='black', size=11)
            )
        )])

        st.plotly_chart(fig_table, use_container_width=True)

    # -------- GRÁFICO INTERATIVO COM PLOTLY --------
    with col_chart:
        import plotly.express as px

        # Gráfico combinado: barras para Crescimento PIB, linha para PIB real
        fig = px.line(
            df_f,
            x="Ano",
            y="PIB_real",
            markers=True,
            text=df_f["PIB_real"].apply(lambda x: f"{x:,.0f}"),
            title="Evolução do PIB Real (milhões de CVE)",
            labels={"PIB_real": "PIB Real (milhões de CVE)", "Ano": "Ano"}
        )

        # Estilização
        fig.update_traces(line=dict(color='blue', width=3), marker=dict(size=10, color='blue'), textposition='top center')
        fig.update_layout(template="plotly_white", xaxis=dict(dtick=1))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        st.plotly_chart(fig, use_container_width=True)
        # Fonte / nota abaixo da tabela
        st.markdown("<p style='text-align:center; font-size:12px; color:gray;'>Fonte: Instituto Nacional de Estatística de Cabo Verde</p>", unsafe_allow_html=True)

# -------- SETOR FISCAL --------
elif escolha == "Setor Fiscal":
    st.title("💰 Setor Fiscal")
    df = carregar_excel(FICHEIRO_EXCEL, folha="Fiscal")
    intervalo = st.slider("Selecione o período", int(df["Ano"].min()), int(df["Ano"].max()), (int(df["Ano"].min()), int(df["Ano"].max())))
    df_f = df[(df["Ano"] >= intervalo[0]) & (df["Ano"] <= intervalo[1])]
    ultimo = df_f.iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric("📥 Receitas", f"{ultimo['Receitas']:,.0f} Milhões de CVE", f"{ultimo['CrescimentoReceitas']:.1f}%")
    col2.metric("📤 Despesas", f"{ultimo['Despesas']:,.0f} Milhões de CVE", f"{ultimo['CrescimentoDespesas']:.1f}%")

    col_table, col_chart = st.columns([1.8,1.2])
    with col_table: st.dataframe(df_f)
    with col_chart:
        fig = px.bar(x=["Receitas","Despesas"], y=[ultimo["Receitas"], ultimo["Despesas"]],
                     text=[f"{ultimo['Receitas']:,.0f}", f"{ultimo['Despesas']:,.0f}"], color=["Receitas","Despesas"])
        fig.update_layout(title="Receitas vs Despesas", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# -------- SETOR MONETÁRIO --------
elif escolha == "Setor Monetário":
    st.title("🏦 Setor Monetário")
    df = carregar_excel(FICHEIRO_EXCEL, folha="Monetario")

    # Slider de anos
    intervalo = st.slider("Selecione o período",
        int(df["Ano"].min()),
        int(df["Ano"].max()),
        (int(df["Ano"].min()), int(df["Ano"].max()))
    )

    # Filtrar dados
    df_f = df[(df["Ano"] >= intervalo[0]) & (df["Ano"] <= intervalo[1])]
    ultimo = df_f.iloc[-1]

    # Métricas principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Taxa de Juro", f"{ultimo['TaxaJuro']:.1f}%")
    col2.metric("Massa Monetária (M2)", f"{ultimo['MassaMonetaria']:,.0f}")
    col3.metric("Crédito à Economia (Milhões de CVE)", f"{ultimo['CreditoEconomia']:,.0f}")

    # Layout: gráfico das taxas de juros à esquerda, tabela à direita
    col_chart, col_table = st.columns([1.4, 1.6])
    with col_chart:
        fig, ax = plt.subplots()
        ax.plot(df_f["Ano"], df_f["TaxaJuro"], marker="o", color="blue")
        ax.set_title("Evolução da Taxa de Juro")
        ax.set_xlabel("Ano")
        ax.set_ylabel("Taxa de Juro (%)")
        ax.grid(False)

        # Adicionar rótulos nos pontos
        for x, y in zip(df_f["Ano"], df_f["TaxaJuro"]):
            ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

        st.pyplot(fig)

    with col_table:
        st.subheader("Massa Monetária e Crédito à Economia")
        st.dataframe(df_f[["Ano", "MassaMonetaria", "CreditoEconomia"]])

# -------- SETOR EXTERNO --------
elif escolha == "Setor Externo":
    st.title("🌍 Setor Externo")
    df = carregar_excel(FICHEIRO_EXCEL, folha="Externo")

    # Slider de anos
    intervalo = st.slider("Selecione o período",
        int(df["Ano"].min()),
        int(df["Ano"].max()),
        (int(df["Ano"].min()), int(df["Ano"].max()))
    )

    # Filtrar dados
    df_f = df[(df["Ano"] >= intervalo[0]) & (df["Ano"] <= intervalo[1])]
    ultimo = df_f.iloc[-1]

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Exportações (Milhões de CVE)", f"{ultimo['Exportacoes']:,.0f}")
    col2.metric("Importações (Milhões de CVE)", f"{ultimo['Importacoes']:,.0f}")
    saldo_ultimo = ultimo['Exportacoes'] - ultimo['Importacoes']
    col3.metric("Saldo da Balança", f"{saldo_ultimo:,.0f}")

    # Layout: tabela à esquerda, gráfico à direita
    col_table, col_chart = st.columns([1.6, 1.4])
    with col_table:
        st.dataframe(df_f[["Ano", "Exportacoes", "Importacoes"]])

    with col_chart:
        fig, ax = plt.subplots()

        # Gráfico de barras
        x = np.arange(len(df_f["Ano"]))
        width = 0.35
        ax.bar(x - width/2, df_f["Exportacoes"], width, label="Exportações", color="green")
        ax.bar(x + width/2, df_f["Importacoes"], width, label="Importações", color="red")

        # Linha para o saldo
        saldo = df_f["Exportacoes"] - df_f["Importacoes"]
        ax.plot(x, saldo, marker="o", color="blue", label="Saldo")

        ax.set_xticks(x)
        ax.set_xticklabels(df_f["Ano"])
        ax.set_title("Setor Externo: Exportações, Importações e Saldo")
        ax.set_ylabel("Milhões")
        ax.grid(False)
        ax.legend()
     
        st.pyplot(fig)

# -------- RÁCIOS MACRO --------
elif escolha == "Principais Rácios Macroeconómicos":
    st.title("📈 Principais Rácios Macroeconómicos")
    df = carregar_excel(FICHEIRO_EXCEL, folha="Racios")

    # Slider de anos
    intervalo = st.slider(
        "Selecione o período",
        int(df["Ano"].min()),
        int(df["Ano"].max()),
        (int(df["Ano"].min()), int(df["Ano"].max()))
    )

    # Filtrar dados pelo período
    df_f = df[(df["Ano"] >= intervalo[0]) & (df["Ano"] <= intervalo[1])]
    ultimo = df_f.iloc[-1]
    st.markdown(f"**Dados referentes ao ano :** {int(ultimo['Ano'])}")

    # Métricas principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Dívida / PIB", f"{ultimo['DividaPIB']:.1f}%")
    col2.metric("Défice / PIB", f"{ultimo['DeficePIB']:.1f}%")
    col3.metric("Investimento / PIB", f"{ultimo['InvestimentoPIB']:.1f}%")

    # -------- COMBOBOX DE FILTRO --------
    opcao_racio = st.selectbox(
        "Filtrar por rácio",
        ["Todos", "Dívida / PIB", "Défice / PIB", "Investimento / PIB"]
    )

    # Ajustar o DataFrame de acordo com a seleção
    if opcao_racio == "Dívida / PIB":
        df_f_plot = df_f[["Ano", "DividaPIB"]]
    elif opcao_racio == "Défice / PIB":
        df_f_plot = df_f[["Ano", "DeficePIB"]]
    elif opcao_racio == "Investimento / PIB":
        df_f_plot = df_f[["Ano", "InvestimentoPIB"]]
    else:
        df_f_plot = df_f.copy()  # Todos os rácios

    # Layout: tabela à esquerda, gráfico à direita
    col_table, col_chart = st.columns([1.3, 1.7])

    # -------- TABELA BONITA COM PLOTLY --------
    with col_table:
        import plotly.graph_objects as go

        if opcao_racio == "Todos":
            header_values = ["Ano", "Dívida / PIB", "Défice / PIB", "Investimento / PIB"]
            cell_values = [
                df_f["Ano"],
                [f"{v:.1f}%" for v in df_f["DividaPIB"]],
                [f"{v:.1f}%" for v in df_f["DeficePIB"]],
                [f"{v:.1f}%" for v in df_f["InvestimentoPIB"]]
            ]
        else:
            header_values = df_f_plot.columns.tolist()
            cell_values = [
                df_f_plot[col] if col=="Ano" else [f"{v:.1f}%" for v in df_f_plot[col]] 
                for col in df_f_plot.columns
            ]

        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color='lightblue',
                align='center',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=cell_values,
                fill_color='white',
                align='center',
                font=dict(color='black', size=11)
            )
        )])

        st.plotly_chart(fig_table, use_container_width=True)
        # Fonte / nota abaixo da tabela
        st.markdown("<p style='text-align:center; font-size:12px; color:gray;'>Fonte: Autor</p>", unsafe_allow_html=True)


    # -------- GRÁFICO INTERATIVO COM PLOTLY --------
    with col_chart:
        fig = go.Figure()

        if opcao_racio == "Dívida / PIB":
            fig.add_trace(go.Bar(
                x=df_f_plot["Ano"],
                y=df_f_plot["DividaPIB"],
                name="Dívida / PIB",
                marker_color='gray',
                text=[f"{v:.1f}%" for v in df_f_plot["DividaPIB"]],
                textposition='outside'
            ))
        elif opcao_racio == "Défice / PIB":
            fig.add_trace(go.Scatter(
                x=df_f_plot["Ano"],
                y=df_f_plot["DeficePIB"],
                name="Défice / PIB",
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_f_plot["DeficePIB"]],
                textposition='top center',
                line=dict(color='red', width=2)
            ))
        elif opcao_racio == "Investimento / PIB":
            fig.add_trace(go.Scatter(
                x=df_f_plot["Ano"],
                y=df_f_plot["InvestimentoPIB"],
                name="Investimento / PIB",
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_f_plot["InvestimentoPIB"]],
                textposition='top center',
                line=dict(color='green', width=2)
            ))
        else:
            # Todos os rácios
            fig.add_trace(go.Bar(
                x=df_f["Ano"],
                y=df_f["DividaPIB"],
                name="Dívida / PIB",
                marker_color='gray',
                text=[f"{v:.1f}%" for v in df_f["DividaPIB"]],
                textposition='outside'
            ))
            fig.add_trace(go.Scatter(
                x=df_f["Ano"],
                y=df_f["DeficePIB"],
                name="Défice / PIB",
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_f["DeficePIB"]],
                textposition='top center',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_f["Ano"],
                y=df_f["InvestimentoPIB"],
                name="Investimento / PIB",
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in df_f["InvestimentoPIB"]],
                textposition='top center',
                line=dict(color='green', width=2)
            ))

        fig.update_layout(
            title="Rácios Macroeconómicos em % do PIB",
            yaxis_title="%",
            xaxis_title="Ano",
            barmode='group',
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)
        # Fonte / nota abaixo da tabela
        st.markdown("<p style='text-align:center; font-size:12px; color:gray;'>Fonte: Autor</p>", unsafe_allow_html=True)







