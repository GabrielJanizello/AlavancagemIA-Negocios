import os
import google.generativeai as genai
import streamlit as st

# Obtém a chave da API a partir da variável de ambiente
api_key = os.environ.get("GOOGLE_API_KEY")

# Verifica se a chave está definida
if not api_key:
    raise EnvironmentError("A variável de ambiente GOOGLE_API_KEY não está definida!")

# Configura a API do Gemini com a chave obtida
genai.configure(api_key=api_key)

# A partir daqui, você pode fazer chamadas à API normalmente
# Configura o cliente da SDK do Gemini
MODEL_ID = "gemini-2.0-flash"

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types  # Para criar conteúdos (Content e Part)
from google.adk.tools import google_search
import textwrap  # Para formatar melhor a saída de texto
import requests  # Para fazer requisições HTTP
import warnings

warnings.filterwarnings("ignore")

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
import uuid


def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()

    # Gera IDs únicos para usuário e sessão
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Cria a sessão
    session = session_service.create_session(
        app_name=agent.name,
        user_id=user_id,
        session_id=session_id
    )

    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    for event in runner.run(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text + "\n"
    return final_response


##########################################
# --- Agente 1: Cotação dos Produtos --- #
##########################################
def agente_cotador(produto):
    cotador = Agent(
        name="agente_cotador",
        model="gemini-2.0-flash",
        description="Agente que faz uma cotação dos valores dos produtos com base na pesquisa da internet com o Google Search.",
        tools=[google_search],
        instruction="""Você irá realizar uma busca para fazer uma cotação atual dos valores dos produtos/ingredientes usados.
        Utilize fontes confiáveis e atualizadas, como sites de supermercados, distribuidores ou portais especializados.
        Priorize preços médios de mercado, considerando variações regionais quando relevante.
        Registre o nome do produto, o preço, a unidade de medida e a fonte da informação."""
    )

    entrada_do_agente_cotador = f"Produto: {produto}"

    lancamentos = call_agent(cotador, entrada_do_agente_cotador)
    return lancamentos


##########################################
# --- Agente 2: Precificação dos Produtos --- #
##########################################
def agente_precificador(produto, lancamentos_cotador):
    precificador = Agent(
        name="agente_precificador",
        model="gemini-2.0-flash",
        description="Agente que faz uma precificação correta dos produtos.",
        instruction="""Você é um assistente especializado em precificação de produtos para pequenos e médios negócios.
        Com base nos dados de custo fornecidos pelo agente cotador, elabore uma precificação correta considerando margem de lucro, custos fixos e variáveis, e práticas de mercado.
        Forneça sugestões de preço final, margem aplicada e justificativas claras."""
    )

    entrada_do_agente_precificador = f"Produto: {produto}\nLançamentos cotados: {lancamentos_cotador}"

    precificador_dos_produtos = call_agent(precificador, entrada_do_agente_precificador)
    return precificador_dos_produtos


################################################
# --- Agente 3: Gestor de Informações --- #
################################################
def agente_informacoes(produto, lancamentos_precificador):
    informacoes = Agent(
        name="agente_informacoes",
        model="gemini-2.0-flash",
        description="Agente que analisa o produto precificado e fornece insights estratégicos, tendências de mercado e informações contextuais relevantes",
        instruction="""Você é um gestor de informações estratégicas para negócios. Com base no produto e nos dados de precificação fornecidos, sua tarefa é:
        1. Identificar 2-3 tendências de mercado atuais relevantes para este produto.
        2. Apontar 1-2 oportunidades e 1-2 riscos potenciais associados à comercialização deste produto com a precificação sugerida.
        3. Fornecer uma breve análise da concorrência, se informações estiverem disponíveis publicamente (sem realizar novas buscas, apenas com base no conhecimento geral e nos dados já fornecidos).
        4. Sugerir um público-alvo ideal para o produto com base nas informações.
        Seja conciso e forneça informações acionáveis.""",
    )

    entrada_do_agente_informacoes = f"Produto:{produto}\nLançamentos precificados: {lancamentos_precificador}"
    gestor_de_informacoes = call_agent(informacoes, entrada_do_agente_informacoes)
    return gestor_de_informacoes


##########################################
# --- Agente 4: Apresentador do Relatório/Documentação --- #
##########################################
def agente_apresentador(produto, gestor_de_informacoes):
    apresentador = Agent(
        name="agente_apresentador",
        model="gemini-2.0-flash",
        description="Agente criador e apresentador dos dados gerados.",
        instruction="""O agente apresentador é responsável por revisar, sintetizar e apresentar os dados e análises produzidos pelos demais agentes,
        transformando-os em relatórios claros, objetivos e acionáveis para o empresário. Você deve Analisar todas as respostas e informações geradas pelos outros agentes.
        Corrigir erros de linguagem, inconsistências nos dados ou interpretações equivocadas.
        Garantir que todas as informações estejam atualizadas, bem fundamentadas e organizadas.
        Eliminar redundâncias, simplificar termos e destacar o que é realmente relevante.
        Resumir os principais pontos em tópicos claros, utilizando recursos como listas, quadros comparativos, e gráficos quando necessário."""
    )
    entrada_do_agente_apresentador = f"Produto: {produto}\nRascunho: {gestor_de_informacoes}"
    informacao_completa = call_agent(apresentador, entrada_do_agente_apresentador)
    return informacao_completa


# --- Streamlit UI ---
st.title("🚀 Sistema AlavancagemIA de Negócios! 🚀")

st.markdown("**Informe as suas receitas que deseja precificar, produtos e ticket médio e deixe que faremos o resto!**\n\n"
            "**Trabalhamos melhor com você!** \n\n Quanto mais precisas forem suas informações, melhores serão as soluções que podemos criar juntos.")

# Entrada de produto
produto = st.text_input("❓ Por favor, digite o nome dos produtos/receitas para análise: ")

# Verifica se o produto foi fornecido
if produto:
    st.write(f"🔍 Analisando o produto: **{produto}**")

    # Chama os agentes
    lancamentos_cotador = agente_cotador(produto)
    st.subheader('🔢 Resultados do Agente 1 - Cotação dos Valores.')
    st.write(lancamentos_cotador)

    lancamentos_precificador = agente_precificador(produto, lancamentos_cotador)
    st.subheader('🔢 Resultados do Agente 2 - Precificação')
    st.write(lancamentos_precificador)

    gestor_info = agente_informacoes(produto, lancamentos_precificador)
    st.subheader('📊 Resultados do Agente 3 - Informações Estratégicas')
    st.write(gestor_info)

    relatorio_final = agente_apresentador(produto, gestor_info)
    st.subheader('📋 Relatório Final do Agente 4')
    st.write(relatorio_final)

    st.success("✅ Sistema finalizado com sucesso!")
else:
    st.warning("⚠️ Você precisa fornecer um nome de produto para análise!")
