# ============================================================
# GeoAI Mentor — Chatbot Mentor de Carreira para Geocientistas
# ============================================================

# -------- IMPORTS --------
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -------- CONFIGURAÇÃO DA API --------
load_dotenv()

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)


# ============================================================
# ETAPA 1 — Conexão Básica (sem memória, sem persona)
# ============================================================
def etapa_1_basico():
    """Demonstra que SEM memória, a 2ª pergunta perde contexto."""
    perguntas = [
        "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
        "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
    ]

    print("=" * 60)
    print("ETAPA 1 — Sem Memória (cada pergunta é independente)")
    print("=" * 60)

    for pergunta in perguntas:
        resposta = modelo.invoke(pergunta)
        print(f"\n🧑 Pergunta: {pergunta}")
        print(f"🤖 Resposta: {resposta.content}")


# ============================================================
# ETAPA 2 — Persona + Prompt Template + LCEL Chain
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o 'GeoAI Mentor', um assistente especializado em ajudar "
     "geocientistas (geofísicos, geólogos) a migrar para a área de "
     "Ciência de Dados e Inteligência Artificial. Seja amigável, "
     "didático e conecte sempre os conceitos ao domínio das Geociências."),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])

cadeia = prompt | modelo | StrOutputParser()


# ============================================================
# ETAPA 3 — Memória de Sessão (RunnableWithMessageHistory)
# ============================================================
memoria_sessoes = {}


def obter_historico_por_sessao(session_id: str) -> InMemoryChatMessageHistory:
    """Padrão singleton: garante 1 histórico por session_id."""
    if session_id not in memoria_sessoes:
        memoria_sessoes[session_id] = InMemoryChatMessageHistory()
    return memoria_sessoes[session_id]


cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=obter_historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)


# ============================================================
# LOOP PRINCIPAL
# ============================================================
def main():
    perguntas = [
        "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
        "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
    ]

    # --- ANTES: sem memória ---
    etapa_1_basico()

    # --- DEPOIS: com memória ---
    print("\n")
    print("=" * 60)
    print("ETAPA 3 — Com Memória (conversa contínua)")
    print("=" * 60)

    config = {"configurable": {"session_id": "sessao_geocientista_01"}}

    for pergunta in perguntas:
        resposta = cadeia_com_memoria.invoke(
            {"query": pergunta},
            config=config
        )
        print(f"\n🧑 Você: {pergunta}")
        print(f"🤖 GeoAI Mentor: {resposta}")


if __name__ == "__main__":
    main()
