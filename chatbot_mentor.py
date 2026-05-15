# ============================================================
# GeoAI Mentor — Chatbot Mentor de Carreira para Geocientistas
# ============================================================

# -------- IMPORTS --------
import logging
import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

log = logging.getLogger(__name__)


# -------- CONFIGURAÇÃO DA API --------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "sua_chave_secreta_aqui":
    log.error("OPENAI_API_KEY não encontrada ou é o placeholder padrão.")
    log.error("Edite o arquivo .env com sua chave real da OpenAI.")
    sys.exit(1)

if not api_key.startswith("sk-"):
    log.error("OPENAI_API_KEY não segue o formato esperado (deve começar com 'sk-').")
    sys.exit(1)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    request_timeout=30,
    max_retries=2,
)


# ============================================================
# ETAPA 1 — Conexão Básica (sem memória, sem persona)
# ============================================================
def etapa_1_basico() -> None:
    """
    Demonstra funcionamento SEM memória de sessão.

    Executa duas perguntas sequenciais de forma independente,
    mostrando como o modelo perde contexto entre chamadas.
    """
    perguntas = [
        "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
        "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
    ]

    log.info("=" * 60)
    log.info("ETAPA 1 — Sem Memória (cada pergunta é independente)")
    log.info("=" * 60)

    for pergunta in perguntas:
        try:
            resposta = modelo.invoke(pergunta)
            log.info("Pergunta: %s", pergunta)
            log.info("Resposta: %s", resposta.content)
        except Exception as e:
            log.error("Erro na chamada à API: %s", e)


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
memoria_sessoes: dict[str, InMemoryChatMessageHistory] = {}


def obter_historico_por_sessao(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retorna o histórico de mensagens para a sessão informada.

    Implementa padrão singleton: cria um novo histórico apenas se
    o session_id ainda não existir no dicionário de sessões.

    Args:
        session_id: Identificador único da conversa.

    Returns:
        Objeto de histórico de mensagens associado à sessão.
    """
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
def main() -> None:
    """Executa a demonstração completa: sem memória vs com memória."""
    perguntas = [
        "Eu sou geofísico e quero migrar para a área de dados. Qual linguagem de programação devo aprender primeiro?",
        "E que tipo de projeto de portfólio eu poderia criar usando essa linguagem?"
    ]

    # --- ANTES: sem memória ---
    etapa_1_basico()

    # --- DEPOIS: com memória ---
    log.info("=" * 60)
    log.info("ETAPA 3 — Com Memória (conversa contínua)")
    log.info("=" * 60)

    config = {"configurable": {"session_id": "sessao_geocientista_01"}}

    for pergunta in perguntas:
        try:
            resposta = cadeia_com_memoria.invoke(
                {"query": pergunta},
                config=config
            )
            log.info("Você: %s", pergunta)
            log.info("GeoAI Mentor: %s", resposta)
        except Exception as e:
            log.error("Erro na cadeia com memória: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
