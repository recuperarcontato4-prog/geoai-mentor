"""Smoke test — verifica imports e funcoes auxiliares sem chamar LLM."""
from unittest.mock import patch, MagicMock

import pytest


def test_obter_historico_cria_sessao():
    """Historico de sessao deve ser criado sob demanda."""
    # Precisamos mockar a API key antes de importar o modulo
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-test-key-1234567890"}):
        with patch("langchain_openai.ChatOpenAI"):
            from chatbot_mentor import obter_historico_por_sessao, memoria_sessoes
            h = obter_historico_por_sessao("test_session_42")
            assert "test_session_42" in memoria_sessoes
            assert h is memoria_sessoes["test_session_42"]


def test_obter_historico_reutiliza_sessao():
    """Mesma sessao retorna mesmo objeto."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake-test-key-1234567890"}):
        with patch("langchain_openai.ChatOpenAI"):
            from chatbot_mentor import obter_historico_por_sessao
            h1 = obter_historico_por_sessao("test_reuse_99")
            h2 = obter_historico_por_sessao("test_reuse_99")
            assert h1 is h2
