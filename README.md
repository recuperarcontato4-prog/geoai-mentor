# GeoAI Mentor

Chatbot assistente especializado em ajudar geocientistas (geofísicos, geólogos) a migrar para a área de Ciência de Dados e Inteligência Artificial.

Desenvolvido como desafio do curso de IA da Alura, utilizando LangChain + OpenAI com memória de sessão.

## Arquitetura

```
┌──────────────────────┐
│   Pergunta do User   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ChatPromptTemplate  │  ← Persona "GeoAI Mentor"
│  + {historico}       │
│  + {query}           │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   ChatOpenAI (LLM)   │  ← gpt-3.5-turbo, temp=0.7
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  StrOutputParser     │  ← Limpa output para string
└──────────┬───────────┘
           ▼
┌──────────────────────────────────┐
│  RunnableWithMessageHistory      │  ← Memória de sessão
│  (InMemoryChatMessageHistory)    │
└──────────────────────────────────┘
```

A cadeia LCEL (`prompt | modelo | parser`) é envolvida pelo `RunnableWithMessageHistory`, que injeta automaticamente o histórico da conversa no placeholder `{historico}` do template. Isso transforma uma IA sem estado em um mentor que lembra do contexto.

## Como configurar

```bash
# 1. Clone o repositório
git clone https://github.com/recuperarcontato4-prog/geoai-mentor.git
cd geoai-mentor

# 2. Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure sua chave da OpenAI
cp .env.example .env
# Edite o .env e cole sua chave
```

## Como executar

```bash
python chatbot_mentor.py
```

O script executa automaticamente duas demonstrações:
- **Etapa 1 (Sem Memória):** Cada pergunta é independente — o modelo perde contexto
- **Etapa 3 (Com Memória):** Conversa contínua — o mentor lembra das respostas anteriores

## Exemplo de Interação

### SEM memória (Etapa 1)

```
Pergunta 1: "Sou geofísico e quero migrar para dados. Qual linguagem aprender?"
Resposta: "Python, SQL, R..."

Pergunta 2: "Que projeto de portfólio eu poderia criar usando essa linguagem?"
Resposta: "Jogo da velha, bot de redes sociais, app de previsão do tempo..."
           ↑ Perdeu completamente o contexto de geofísica!
```

### COM memória (Etapa 3)

```
Pergunta 1: "Sou geofísico e quero migrar para dados. Qual linguagem aprender?"
Resposta: "Python! Muito usado em Geociências com NumPy, Pandas..."

Pergunta 2: "Que projeto de portfólio eu poderia criar usando essa linguagem?"
Resposta: "Análise de dados sísmicos, visualização geoespacial,
           previsão de terremotos com machine learning..."
           ↑ Lembrou que você é geofísico e conectou ao domínio!
```

## Tecnologias

- **Python 3.12**
- **LangChain** — Framework para aplicações com LLMs
- **OpenAI GPT-3.5-turbo** — Modelo de linguagem
- **LCEL** — LangChain Expression Language para composição de cadeias
- **InMemoryChatMessageHistory** — Memória de sessão para contexto conversacional

## Conceitos aplicados

| Conceito | Implementação |
|---|---|
| Conexão com LLM | `ChatOpenAI` com configuração de modelo e temperatura |
| Prompt Engineering | `ChatPromptTemplate` com persona de sistema |
| Composição LCEL | `prompt \| modelo \| parser` |
| Memória de sessão | `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` |
| Gerenciamento de secrets | `.env` + `python-dotenv` + `.gitignore` |

## Licença

MIT
