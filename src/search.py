import os
import sys
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()
for k in ("GOOGLE_API_KEY", "DATABASE_URL", "LLM_MODEL", "EMBEDDING_MODEL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

gemini_llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(user_input: str):
  print("Iniciando busca...")
  embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL","models/embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
  )

  print("Conectando ao banco de dados...")
  store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("DATABASE_URL"),
      use_jsonb=True,
  )

  print("Buscando resultados...")
  results = store.similarity_search_with_score(user_input, k=10)

  if not results:
    print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
    return 
  
  template = PromptTemplate(
      input_variables=["name"],
      template=PROMPT_TEMPLATE
  )

  formatedPrompt = template.format(pergunta=user_input, contexto="\n".join([doc.page_content for doc, _ in results]))
  print("\n" + formatedPrompt)

  answer_gemini = gemini_llm.invoke(formatedPrompt)
  return answer_gemini