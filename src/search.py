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
    model=os.getenv("LLM_MODEL"),
    # temperature=0
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

def _generate_prompt_template():
  return PromptTemplate(
      template=PROMPT_TEMPLATE,
      input_variables=["contexto", "pergunta"]
  )

def _print_results_and_metadata(results):
  for i, (doc, score) in enumerate(results, start=1):
    print("-"*50)
    print(f" Resultado {i} (score: {score:.2f}):")
    print("\n Texto:\n")
    print(doc.page_content.strip())

    print("\n Metadados:\n")
    for k, v in doc.metadata.items():
        print(f" {k}: {v}")
  print("\n\n")

def _get_results_from_database(user_input: str):
  print("\n  Iniciando busca...")
  embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
  )

  print("  Conectando ao banco de dados...")
  store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("DATABASE_URL"),
      use_jsonb=True,
  )

  print("  Buscando resultados...")
  results = store.similarity_search_with_score(user_input, k=10)
  return results

def search_prompt(user_input: str):
  results = _get_results_from_database(user_input)
  if not results:
    print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
    return 
  # _print_results_and_metadata(results)

  print("  Gerando prompt template e enviando ao LLM...\n")
  chain = _generate_prompt_template() | gemini_llm
  result = chain.invoke({"contexto": results, "pergunta": user_input})
  
  return result