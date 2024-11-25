import os
import re
from typing import List
from autogen import ConversableAgent
from flaml import autogen
from pypdf import PdfReader
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import absl.logging
import google.generativeai as genai
import json

# Suppress unnecessary logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Configuration for AutoGen
config_list_gemini = autogen.config_list_from_json("model_config.json")

# Helper function to retrieve API key from the configuration
def get_api_key(config_list, model_name):
    for config in config_list:
        if config["model"] == model_name:
            return config["api_key"]
    raise ValueError(f"API key for model '{model_name}' not found in the configuration.")

# LLM1 - Text2SQL
text2sql_agent = ConversableAgent(
    name="Text2SQL Agent",
    system_message="You are a Text2SQL generator. Given a schema and question, generate an SQL query that directly answers the question.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

def generate_sql_query(schema, user_question):
    """
    Generates an SQL query based on the provided schema and question.
    """
    print("Generating SQL query...")
    response = text2sql_agent.generate_reply(
        messages=[{"content": f"{schema}\nQuestion: {user_question}", "role": "user"}]
    )
    return response['content']

# LLM2 - RAG
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, input: Documents) -> Embeddings:
        genai.configure(api_key=self.api_key)
        model = "models/embedding-001"
        title = "Embedding generation for retrieval"
        embeddings = []
        for content in input:
            response = genai.embed_content(
                model=model, content=content, task_type="retrieval_document", title=title
            )
            embeddings.append(response["embedding"])
        return embeddings

def create_chroma_db(documents: List[str], path: str, name: str, api_key: str):
    chroma_client = chromadb.PersistentClient(path=path)
    embedding_function = GeminiEmbeddingFunction(api_key=api_key)
    try:
        db = chroma_client.get_collection(name=name, embedding_function=embedding_function)
        print(f"Collection '{name}' already exists. Loading the existing collection.")
    except chromadb.errors.CollectionNotFoundError:
        print(f"Creating new collection '{name}'.")
        db = chroma_client.create_collection(name=name, embedding_function=embedding_function)
        db.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    return db

def generate_answer(db, query: str, api_key: str) -> str:
    relevant_text = db.query(query_texts=[query], n_results=3)["documents"][0]
    prompt = f"""
    You are a helpful and informative bot. Use the text from the reference passage below to answer the question.
    Be comprehensive and explain in simple terms.

    QUESTION: {query}
    PASSAGE: {" ".join(relevant_text)}

    ANSWER:
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# LLM3 - Web Search
web_search_agent = ConversableAgent(
    name="Web Search Agent",
    system_message="You are a web search agent. Search for the information requested and provide accurate responses.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

def execute_web_search(question):
    """
    Executes the Web Search Agent to find additional information related to the question.
    """
    print("Executing Web Search Agent...")
    response = web_search_agent.generate_reply(
        messages=[{"content": question, "role": "user"}]
    )
    return response['content']

# Main Execution
if __name__ == "__main__":
    # Retrieve the API key for Gemini models
    api_key = get_api_key(config_list_gemini, "gemini-1.5-pro")

    # Example for LLM1 - Text2SQL
    schema = """
    CREATE TABLE IF NOT EXISTS stock_data (
     date DATE,
    ticker VARCHAR(10),
    adj_close DECIMAL(10, 6),
    close DECIMAL(10, 6),
    high DECIMAL(10, 6),
    low DECIMAL(10, 6),
    open DECIMAL(10, 6),
    volume BIGINT,
    PRIMARY KEY (date, ticker)
    );
    """
    user_question = "Generate an SQL query to What was the last open and close value for Tesla"
    sql_query = generate_sql_query(schema, user_question)
    print("\nGenerated SQL Query:")
    print(sql_query)

    # Example for LLM2 - RAG
    pdf_file_path = "C:\\Users\\Wissen\\PycharmProjects\\pythonProject1\\PDF\\sample_report.pdf"  # Replace with your PDF file path
    pdf_text = PdfReader(pdf_file_path).pages[0].extract_text()
    documents = re.split(r"\n \n", pdf_text)
    documents = [doc.strip() for doc in documents if doc.strip()]
    db_path = "C:\\Repos\\RAG\\contents"
    collection_name = "rag_experiment"
    db = create_chroma_db(documents, db_path, collection_name, api_key)
    user_query = "Provide me IT Department Report."
    answer = generate_answer(db, user_query, api_key)
    print("\nGenerated Answer:")
    print(answer)

    # Example for LLM3 - Web Search
    web_search_question = "What are the latest trends in Stock Market?"
    web_search_response = execute_web_search(web_search_question)
    print("\nWeb Search Agent Response:")
    print(web_search_response)
