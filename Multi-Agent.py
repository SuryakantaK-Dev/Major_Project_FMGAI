import os
from autogen import ConversableAgent
from flaml import autogen
import absl.logging
from datetime import date, timedelta
import sqlite3
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from duckduckgo import extract_news

# Suppress unnecessary logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Load the .env file
load_dotenv()

# Configuration for AutoGen
config_list_gemini = autogen.config_list_from_json("model_config.json")
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0.8, "seed": 1234}

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
    # Strip the code blocks from the response
    sql_query = response['content'].strip('```sql\n')
    return sql_query

# LLM2 - RAG Placeholder
def llm2_display_url():
    """
    Displays the URL link for LLM2.
    """
    url = "http://192.168.0.169:8501"
    return f"For Use Of RAG And PDF Chat is available at {url}"

# LLM3 - Web Search
def execute_web_search_llm3(problem):
    """
    Executes the Web Search using DuckDuckGo for the given problem.
    """
    print("\nExecuting Web Search LLM - 3...")
    extractor = extract_news(text=problem)
    extractor.extract_entity()
    news = extractor.call_duckduckgo()
    return news

# SQLite Table Creation
def create_table():
    conn = sqlite3.connect("./stock_data.db")
    cursor = conn.cursor()
    cursor.execute('''
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
        ''')
    conn.commit()
    conn.close()

# Fetch Stock Data
def get_stock_data(ticker, start_date, end_date, filename="stock_data.xlsx"):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.to_excel(filename)
    conn = sqlite3.connect("./stock_data.db")
    cursor = conn.cursor()
    for date, row in stock_data.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT OR REPLACE INTO stock_data (date, ticker, adj_close, close, high, low, open, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date_str,
            ticker,
            row[('Adj Close', ticker)],
            row[('Close', ticker)],
            row[('High', ticker)],
            row[('Low', ticker)],
            row[('Open', ticker)],
            row[('Volume', ticker)]
        ))
    conn.commit()
    conn.close()
    return stock_data

def fetch_data(schema, question):
    create_table()
    get_stock_data('TSLA', '2024-11-20', '2024-11-26')
    get_stock_data('AAPL', '2024-11-20', '2024-11-26')
    conn = sqlite3.connect("./stock_data.db")
    print(question)
    query = generate_sql_query(schema, question)
    print("\nGenerated SQL Query:")
    print(query)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Data retrieved from the database:")
    return df

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
    user_question = "What was the last open and close value for TSLA?"
    df = fetch_data(schema, user_question)
    print(df)

    # Example for LLM2 - RAG Placeholder
    print("\nRAG LLM2 Output:")
    print(llm2_display_url())

    # Example for LLM3 - Web Search
    problem = "Can I invest in the Zomato stocks?"
    web_search_news = execute_web_search_llm3(problem)
    print("\nWeb Search LLM-3 Output:")
    print(web_search_news)
