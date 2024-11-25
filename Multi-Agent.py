import os
from autogen import ConversableAgent
from flaml import autogen
import absl.logging
from datetime import date, timedelta
import sqlite3
import yfinance as yf
import pandas as pd
import sqlite3
import sqlite3
import pandas as pd

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
def create_table():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("./stock_data.db")
    cursor = conn.cursor()

    # Create the table using the schema
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

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Function to fetch stock data and save to Excel and SQLite
def get_stock_data(ticker, start_date, end_date, filename="stock_data.xlsx"):
    # Fetch historical market data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Remove timezone information from the Date index
    stock_data.index = stock_data.index.tz_localize(None)

    # Save to Excel (you can modify this to save with different filenames if desired)
    stock_data.to_excel(filename)

    # Insert data into SQLite database
    conn = sqlite3.connect("./stock_data.db")
    cursor = conn.cursor()

    # Prepare SQL insert statement
    for date, row in stock_data.iterrows():
        # Convert date to string format
        date_str = date.strftime('%Y-%m-%d')  # Format the date to YYYY-MM-DD

        # Access the row values using the MultiIndex structure
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

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    # Return the DataFrame for further use or display
    return stock_data


def fetch_data(schema, question):
    # Connect to the SQLite database
    create_table()
    #Database has default data to tesla and apple  for current use case
    get_stock_data('TSLA', '2024-11-20', '2024-11-26')
    get_stock_data('AAPL', '2024-11-20', '2024-11-26')
    conn = sqlite3.connect("./stock_data.db")
    # question="Write a SQL query to select All Stocks of AAPL"
    #question = "What was the difference between last open and close value for Tesla"
    print(question)

    query = generate_sql_query(schema, user_question)
    print("\nGenerated SQL Query:")
    print(query)
    # Fetch the data into a DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    # Display the fetched data
    print("Data retrieved from the database:")
    #print(df)
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
    df=fetch_data(schema, user_question)
    #print("\nGenerated SQL Query:")
    print(df)

    # Example for LLM2 - RAG Placeholder
    print("\nRAG LLM2 Output:")
    print(llm2_display_url())

    # Example for LLM3 - Web Search
    web_search_question = "What are the latest trends in the stock market?"
    web_search_response = execute_web_search(web_search_question)
    print("\nWeb Search Agent Response:")
    print(web_search_response)