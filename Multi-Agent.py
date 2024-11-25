import os
from autogen import ConversableAgent
from flaml import autogen
import absl.logging

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
    user_question = "What was the last open and close value for Tesla?"
    sql_query = generate_sql_query(schema, user_question)
    print("\nGenerated SQL Query:")
    print(sql_query)

    # Example for LLM2 - RAG Placeholder
    print("\nRAG LLM2 Output:")
    print(llm2_display_url())

    # Example for LLM3 - Web Search
    web_search_question = "What are the latest trends in the stock market?"
    web_search_response = execute_web_search(web_search_question)
    print("\nWeb Search Agent Response:")
    print(web_search_response)
