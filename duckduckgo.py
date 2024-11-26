from duckduckgo_search import DDGS
import pandas as pd
from together import Together
import os
from dotenv import load_dotenv
load_dotenv()

# def news(
#     keywords: str,
#     region: str = "wt-wt",
#     safesearch: str = "moderate",
#     timelimit: str = None,
#     max_results: str = None) -> list[dict[str, str]]:
#     """DuckDuckGo news search. Query params: https://duckduckgo.com/params.
    
#     Args:
#         keywords: keywords for query.
#         region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
#         safesearch: on, moderate, off. Defaults to "moderate".
#         timelimit: d, w, m. Defaults to None.
#         max_results: max number of results. If None, returns results only from the first response. Defaults to None.
    
#     Returns:
#         List of dictionaries with news search results.
#     """

class extract_news:
    
    def __init__(self,
        text: str
        ) -> None:
    
        super().__init__()
        self.text = text
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.prompt = f"""
                        From the below content only extract the company entity name
                        ```{text}```
                        Instructions: Don't response anything else, Just the company entity name
                        """
    
    def extract_entity(self):
        response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": self.prompt}],
                model="meta-llama/Llama-3-8b-chat-hf",
                temperature=0, # this is the degree of randomness of the model's output
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
              )
        self.response = response.choices[0].message.content

    def call_duckduckgo(self):
        results = DDGS().news(self.response,
                            region="wt-wt",
                            safesearch = "moderate",
                            timelimit = '1d',
                            max_results=5)

        df = pd.DataFrame(results)

        df['news'] = df['title'] + ' ' + df['body'] 
        #print(df)
        news = " ".join(df['news'])
        return news
        #df.to_csv('data/news.csv',index=False)
        # file_path = "data/news.md"
        # with open(file_path, "w") as file:
        #     file.write(news)
