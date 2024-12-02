from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import runnable
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the openai language model
openai = OpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    openai_api_key=api_key)

template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: {text}",
)

# @runnable
chain = template | openai


def summarize(text):    
    """
    Summarizes the given text using a language model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    return chain.invoke({"text":text})  

if __name__ == "__main__":
    print(summarize("The quick brown fox jumps over the lazy dog."))