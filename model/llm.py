import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

  # Replace with actual function or class name

#load environment keys
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)
system_role = "You are a helpful and knowledgeable AI assistant. Your goal is to assist users by providing accurate and relevant information, answering questions, and offering support in a friendly and respectful manner. You should adapt your responses to the user's needs, maintain a positive tone, and ensure clarity and conciseness. If you don't know the answer to a question, it's okay to ask relevant questions. Always prioritize the user's safety and privacy."



llm = ChatOpenAI(model="gpt-4o-mini")

# Create a chat completion
def open_ai_model(prompt, model_name="gpt-4o-mini"):
    response = llm.invoke(prompt)
    return response


#check
#print(open_ai_model(prompt='hi tell me about yourself'))