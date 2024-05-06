from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage

load_dotenv(find_dotenv())

loader = TextLoader("C:/Users/TAMIM M/Downloads/create-chatbot-html-css-javascript/Chatbot/smvecdataset.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=500)
texts = text_splitter.split_documents(documents)

def get_response(question):
    print(question)

    embeddings = OpenAIEmbeddings()
    retriever = Chroma.from_documents(texts, embeddings).as_retriever()
    chat = ChatOpenAI(temperature=0)

    prompt_template = """You are a helpful dicord bot that helps users to know a college named Sri Mankula Vinayagar Engineering College and you must also work as a ChatGPT-clone for common questions.

    {context}

    Please provide the most suitable response for the users question and if there is a link associated with the question display it. change https links to actual working link
    Answer:"""

    prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    try:
        docs = retriever.get_relevant_documents(query=question)
        formatted_prompt = system_message_prompt.format(context=docs)

        messages = [formatted_prompt, HumanMessage(content=question)]
        result = chat(messages)
        print(result.content)
        return(result.content)
    except Exception as e:
        print(f"Error occurred: {e}")
        return ("Sorry, I was unable to process your question.")

if __name__ == "__main__":
    

    get_response("hello")