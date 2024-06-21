from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import boto3
import streamlit as st

## Bedrock Clients
bedrock_client=boto3.client(service_name="bedrock-runtime",region_name="ap-south-1")
embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-image-v1",client=bedrock_client)


## Data ingestion
def data_ingestion():
    loader=PyPDFLoader("06._Volume_II_Section_VI_Conditions_of_Contract_Particular_Conditions.pdf")
    documents=loader.load()

    chunk_gen=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs_arr=chunk_gen.split_documents(documents)
    return docs_arr

## Vector Embedding and vector store

def vector_store_gen(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def call_mistral_llm():
    ##create the Mistral Model
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock_client,
                model_kwargs={'maxTokens':512})
    
    return llm

def call_llama3_llm():
    ##create the Llama Model
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock_client,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the pieces of context to generate a 
concise answer to the question at the end with atleast 250 words. If unable to answer, say so instead of hallucinating an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def call_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF")

    user_question = st.text_input("Query the contract")

    with st.sidebar:
        st.title("Update/Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vector_store_gen(docs)
                st.success("Done")

    if st.button("Mistral Output"):
        with st.spinner("Generating..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
            llm=call_mistral_llm()
            
            st.write(call_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Generating..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
            llm=call_llama3_llm()
            
            st.write(call_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()