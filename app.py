from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import streamlit as st

#import the model 
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#name dataset
names_dataset = [
    "Geetha", "Geeta", "Gita", "Gitu", "Geethu",
    "Rahul", "Rahool", "Raahul", "Rahulh", "Rahal",
    "Anjali", "Anjaly", "Anjalee", "Anjli", "Anjali R",
    "Priya", "Priyaa", "Preeya", "Priyya", "Pryia",
    "Karthik", "Karthick", "Kartheek", "Kartik", "Karthik R",
    "Sneha", "Snehaa", "Snehah", "Snehya", "Snayha",
    "Vikram", "Vikramm", "Vikran", "Vickram", "Vikram S",
    "Pooja", "Puja", "Poojaa", "Pujaa", "Poojah",
    "Amit", "Amith", "Ameet", "Amitt", "Amit K",
    "Suresh", "Soresh", "Suresh K", "Suresh Kumar", "Suresh R"
]

#creating a document for each name
documents=[Document(page_content=i) for i in names_dataset]

#creating a vector store  where we store all the document embeddings
vector_store=Chroma.from_documents(documents,embedding)

#UI Code
st.title("Name Matching System")

st.write(""" Model Trained on these names : ['Geeta','Rahul','Anjali','Priya','Kartik','Pooja','Amit','Suresh','Sneha']""")

user_input=st.text_input("Enter a name")


if user_input:
    result=vector_store.similarity_search_with_score(user_input,k=5)
    st.subheader("Best Match")
    st.write(f"{result[0][0].page_content},\tSimilarity Score : {1-result[0][1]:.2f}")

    st.subheader("Similar Names")
    for doc,score in result:
        st.write(f"{doc.page_content},\tSimilarity Score : {1-score:.2f}")