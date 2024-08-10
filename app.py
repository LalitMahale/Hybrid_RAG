from utils import VectorStore, Chain, DocRetrievel,ProcessText,LoadDocument
import streamlit as st
import os
cwd = os.getcwd()
folder_path = os.path.join(cwd, "files")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
st.title(":red[Document] Hybrid :red[Search]")
st.caption("👉🏽 Using Keyword Search👈🏽")


uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file != None:
    doc = uploaded_file.read()
    file_name = os.path.join(folder_path,uploaded_file.name)
    with open(file_name,"wb") as f:
        f.write(doc)

if st.sidebar.button("Process") and uploaded_file != None:
    file_name = os.path.join(folder_path,uploaded_file.name)
    doc_loader = LoadDocument().load_pdf(path = file_name)
    split = ProcessText().document_split(document=doc_loader)
    
