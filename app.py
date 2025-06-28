import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss 

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}")
    return text

def generate_embeddings(text_chunks):
    if not text_chunks:
        return np.array([])
    embeddings = EMBEDDING_MODEL.encode(text_chunks, show_progress_bar=False)
    return embeddings

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def get_most_relevant_text(query_embedding, index, text_chunks, k=1):
    if index is None:
        return []
    
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    
    relevant_chunks = []
    for i in I[0]:
        if i < len(text_chunks): 
            relevant_chunks.append(text_chunks[i])
    return relevant_chunks

def simple_llm_response(query, relevant_context):
    if not relevant_context:
        return "Não consegui encontrar informações relevantes para a sua pergunta nos documentos."
    
    context_str = "\n".join(relevant_context)
    
    response = f"Com base nos documentos, sobre '{query}', encontrei as seguintes informações:\n\n{context_str}\n\n" \
               f""
    return response

st.set_page_config(page_title="Tarefa AS05", layout="centered")

st.title("📚 Tarefa AS05 - Pergunte sobre um PDF!")
st.markdown("Faça upload de seus PDFs e faça perguntas sobre o conteúdo deles.")

if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
    st.session_state.text_chunks = []
    st.session_state.embeddings = None
    st.session_state.faiss_index = None

uploaded_files = st.file_uploader("Selecione um ou mais arquivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files and not st.session_state.documents_processed:
    all_extracted_text = ""
    with st.spinner("Processando PDFs... Isso pode levar um momento."):
        for uploaded_file in uploaded_files:
            all_extracted_text += extract_text_from_pdf(uploaded_file)
        
        st.session_state.text_chunks = [chunk.strip() for chunk in all_extracted_text.split('\n\n') if chunk.strip()]
        
        if st.session_state.text_chunks:
            st.session_state.embeddings = generate_embeddings(st.session_state.text_chunks)
            st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
            st.session_state.documents_processed = True
            st.success(f"✔️ {len(uploaded_files)} Pronto! Agora você pode fazer perguntas.")
        else:
            st.warning("Não foi possível extrair texto dos PDFs selecionados.")

if st.session_state.documents_processed:
    st.markdown("---")
    st.subheader("Faça sua pergunta:")
    query = st.text_input("Digite sua pergunta aqui:")

    if query:
        if st.session_state.faiss_index is not None:
            query_embedding = EMBEDDING_MODEL.encode([query])[0]
            relevant_context = get_most_relevant_text(query_embedding, st.session_state.faiss_index, st.session_state.text_chunks, k=3)
            
            response = simple_llm_response(query, relevant_context)
            st.markdown(f"**Resposta:**\n{response}")
        else:
            st.warning("Faça o upload e processe os PDFs primeiro.")
else:
    st.info("Aguardando upload do PDF.")

st.markdown("---")
st.caption("Tarefa AS05 - Yuri")
