import os
import gradio as gr

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "D:/My Projects/GenAI Document Intelligence (RAG)"
FAISS_PATH = "faiss_index"


# =====================================================
# Load Documents (same as notebook)
# =====================================================
documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".docx"):
        loader = Docx2txtLoader(os.path.join(DATA_PATH, file))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file
        documents.extend(docs)

if not documents:
    raise RuntimeError("No DOCX files found in DATA_PATH")

# =====================================================
# Chunking
# =====================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)


# =====================================================
# Embeddings
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =====================================================
# Vector Store
# =====================================================
if os.path.exists(FAISS_PATH):
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_PATH)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})


# =====================================================
# LLM (Offline)
# =====================================================
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)


# =====================================================
# Prompt
# =====================================================
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)


# =====================================================
# RAG Chain (Runnable)
# =====================================================
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =====================================================
# Inference Function (SAFE)
# =====================================================
def ask_question(question: str):
    question = str(question).strip()

    if not question:
        return "Please enter a question.", ""

    try:
        answer = rag_chain.invoke(question)

        docs = retriever.invoke(question)
        sources = sorted({d.metadata.get("source", "Unknown") for d in docs})

        sources_text = "\n".join(f"- {s}" for s in sources)

        return answer, sources_text

    except Exception as e:
        return f"Error during inference: {str(e)}", ""


# =====================================================
# Gradio UI (STABLE)
# =====================================================
with gr.Blocks(title="GenAI Document Intelligence (RAG)") as demo:

    gr.Markdown(
        """
        # 📄 GenAI Document Intelligence (RAG)
        Ask financial and business questions from Microsoft FY2022 documents  
        using an **offline Retrieval-Augmented Generation system**.
        """
    )

    question_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g. How did cloud services perform in FY2022?",
        lines=2
    )

    ask_btn = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    sources_output = gr.Textbox(
        label="Sources",
        lines=4
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

demo.launch()
