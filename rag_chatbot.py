from io import BytesIO

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "qwen2.5:3b"


def read_pdf(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(BytesIO(uploaded_file.read()))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number
                    }
                )
            )

    return documents


def read_txt(uploaded_file):
    uploaded_file.seek(0)

    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": uploaded_file.name,
                "page": 1
            }
        )
    ]


def load_uploaded_files(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            documents.extend(
                read_pdf(uploaded_file)
            )

        elif filename.endswith(".txt"):
            documents.extend(
                read_txt(uploaded_file)
            )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_documents(documents)


def build_vector_store(documents):
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No readable text was found in the uploaded files."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, chunks


def answer_question(
    vector_store,
    question,
    chat_history=None,
    top_k=4
):
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": top_k
        }
    )

    retrieved_docs = retriever.invoke(question)

    context_parts = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1
    ):
        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        page = doc.metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
Source {index}
File: {source}
Page: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    history_text = ""

    if chat_history:
        recent_history = chat_history[-6:]

        history_text = "\n".join(
            [
                f"{item['role']}: {item['content']}"
                for item in recent_history
            ]
        )

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    prompt = f"""
You are a financial research assistant.

Answer the user's question using only the retrieved evidence below.

Rules:
1. Do not invent facts.
2. If the evidence is insufficient, clearly say so.
3. Explain financial terms in plain language.
4. When possible, mention the source file and page number.
5. Distinguish between facts stated in the document and your interpretation.
6. Do not provide personalized financial advice.
7. Keep the answer concise but complete.

Recent conversation:
{history_text}

Retrieved evidence:
{context}

User question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs
