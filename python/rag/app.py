import os
import gradio as gr
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Tuple, Optional
import fitz
from ragtab.pipeline import extract_table

load_dotenv()

API_KEY = os.getenv("RAG_QWEN3_NEXT_80B_A3B_THINKING")
BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = "qwen3-next-80b-a3b-thinking"
EMBEDDING_MODEL = "text-embedding-v3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Dùng OpenAIEmbeddingFunction của ChromaDB – tương thích DashScope
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    api_base=BASE_URL,
    model_name=EMBEDDING_MODEL
)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    embedding_function=embedding_fn
)

# Hàm thêm tài liệu văn bản vào ChromaDB
def add_document(text: str, source: str = "unknown", chunk_size: int = 500):
    """Chia văn bản thành các chunk và thêm vào collection."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return

    ids = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

# Hàm tìm kiếm ngữ cảnh
def search_context(query: str, k: int = 4) -> Tuple[str, List[dict]]:
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    context = "\n\n".join(docs) if docs else ""
    sources = [{"source": m.get("source", "unknown"), "content": d} for d, m in zip(docs, metas)]
    return context, sources

# Hàm chat với LLM (đã sửa để nhận history dạng dict)
def chat_rag(query: str, context: str, history: Optional[List[dict]] = None) -> str:
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "The context may contain OCR errors (e.g., 'c' and 'o' swapped, 'g' and '9' swapped). "
        "Please take that into account and use your knowledge to interpret correctly. "
        "If the answer cannot be found in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append(msg)
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        stream=False
    )
    return response.choices[0].message.content

# Hàm xử lý file ảnh/PDF
def process_image_or_pdf(file_path: str, model_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        all_md = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)
            md, _ = extract_table(img_path, model_path=model_path)
            if md.strip():
                all_md.append(md)
            os.remove(img_path)
        combined_md = "\n\n".join(all_md)
    else:
        md, _ = extract_table(file_path, model_path=model_path)
        combined_md = md

    if not combined_md.strip():
        return "Không tìm thấy bảng."

    num_chunks = add_document(combined_md, source=os.path.basename(file_path))
    return f"Đã thêm {num_chunks} đoạn từ bảng vào RAG.\n\n```markdown\n{combined_md[:1000]}...\n```"

# Giao diện Gradio
with gr.Blocks(title="📚 RAG Qwen Assistant") as demo:
    gr.Markdown("# 🤖 RAG Assistant with Qwen3 + RagTable")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=500)
                msg = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi...")
                clear = gr.Button("Xóa lịch sử")
            with gr.Column(scale=1):
                context_box = gr.HTML(label="Ngữ cảnh liên quan")

        def respond(message, chat_history):
            if not message:
                return "", chat_history, ""
            context, sources = search_context(message)
            answer = chat_rag(message, context, chat_history)
            # Format context box
            if not sources:
                context_html = "*Không tìm thấy tài liệu liên quan.*"
            else:
                html = "<div style='background:#f9f9f9; padding:10px; border-radius:8px;'>"
                for s in sources:
                    html += f"<p><b>📄 {s['source']}:</b><br>{s['content'][:300]}...</p><hr>"
                html += "</div>"
                context_html = html
            # SỬA: Dùng định dạng dict cho mỗi tin nhắn
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer})
            return "", chat_history, context_html

        msg.submit(respond, [msg, chatbot], [msg, chatbot, context_box])
        clear.click(lambda: ([], ""), None, [chatbot, context_box])

    with gr.Tab("Thêm bảng từ ảnh/PDF"):
        file_input = gr.File(label="Tải lên ảnh hoặc PDF")
        model_path_input = gr.Textbox(
            label="Đường dẫn model checkpoint",
            value="/Users/macbookpro14m1pro/Desktop/RagTable/data/table-seg-checkpoints-v2/best_model_v2.pt"
        )
        extract_btn = gr.Button("Trích xuất & thêm vào RAG")
        output_md = gr.Markdown()

        def on_extract(file_obj, model_path):
            if file_obj is None:
                return "Vui lòng chọn file."
            temp_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
            return process_image_or_pdf(temp_path, model_path)

        extract_btn.click(on_extract, [file_input, model_path_input], output_md)

demo.launch(inbrowser=True)