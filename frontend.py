import streamlit as st
import requests
import subprocess
import os
import tempfile

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG_researcher Chat", page_icon="🤖", layout="wide")

st.title("🤖 RAG_researcher Chatbot")
st.markdown("Hỏi đáp trực tiếp với hệ thống Agentic RAG của bạn.")

import pypdf

# --- SIDEBAR: Upload PDF ---
with st.sidebar:
    st.header("📂 Nạp Tài Liệu")
    uploaded_files = st.file_uploader("Kéo thả file (PDF, TXT, MD) vào đây", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Nạp vào cơ sở dữ liệu"):
            with st.spinner("Đang phân tích và nạp tài liệu..."):
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    # Extract text if PDF
                    if uf.name.endswith('.pdf'):
                        pdf_reader = pypdf.PdfReader(uf)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        
                        text_content = text_content.replace("\x00", "")
                        
                        # Save extracted text to a .txt file for ingestion
                        temp_path = os.path.join(temp_dir, uf.name + ".txt")
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                    else:
                        # Save as original extension for txt/md
                        temp_path = os.path.join(temp_dir, uf.name)
                        with open(temp_path, 'wb') as f:
                            f.write(uf.getvalue())
                
                # Run the full ingestion pipeline
                try:
                    with st.status("Đang xử lý tài liệu (Vui lòng đợi...)", expanded=True) as status:
                        st.write(f"⏳ Đang nạp {len(uploaded_files)} tài liệu thô...")
                        res_ingest = subprocess.run(["uv", "run", "rag_researcher-ingest", temp_dir], capture_output=True, text=True, check=True, cwd=os.getcwd())
                        
                        st.write("✂️ Đang chia nhỏ đoạn văn bản (Chunking)...")
                        res_chunk = subprocess.run(["uv", "run", "rag_researcher-chunk"], capture_output=True, text=True, check=True, cwd=os.getcwd())
                        
                        st.write("🧠 Đang mã hóa vector bằng Ollama (Embedding)...")
                        res_embed = subprocess.run(["uv", "run", "rag_researcher-embed"], capture_output=True, text=True, check=True, cwd=os.getcwd())
                        
                        st.write("🗂️ Đang lưu vào máy chủ tìm kiếm (Indexing)...")
                        res_index = subprocess.run(["uv", "run", "rag_researcher-index"], capture_output=True, text=True, check=True, cwd=os.getcwd())
                        
                        status.update(label="Hoàn tất! Tài liệu đã sẵn sàng.", state="complete", expanded=False)
                    st.success(f"Đã nạp thành công {len(uploaded_files)} tài liệu!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Lỗi khi nạp tài liệu (Code {e.returncode}):")
                    st.text_area("Chi tiết lỗi (Stderr)", e.stderr, height=150)
                    st.text_area("Chi tiết lỗi (Stdout)", e.stdout, height=150)
                finally:
                    # Cleanup
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

    st.divider()
    st.header("📊 Đánh Giá & Benchmark")
    
    if st.button("1. Tự động sinh bộ câu hỏi (QA)"):
        with st.spinner("Đang đọc tài liệu và sinh bộ câu hỏi... (Cần vài phút)"):
            try:
                res = subprocess.run(["uv", "run", "rag_researcher-generate-eval-qa"], capture_output=True, text=True, check=True, cwd=os.getcwd())
                st.success("✅ Đã tạo xong bộ câu hỏi `qa_gold.json`!")
            except subprocess.CalledProcessError as e:
                st.error("❌ Lỗi khi sinh câu hỏi!")
                st.text_area("Chi tiết lỗi (Stderr)", e.stderr, height=150)
                
    if st.button("2. Chạy Benchmark"):
        with st.spinner("Đang chấm điểm thuật toán RAG..."):
            try:
                res = subprocess.run(["uv", "run", "rag_researcher-eval", "benchmark"], capture_output=True, text=True, check=True, cwd=os.getcwd())
                st.success("✅ Đánh giá thành công! Bạn có thể xem kết quả bên phải.")
            except subprocess.CalledProcessError as e:
                st.error("❌ Lỗi khi chạy benchmark!")
                st.text_area("Chi tiết lỗi (Stderr)", e.stderr, height=150)

# --- MAIN CHAT INTERFACE ---
report_path = "reports/evaluation/benchmark.md"
if os.path.exists(report_path):
    with st.expander("📊 Xem Báo cáo Kết quả Benchmark mới nhất", expanded=False):
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hỏi tôi về các tài liệu đã nạp..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/agent",
                    json={"query": prompt, "max_rewrite": 3},
                    timeout=1500
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("generation", "Không có câu trả lời.")
                    
                    # Optional: display trace/latency
                    latency = data.get("latency_ms", 0) / 1000
                    guardrail_passed = data.get("guardrail_passed", True)
                    
                    if not guardrail_passed:
                        reason = data.get("guardrail_reason", "unknown")
                        meta_info = f"\n\n*(Câu hỏi bị từ chối bởi bộ lọc: {reason})*"
                    else:
                        docs_count = data.get("documents_count", 0)
                        meta_info = f"\n\n*(Tìm thấy {docs_count} đoạn tài liệu trong {latency:.2f} giây)*"
                    
                    full_response = answer + meta_info
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Lỗi API ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến Backend: {str(e)}")
