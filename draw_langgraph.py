import sys
from pathlib import Path

def draw_graph():
    # Thêm src vào sys.path để có thể import rag_researcher
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        
    from rag_researcher.agent.workflow import build_agentic_rag_graph

    # Lấy đồ thị đã được compile
    graph = build_agentic_rag_graph()
    
    # Trích xuất và vẽ biểu đồ Mermaid dưới dạng PNG
    # Lưu ý: hàm draw_mermaid_png() gọi một API miễn phí trên mạng để render
    png_data = graph.get_graph().draw_mermaid_png()
    
    # Đường dẫn xuất file
    output_path = Path(__file__).parent / "langgraph_workflow.png"
    
    # Ghi file
    with open(output_path, "wb") as f:
        f.write(png_data)
        
    print(f"✅ Đã xuất đồ thị LangGraph ra file: {output_path.resolve()}")

if __name__ == "__main__":
    draw_graph()
