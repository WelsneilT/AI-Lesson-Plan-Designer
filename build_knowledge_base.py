import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tải các biến môi trường (nếu cần)
load_dotenv()

# --- CẤU HÌNH ---
# 1. Đường dẫn Tesseract OCR
#    Hãy thay đổi đường dẫn này cho đúng với máy của bạn
TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Đường dẫn đến file PDF nguồn
PDF_PATH = "data/thuvienhoclieu.com-SGK-Toan-9-KNTT-tap-1.pdf" 

# 3. Nơi lưu trữ kho tri thức sau khi xử lý
VECTOR_STORE_PATH = "vector_store/sgk_toan_9"

# 4. Tên mô hình embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- CÁC HÀM XỬ LÝ ---

def check_tesseract_installed():
    """Kiểm tra xem Tesseract có được cài đặt đúng đường dẫn không."""
    if not os.path.exists(TESSERACT_CMD_PATH):
        print(f"❌ LỖI: Không tìm thấy Tesseract tại '{TESSERACT_CMD_PATH}'.")
        print("Vui lòng cài đặt Tesseract OCR và cập nhật lại đường dẫn TESSERACT_CMD_PATH trong file này.")
        return False
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    return True

def extract_text_with_ocr(pdf_path: str) -> str:
    """Trích xuất văn bản từ PDF dạng ảnh bằng OCR."""
    print(f"--- ⏳ Bắt đầu quá trình OCR cho file: {pdf_path} ---")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300) # Tăng DPI để cải thiện chất lượng nhận dạng
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        
        try:
            # Sử dụng Tesseract để đọc chữ từ ảnh (ngôn ngữ Tiếng Việt)
            text = pytesseract.image_to_string(image, lang='vie')
            full_text += text + "\n\n" # Thêm khoảng trắng giữa các trang
            print(f"      ✅ Đã xử lý xong trang {page_num + 1}/{len(doc)}")
        except Exception as e:
            print(f"      ⚠️ Lỗi khi OCR trang {page_num + 1}: {e}")

    doc.close()
    print("--- ✅ Đã OCR xong toàn bộ file PDF. ---")
    return full_text

def build_and_save_vector_store():
    """Hàm chính để xây dựng và lưu kho tri thức."""
    if not check_tesseract_installed():
        return
        
    if not os.path.exists(PDF_PATH):
        print(f"❌ LỖI: Không tìm thấy file PDF tại '{PDF_PATH}'.")
        return
        
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"✅ Kho tri thức tại '{VECTOR_STORE_PATH}' đã tồn tại. Bỏ qua.")
        return

    # 1. Trích xuất văn bản
    book_content = extract_text_with_ocr(PDF_PATH)
    
    # 2. Tạo đối tượng Document
    documents = [Document(page_content=book_content)]
    
    # 3. Cắt văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        print("❌ LỖI NGHIÊM TRỌNG: Không thể chia tài liệu thành các đoạn. Quá trình OCR có thể đã thất bại.")
        return
    print(f"\n--- splitting Đã chia tài liệu thành {len(docs)} đoạn văn bản. ---")
    
    # 4. Tạo embeddings
    print(f"\n--- 🧠 Đang tải mô hình embedding: {EMBEDDING_MODEL}... ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 5. Xây dựng và lưu vector store
    print("\n--- 🏗️ Bắt đầu xây dựng và lưu trữ kho tri thức FAISS... ---")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    
    print(f"\n🎉 THÀNH CÔNG! Kho tri thức đã được xây dựng và lưu tại: '{VECTOR_STORE_PATH}'")

# --- ĐIỂM THỰC THI CHÍNH ---
if __name__ == '__main__':
    build_and_save_vector_store()