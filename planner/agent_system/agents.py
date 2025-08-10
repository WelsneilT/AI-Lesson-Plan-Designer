import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from .state import TeacherState

load_dotenv()

def get_llm():
    try:
        llm = ChatGroq(
            temperature=0.1, model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"), max_tokens=2048
        )
        return llm
    except Exception as e:
        print(f"LỖI KHỞI TẠO LLM: {e}")
        return None

llm = get_llm()

class ParsedObjective(BaseModel):
    action_verb: str = Field(description="Động từ hành động chính, ví dụ: 'phân tích', 'trình bày'.")
    bloom_level: int = Field(description="Cấp độ tư duy theo thang Bloom (1-6).")
    topic: str = Field(description="Chủ đề chính của bài học.")
    grade_level: str = Field(description="Cấp lớp của học sinh, ví dụ: 'Lớp 9'.")

async def objective_interpreter_agent(state: TeacherState) -> Dict[str, Any]:
    print("\n--- [Agent: Objective Interpreter] ---")
    user_request = state['messages'][-1].content
    
    prompt = f"""Phân tích yêu cầu của giáo viên sau và trích xuất mục tiêu học tập cốt lõi.
    THANG ĐO BLOOM: 1-Nhớ, 2-Hiểu, 3-Vận dụng, 4-Phân tích, 5-Đánh giá, 6-Sáng tạo.
    YÊU CẦU: "{user_request}" """

    structured_llm = llm.with_structured_output(ParsedObjective)
    
    try:
        parsed_result = await structured_llm.ainvoke(prompt)
        analyzed_objective_dict = {
            "action_verb": parsed_result.action_verb,
            "bloom_level": parsed_result.bloom_level,
            "topic": parsed_result.topic,
            "grade_level": parsed_result.grade_level,
            "constraints": []
        }
        print(f"Phân tích thành công: {parsed_result.dict()}")
        return {"analyzed_objective": analyzed_objective_dict}
    except Exception as e:
        print(f"Lỗi: {e}")
        return {"analyzed_objective": None}