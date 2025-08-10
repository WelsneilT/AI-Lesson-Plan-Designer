import asyncio
import json
from django.shortcuts import render
from langchain_core.messages import HumanMessage, BaseMessage
from .agent_system.graph import get_graph_app

def run_async_in_sync(coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

def make_state_serializable(state: dict):
    """
    Hàm mới: Chuyển đổi các đối tượng không thể serialize trong state thành dạng dict.
    """
    serializable_state = state.copy()
    
    # Xử lý trường 'messages'
    if 'messages' in serializable_state and serializable_state['messages']:
        new_messages = []
        for msg in serializable_state['messages']:
            if isinstance(msg, BaseMessage):
                # Chuyển đối tượng BaseMessage (như HumanMessage) thành dict
                new_messages.append(msg.dict())
            else:
                new_messages.append(msg)
        serializable_state['messages'] = new_messages
        
    return serializable_state

def planner_view(request):
    final_result_str = ""
    if request.method == 'POST':
        user_request = request.POST.get('user_request', '')
        
        if user_request:
            print(f"Nhận yêu cầu từ người dùng: {user_request}")
            app = get_graph_app()
            
            initial_state = {
                "messages": [HumanMessage(content=user_request)]
            }
            
            print(">>> Bắt đầu thực thi Graph AI...")
            final_state = run_async_in_sync(app.ainvoke(initial_state))
            print("<<< Graph AI đã thực thi xong.")

            # *** THAY ĐỔI QUAN TRỌNG Ở ĐÂY ***
            # Dọn dẹp state trước khi dump thành JSON
            serializable_final_state = make_state_serializable(final_state)
            
            # Bây giờ dump state đã được dọn dẹp
            final_result_str = json.dumps(serializable_final_state, indent=2, ensure_ascii=False)
            
            print(f"Trạng thái cuối cùng (đã serialize):\n{final_result_str}")

    return render(request, 'planner/index.html', {'final_result': final_result_str})