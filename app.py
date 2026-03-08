import streamlit as st

from ai_agent import ask_ai_agent
from rag_system import sync_dropbox_files, create_or_update_vector_database, VECTOR_INDEX_PATH, VECTOR_META_PATH


st.set_page_config(page_title="المهندس الآلي - الرواد", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&display=swap');

/* Professional Dark Theme & RTL support */
.stApp, .stApp > header {
    background-color: #0E1117 !important;
}
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
* {
    font-family: 'Cairo', sans-serif !important;
    color: #f8f9fa;
}
body, .stChatMessage {
    direction: RTL;
    text-align: right;
}
p, div, span, li, h1, h2, h3, h4, h5, h6 {
    text-align: right;
}

/* Sidebar Overrides */
[data-testid="stSidebar"] {
    background-color: #0B0E14 !important; /* Matches main background but slightly darker for depth */
    border-left: 1px solid #1E2631;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #0d47a1;
    color: white;
    border-radius: 8px;
    border: 1px solid #1565c0;
    transition: all 0.3s ease;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1976d2;
    transform: scale(1.02);
}

/* Chat Bubbles Overrides */
/* User Bubble - Right / Royal Blue */
[data-testid="stChatMessage"]:has(.user-msg) {
    background-color: #0d3b82 !important; /* Royal Dark Blue */
    border-radius: 20px 20px 0px 20px !important;
    border: 1px solid #1555a6;
    padding: 15px;
    margin-right: 0;
    margin-left: auto;
    width: 80%;
}
/* Assistant Bubble - Left / Charcoal Gray */
[data-testid="stChatMessage"]:has(.assistant-msg) {
    background-color: #1A202C !important; /* Charcoal */
    border-radius: 20px 20px 20px 0px !important;
    border: 1px solid #2D3748;
    padding: 15px;
    flex-direction: row-reverse; /* Avatar on the left in RTL */
    margin-right: auto;
    margin-left: 0;
    width: 90%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

/* Pill-Shaped Input */
[data-testid="stChatInputContainer"] {
    border-radius: 30px !important;
    background-color: #1E2631 !important;
    border: 1px solid #3b82f6 !important;
    padding: 5px 20px !important;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
}
[data-testid="stChatInputContainer"]:focus-within {
    border: 1px solid #60A5FA !important;
    box-shadow: 0 6px 15px rgba(59, 130, 246, 0.2);
}
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea {
    color: #f8f9fa !important;
    caret-color: #f8f9fa !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #a0aec0 !important;
}

/* Suggested Chips */
section.main [data-testid="stButton"] button {
    border-radius: 20px !important;
    background-color: #161D27 !important;
    border: 1px solid #3b82f6 !important; /* Light blue border */
    color: #93c5fd !important;
    font-weight: 600;
    transition: all 0.2s ease;
}
section.main [data-testid="stButton"] button:hover {
    background-color: #1e3a8a !important; /* Darker blue on hover */
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Misc Styles */
.source-card {
    background-color: rgba(30, 58, 138, 0.3);
    border-right: 4px solid #3b82f6;
    padding: 12px 15px;
    border-radius: 10px;
    margin-top: 15px;
    color: #93c5fd;
    font-size: 0.95rem;
    font-weight: bold;
}
.stat-box {
    background-color: #1E2631;
    border-right: 4px solid #4da3ff;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: right;
}
.stat-title {
    color: #a0aec0;
    font-size: 0.9rem;
    margin-bottom: 5px;
}
.stat-value {
    color: #f8f9fa;
    font-size: 1.4rem;
    font-weight: bold;
}
.welcome-container {
    text-align: center;
    margin-top: 15vh;
    margin-bottom: 5vh;
}
.welcome-icon {
    font-size: 80px;
    margin-bottom: 15px;
}
.welcome-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8f9fa;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("إعدادات النظام الذكي")
    if st.button("مزامنة وتحديث الملفات 🔄"):
        with st.spinner("جاري المزامنة مع دروبكس وبناء الكشاف..."):
            updated_files = sync_dropbox_files()
            pdf_count, total_chars = create_or_update_vector_database(
                "knowledge_base", 
                VECTOR_INDEX_PATH, 
                VECTOR_META_PATH,
                new_files=updated_files
            )
            
            st.session_state["pdf_count"] = pdf_count
            st.session_state["total_chars"] = total_chars
            st.success("تم التحديث بنجاح!")

    st.markdown("---")
    st.markdown("### 💬 سجل المحادثات")
    if "messages" not in st.session_state:
        st.session_state.messages = []
         
    user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if not user_msgs:
        st.caption("لا توجد استفسارات سابقة.")
    else:
        for msg in user_msgs[::-1][:5]: 
            st.caption(f"▪ {msg[:35]}{'...' if len(msg)>35 else ''}")

    st.markdown("---")
    st.markdown("### 📊 إحصائيات المعرفة")
    
    # Render stats safely from session state to persist after refreshes/loads
    p_cnt = st.session_state.get("pdf_count", 0)
    c_cnt = st.session_state.get("total_chars", 0)
    
    if c_cnt > 0:
        char_display = f"{c_cnt / 1_000_000:.2f}"
    else:
        char_display = "0"
        
    st.markdown(f"""
        <div class="stat-box">
            <div class="stat-title">عدد الملفات المفهرسة</div>
            <div class="stat-value">{p_cnt} ملفاً</div>
        </div>
        <div class="stat-box">
            <div class="stat-title">حجم المعرفة التقنية</div>
            <div class="stat-value">{char_display} مليون حرف</div>
        </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    # Welcome Screen / Empty State
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">كيف يمكن لمهندس الرواد مساعدتك اليوم؟</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("فحص كود F09", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "فحص كود F09"})
            st.rerun()
    with col2:
        if st.button("معايرة إنفرتر Axpert", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "طريقة معايرة إنفرتر Axpert؟"})
            st.rerun()
    with col3:
        if st.button("بيانات بطارية الليثيوم", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "ما هي بيانات بطارية الليثيوم؟"})
            st.rerun()

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown('<div class="user-msg"></div>', unsafe_allow_html=True)
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown('<div class="assistant-msg"></div>', unsafe_allow_html=True)
            if "المصدر:" in message["content"]:
                parts = message["content"].split("المصدر:", 1)
                st.markdown(parts[0].strip())
                st.markdown(f'<div class="source-card">📘 المصدر: {parts[1].strip()}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

user_prompt = st.chat_input("اكتب سؤالك الهندسي هنا...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.rerun()

# Process new request if the last message was from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown('<div class="assistant-msg"></div>', unsafe_allow_html=True)
        with st.spinner("جاري التحليل الهندسي..."):
            assistant_reply = ask_ai_agent(last_prompt)
            
        if "المصدر:" in assistant_reply:
            parts = assistant_reply.split("المصدر:", 1)
            st.markdown(parts[0].strip())
            st.markdown(f'<div class="source-card">📘 المصدر: {parts[1].strip()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(assistant_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

