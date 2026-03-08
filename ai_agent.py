import os
import sys
import traceback

from openai import OpenAI


SYSTEM_INSTRUCTIONS = """
أنت مهندس صيانة في مركز الرواد. ممنوع إعطاء إجابات عامة.
يجب أن تبحث في السياق (Context) المرفق أولاً. 
إذا وجدت المعلومة، اذكر الخطوات التقنية بالتفصيل (مثل قيم المقاومات، أو أرقام الترانزستورات مثل MPSA44 وعائلة MOSFETs). 
إذا لم تجد الإجابة في السياق، ابحث في خبراتك العامة في أنظمة الطاقة الشمسية والليثيوم، مع التنويه بذلك بوضوح.
يجب أن تكون الردود تقنية وهندسية بحتة باللغة العربية، وتدعم عرض الجداول الفنية إذا وجدت في الكتالوج.
""".strip()

RAG_INSTRUCTION = (
    "أجب على السؤال بناءً على السياق التالي المستخرج من الكتالوجات كأولوية قصوى. "
    "لا تقبل إجابة بدون ذكر المصدر إذا كانت الإجابة من السياق. "
    "بنهاية الإجابة الفنية، يجب إلزامياً كتابة: 'المصدر: [اسم الملف] - صفحة [رقم الصفحة إن وجد]'. "
    "إذا كان السؤال عاماً أو لم تجد الإجابة في السياق، قدم الحل من خبرتك الواسعة في 'أنظمة الطاقة الشمسية والليثيوم' "
    "واختم إجابتك بعبارة: 'بناءً على المعارف العامة للمركز...'."
)


def _get_secret(key, default=None):
    """Read from st.secrets (cloud) first, then os.getenv (local)."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, default)


# Lazy client — only created when ask_ai_agent is called
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = _get_secret("API_KEY")
        base_url = _get_secret("BASE_URL")
        if not api_key or not base_url:
            raise ValueError("API_KEY or BASE_URL is missing from st.secrets / .env")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def ask_ai_agent(prompt: str) -> str:
    # Import rag_system INSIDE the function to prevent circular imports
    from rag_system import similarity_search, KNOWLEDGE_BASE_DIR

    greetings = ["سلام", "مرحبا", "أهلا", "أهلاً", "من أنت", "السلام عليكم", "هلا", "مرحباً"]
    if any(greet in prompt.lower() for greet in greetings) and len(prompt) < 30:
        return "أهلاً بك! أنا المهندس الآلي لمركز الرواد - سيئون. كيف يمكنني مساعدتك في أنظمة الطاقة الشمسية والليثيوم اليوم؟"

    context_chunks = similarity_search(prompt, k=8, knowledge_folder=KNOWLEDGE_BASE_DIR)

    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"{RAG_INSTRUCTION}\n\n"
            f"السياق:\n{context_text}\n\n"
            f"السؤال:\n{prompt}"
        )
    else:
        user_message = prompt

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        timeout=30,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    test_prompt = (
        "ما التشخيص المبدئي لعطل شائع في الإنفرتر الهجين عندما يظهر خطأ انخفاض جهد "
        "البطارية تحت الحمل رغم أن البطارية تبدو مشحونة؟ اذكر خطوات فحص عملية مرتبة."
    )

    try:
        reply = ask_ai_agent(test_prompt)
        print("AI Agent Response:\n")
        print(reply)
    except Exception:
        traceback.print_exc()
