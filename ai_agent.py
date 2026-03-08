import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAI

from rag_system import similarity_search, KNOWLEDGE_BASE_DIR


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

KNOWLEDGE_BASE_PATH = KNOWLEDGE_BASE_DIR


load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY is missing in .env")
if not BASE_URL:
    raise ValueError("BASE_URL is missing in .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_ai_agent(prompt: str) -> str:
    greetings = ["سلام", "مرحبا", "أهلا", "أهلاً", "من أنت", "السلام عليكم", "هلا", "مرحباً"]
    if any(greet in prompt.lower() for greet in greetings) and len(prompt) < 30:
        return "أهلاً بك! أنا المهندس الآلي لمركز الرواد - سيئون. كيف يمكنني مساعدتك في أنظمة الطاقة الشمسية والليثيوم اليوم؟"

    context_chunks = similarity_search(prompt, k=8, knowledge_folder=KNOWLEDGE_BASE_PATH)

    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"{RAG_INSTRUCTION}\n\n"
            f"السياق:\n{context_text}\n\n"
            f"السؤال:\n{prompt}"
        )
    else:
        user_message = prompt

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
