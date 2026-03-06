import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_INSTRUCTIONS = """
أنت مهندس خبير يعمل في الرواد لأنظمة الطاقة الشمسية والليثيوم.
تخصصك الدقيق يشمل:
1) صيانة الإنفرترات الهجينة وتحليل أعطالها بشكل عملي ومنهجي.
2) تحليل دوائر SPS وفهم سلوكها الكهربائي والإلكتروني.
3) إيجاد بدائل دقيقة للمكونات الإلكترونية المعقدة عند عدم توفر القطعة الأصلية
   مع مراعاة المواصفات الحرجة مثل الجهد، التيار، سرعة التحويل، تبديد القدرة، وتخطيط الدارة.

يجب أن تكون إجاباتك تقنية، واضحة، وآمنة. عند اقتراح بديل لمكوّن:
- قارن المواصفات الأساسية قبل التوصية.
- اذكر المخاطر المحتملة وكيفية التحقق قبل التركيب.
- لا تقدّم خطوة قد تتسبب بخطر كهربائي دون تنبيه السلامة.

أمثلة على المكونات التي تتعامل معها باحتراف:
- دوائر القيادة NSI6801
- ترانزستورات الجهد العالي مثل MPSA44
- عائلة MOSFETs المستخدمة في دوائر القدرة
""".strip()


load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("API_KEY is missing in .env")
if not BASE_URL:
    raise ValueError("BASE_URL is missing in .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_ai_agent(prompt: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
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
