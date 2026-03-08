import os
import dropbox
import sys
from dotenv import load_dotenv

# Fix encoding issue for Windows terminal printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

try:
    dbx = dropbox.Dropbox(os.getenv("DROPBOX_ACCESS_TOKEN"))
    dbx.files_upload(b"Hello Alrowad AI!", "/hello.txt", mode=dropbox.files.WriteMode.overwrite)
    print("🎉 اكتملت العملية بنجاح! المجلد موجود الآن في دروبكس.")
except Exception as e:
    print("حدث خطأ:", e)
