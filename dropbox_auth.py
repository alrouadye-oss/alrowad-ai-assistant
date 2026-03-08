import dropbox

APP_KEY = "gru92tmcyl5vr68"
APP_SECRET = "e8xzmxu0uty62kp"

auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
    APP_KEY,
    APP_SECRET,
    token_access_type='offline'
)

authorize_url = auth_flow.start()
print("="*50)
print("1. افتح هذا الرابط في متصفحك:\n" + authorize_url)
print("2. وافق على الصلاحيات (اضغط Allow أو Continue).")
print("3. انسخ الرمز (Authorization Code) الذي سيظهر لك في المتصفح.")
print("="*50)

auth_code = input("ضع الرمز الذي نسخته هنا واضغط Enter: ").strip()

try:
    oauth_result = auth_flow.finish(auth_code)
    print("\n" + "="*50)
    print("🎉 نجاح! تم استخراج الرمز الدائم.")
    print("الرمز الدائم (Refresh Token) الخاص بك هو:")
    print(oauth_result.refresh_token)
    print("="*50)
except Exception as e:
    print('حدث خطأ أثناء الاتصال: %s' % (e,))
