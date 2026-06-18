
import streamlit as st
import pandas as pd
import joblib
import numpy as np
#
# st.markdown("""
#     <style>
#         body, html {
#             direction: RTL;
#             unicode-bidi: bidi-override;
#             text-align: right;
#         }
#         p, div, input, label, h1, h2, h3, h4, h5, h6 {
#             direction: RTL;
#             unicode-bidi: bidi-override;
#             text-align: right;
#         }
#     </style>
# """, unsafe_allow_html=True)

# --- بارگذاری مدل، scaler و ستون‌های ذخیره شده ---
try:
    model = joblib.load('v1/CUSTOMER_CHURN/models/churn_model.joblib')
    scaler = joblib.load('v1/CUSTOMER_CHURN/models/scaler.joblib')
    model_columns = joblib.load('v1/CUSTOMER_CHURN/models/model_columns.joblib')
except FileNotFoundError:
    st.error("فایل‌های مدل یافت نشدند! لطفاً ابتدا اسکریپت train_model.py یا train_model_advanced.py را اجرا کنید.")
    st.stop()

# --- تعریف دیکشنری برای گزینه‌ها (برای تبدیل فارسی به انگلیسی) ---
# این کار باعث می‌شود ظاهر برنامه فارسی باشد ولی مقادیر انگلیسی به مدل فرستاده شود
yes_no_options = {'بله (Yes)': 'Yes', 'خیر (No)': 'No'}
gender_options = {'مرد (Male)': 'Male', 'زن (Female)': 'Female'}
multiple_lines_options = {'خیر (No)': 'No', 'بله (Yes)': 'Yes',
                          'سرویس تلفن ندارد (No phone service)': 'No phone service'}
internet_service_options = {'DSL': 'DSL', 'فیبر نوری (Fiber optic)': 'Fiber optic', 'ندارد (No)': 'No'}
internet_dependent_options = {'خیر (No)': 'No', 'بله (Yes)': 'Yes',
                              'سرویس اینترنت ندارد (No internet service)': 'No internet service'}
contract_options = {'ماه به ماه (Month-to-month)': 'Month-to-month', 'یک ساله (One year)': 'One year',
                    'دو ساله (Two year)': 'Two year'}
payment_method_options = {
    'چک الکترونیکی (Electronic check)': 'Electronic check',
    'چک پستی (Mailed check)': 'Mailed check',
    'انتقال بانکی (خودکار)': 'Bank transfer (automatic)',
    'کارت اعتباری (خودکار)': 'Credit card (automatic)'
}

# --- ایجاد رابط کاربری وب ---
st.set_page_config(layout="wide")
st.title('🤖 پیش‌بینی احتمال ریزش مشتریان')
st.markdown(
    'این برنامه با دریافت مشخصات یک مشتری، پیش‌بینی می‌کند که او با چه احتمالی اشتراک خود را با شرکت قطع خواهد کرد.')
st.markdown('---')

# --- بخش ورودی اطلاعات ---
st.header('لطفاً به سوالات زیر در مورد مشتری پاسخ دهید:')

# ایجاد ستون برای چیدمان بهتر
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("اطلاعات شخصی")
    gender_fa = st.selectbox('۱. جنسیت مشتری چیست؟', list(gender_options.keys()))
    senior_citizen_fa = st.selectbox('۲. آیا مشتری شهروند سالمند (بالای ۶۵ سال) است؟', list(yes_no_options.keys()))
    partner_fa = st.selectbox('۳. آیا مشتری شریک زندگی (همسر) دارد؟', list(yes_no_options.keys()))
    dependents_fa = st.selectbox('۴. آیا مشتری فرد تحت تکفل (مانند فرزند) دارد؟', list(yes_no_options.keys()))

with col2:
    st.subheader("سرویس‌های اصلی")
    tenure = st.slider('۵. مشتری چند ماه است که اشتراک دارد؟ (Tenure)', 0, 72, 12)
    phone_service_fa = st.selectbox('۶. آیا مشتری سرویس تلفن ثابت دارد؟', list(yes_no_options.keys()))
    multiple_lines_fa = st.selectbox('۷. آیا مشتری بیش از یک خط تلفن دارد؟', list(multiple_lines_options.keys()))
    internet_service_fa = st.selectbox('۸. نوع سرویس اینترنت مشتری چیست؟', list(internet_service_options.keys()))

with col3:
    st.subheader("سرویس‌های جانبی اینترنت")
    online_security_fa = st.selectbox('۹. آیا سرویس "امنیت آنلاین" فعال است؟', list(internet_dependent_options.keys()))
    online_backup_fa = st.selectbox('۱۰. آیا سرویس "پشتیبان‌گیری آنلاین" فعال است؟',
                                    list(internet_dependent_options.keys()))
    device_protection_fa = st.selectbox('۱۱. آیا سرویس "حفاظت از دستگاه" (بیمه) فعال است؟',
                                        list(internet_dependent_options.keys()))
    tech_support_fa = st.selectbox('۱۲. آیا سرویس "پشتیبانی فنی ویژه" فعال است؟',
                                   list(internet_dependent_options.keys()))
    streaming_tv_fa = st.selectbox('۱۳. آیا سرویس "پخش آنلاین تلویزیون" فعال است؟',
                                   list(internet_dependent_options.keys()))
    streaming_movies_fa = st.selectbox('۱۴. آیا سرویس "پخش آنلاین فیلم" فعال است؟',
                                       list(internet_dependent_options.keys()))

st.markdown('---')
st.header('اطلاعات قرارداد و پرداخت')
col4, col5 = st.columns(2)

with col4:
    st.subheader("قرارداد")
    contract_fa = st.selectbox('۱۵. نوع قرارداد مشتری چیست؟', list(contract_options.keys()))
    paperless_billing_fa = st.selectbox('۱۶. آیا مشتری "صورتحساب الکترونیکی" دریافت می‌کند؟',
                                        list(yes_no_options.keys()))
    payment_method_fa = st.selectbox('۱۷. روش پرداخت مشتری چیست؟', list(payment_method_options.keys()))

with col5:
    st.subheader("هزینه‌ها")
    monthly_charges = st.number_input('۱۸. هزینه ماهانه مشتری چقدر است؟ ($)', min_value=0.0, max_value=200.0,
                                      value=70.0, step=0.05)
    total_charges = st.number_input('۱۹. مجموع هزینه‌هایی که تاکنون پرداخت کرده چقدر است؟ ($)', min_value=0.0,
                                    value=float(tenure * monthly_charges), step=1.0)

# --- دکمه پیش‌بینی و نمایش نتیجه ---
st.markdown('---')
if st.button(' تحلیل و پیش‌بینی کن!', use_container_width=True):
    # تبدیل گزینه‌های فارسی به انگلیسی برای مدل
    input_data = {
        'gender': gender_options[gender_fa],
        'SeniorCitizen': 1 if senior_citizen_fa == 'بله (Yes)' else 0,
        'Partner': yes_no_options[partner_fa],
        'Dependents': yes_no_options[dependents_fa],
        'tenure': tenure,
        'PhoneService': yes_no_options[phone_service_fa],
        'MultipleLines': multiple_lines_options[multiple_lines_fa],
        'InternetService': internet_service_options[internet_service_fa],
        'OnlineSecurity': internet_dependent_options[online_security_fa],
        'OnlineBackup': internet_dependent_options[online_backup_fa],
        'DeviceProtection': internet_dependent_options[device_protection_fa],
        'TechSupport': internet_dependent_options[tech_support_fa],
        'StreamingTV': internet_dependent_options[streaming_tv_fa],
        'StreamingMovies': internet_dependent_options[streaming_movies_fa],
        'Contract': contract_options[contract_fa],
        'PaperlessBilling': yes_no_options[paperless_billing_fa],
        'PaymentMethod': payment_method_options[payment_method_fa],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # فرآیند پیش‌پردازش (دقیقا مانند فایل آموزش)
    input_df = pd.DataFrame([input_data])
    input_dummies = pd.get_dummies(input_df)
    input_final = input_dummies.reindex(columns=model_columns, fill_value=0)
    scaled_input = scaler.transform(input_final)

    prediction_proba = model.predict_proba(scaled_input)
    churn_probability = prediction_proba[0][1] * 100

    st.subheader('📈 نتیجه تحلیل:')
    if churn_probability > 50:
        st.error(f'🔴 خطر ریزش بالا: این مشتری به احتمال {churn_probability:.2f}% شرکت را ترک خواهد کرد.')
        st.warning(
            'پیشنهاد می‌شود برای حفظ این مشتری، اقدامات لازم مانند ارائه تخفیف یا تماس از طرف تیم پشتیبانی انجام شود.')
    else:
        st.success(f'🟢 مشتری وفادار: این مشتری به احتمال {(100 - churn_probability):.2f}% در شرکت باقی می‌ماند.')
        st.info('احتمال ریزش این مشتری پایین است و در حال حاضر نیاز به اقدام خاصی نیست.')