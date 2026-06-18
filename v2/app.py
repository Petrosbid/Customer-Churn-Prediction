
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- بارگذاری مدل، scaler و ستون‌های ذخیره شده ---
try:
    model = joblib.load('v2/models/churn_model.joblib')
    scaler = joblib.load('v2/models/scaler.joblib')
    model_columns = joblib.load('v2/models/model_columns.joblib')
except FileNotFoundError:
    st.error("فایل‌های مدل یافت نشدند! لطفاً ابتدا اسکریپت main.py را برای آموزش مدل اجرا کنید.")
    st.stop()

# --- تعریف دیکشنری برای گزینه‌ها (تبدیل نمایش فارسی به مقدار انگلیسی مورد نیاز مدل) ---
gender_options = {'زن (Female)': 'Female', 'مرد (Male)': 'Male'}
subscription_options = {'پایه (Basic)': 'Basic', 'استاندارد (Standard)': 'Standard', 'پرایمیوم (Premium)': 'Premium'}
contract_options = {'ماهانه (Monthly)': 'Monthly', 'سالانه (Annual)': 'Annual', 'سه ماهه (Quarterly)': 'Quarterly'}

# --- تنظیمات صفحه وب استریملیت ---
st.set_page_config(layout="wide", page_title="پیش‌بینی ریزش مشتریان")
st.title('🤖 سامانه هوشمند پیش‌بینی ریزش مشتریان')
st.markdown('این برنامه مشخصات رفتاری و مالی یک مشتری را دریافت کرده و احتمال قطع اشتراک او را تحلیل می‌کند.')
st.markdown('---')

# --- بخش ورودی اطلاعات ---
st.header('لطفاً مشخصات مشتری را وارد کنید:')

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 مشخصات فردی و سابقه")
    age = st.slider('۱. سن مشتری چیست؟', 18, 65, 35)
    gender_fa = st.selectbox('۲. جنسیت مشتری:', list(gender_options.keys()))
    tenure = st.slider('۳. سابقه اشتراک مشتری (به ماه):', 1, 60, 24)

with col2:
    st.subheader("📊 فعالیت و استفاده")
    usage_frequency = st.slider('۴. میزان استفاده در ماه (فرکانس):', 1, 30, 15)
    last_interaction = st.slider('۵. آخرین تعامل یا بازدید (چند روز پیش؟):', 1, 30, 12)
    support_calls = st.number_input('۶. تعداد تماس‌ها با پشتیبانی:', min_value=0, max_value=10, value=2, step=1)

with col3:
    st.subheader("💳 وضعیت قرارداد و مالی")
    sub_type_fa = st.selectbox('۷. نوع اشتراک (Subscription Type):', list(subscription_options.keys()))
    contract_fa = st.selectbox('۸. مدت قرارداد (Contract Length):', list(contract_options.keys()))
    payment_delay = st.slider('۹. تعداد روزهای تاخیر در پرداخت صورتحساب:', 0, 30, 4)
    total_spend = st.number_input('۱۰. کل هزینه پرداختی مشتری تاکنون ($):', min_value=100, max_value=1000, value=500,
                                  step=10)

st.markdown('---')

# --- دکمه پیش‌بینی و نمایش نتیجه ---
if st.button('🚀 تحلیل هوشمند و پیش‌بینی ریزش مشتری', use_container_width=True):
    # نگاشت مقادیر ورودی به فرمت انگلیسی دیتاست
    input_data = {
        'Age': age,
        'Gender': gender_options[gender_fa],
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': subscription_options[sub_type_fa],
        'Contract Length': contract_options[contract_fa],
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }

    # پیش‌پردازش ورودی‌ها هماهنگ با ساختار آموزش مدل
    input_df = pd.DataFrame([input_data])
    input_dummies = pd.get_dummies(input_df)

    # همترازی ستون‌ها با مدل اصلی و پر کردن ستون‌های مفقود دامی با 0
    input_final = input_dummies.reindex(columns=model_columns, fill_value=0)
    scaled_input = scaler.transform(input_final)

    # پیش‌بینی احتمال ریزش
    prediction_proba = model.predict_proba(scaled_input)
    churn_probability = prediction_proba[0][1] * 100

    # نمایش نتایج نهایی به کاربر
    st.subheader('📈 نتیجه ارزیابی هوش مصنوعی:')

    if churn_probability > 50:
        st.error(f'🔴 خطر ریزش بالا: این مشتری به احتمال {churn_probability:.2f}% شرکت را ترک خواهد کرد.')
        st.warning('⚠️ پیشنهاد می‌شود سریعاً برای این مشتری آفرها یا تخفیف‌های ویژه جهت تمدید اشتراک ارسال شود.')
    else:
        st.success(f'🟢 مشتری وفادار: این مشتری به احتمال {(100 - churn_probability):.2f}% با شرکت خواهد ماند.')
        st.info('ℹ️ وضعیت تعامل مشتری نرمال ارزیابی شده و نیاز به اقدام فوری ندارد.')