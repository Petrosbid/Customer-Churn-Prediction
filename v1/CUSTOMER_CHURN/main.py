
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

print("Advanced training model script started...")

# --- 1. بارگذاری و پاک‌سازی داده‌ها ---\r
csv_path = 'inputs/WA_Fn-UseC_-Telco-Customer-Churn.csv'
if not os.path.exists(csv_path) and os.path.exists('WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(csv_path)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df_for_viz = df.copy()

df = df.drop('customerID', axis=1)

# تبدیل مقادیر ستون هدف به 0 و 1
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

y = df['Churn'].values.astype(int)

X_raw = df.drop(columns=['Churn'])
X = pd.get_dummies(X_raw)

print("Data preprocessing completed.")

# --- 2. آماده‌سازی داده‌ها برای مدل‌سازی ---
model_columns = list(X.columns)

# تقسیم داده‌ها به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی ویژگی‌های عددی
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ذخیره کردن ستون‌های مدل و scaler برای استفاده در app.py
joblib.dump(model_columns, 'models/model_columns.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

# تنظیم پارامترها برای GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# تعریف مدل پایه‌ای XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# جستجوی شبکه (Grid Search) برای یافتن بهترین هایپرپارامترها
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n--- Best parameters found ---")
print(grid_search.best_params_)
print("-----------------------------------\n")

# ذخیره مدل نهایی برتر
joblib.dump(best_model, 'models/churn_model.joblib')

# --- 3. پیش‌بینی و ارزیابی مدل نهایی ---
print("--- Evaluation results of the best model (XGBoost) ---")

# پیش‌بینی کلاس‌ها (0 یا 1) به صورت قطعی و تبدیل صریح به int
y_pred = best_model.predict(X_test).astype(int)

print(f"دقت مدل: {accuracy_score(y_test, y_pred):.4f}")
print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# ایجاد پوشه خروجی برای نمودارها در صورت عدم وجود
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# رسم نمودارهای توزیع داده‌ها بر اساس کپی اولیه (df_for_viz)
sns.countplot(x='Churn', data=df_for_viz).get_figure().savefig(f'{output_dir}/churn_distribution.png')
plt.close()
sns.histplot(data=df_for_viz, x='tenure', hue='Churn', multiple='stack', kde=True).get_figure().savefig(f'{output_dir}/tenure_distribution.png')
plt.close()

# محاسبه احتمالات برای رسم منحنی ROC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# رسم منحنی ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

print(f"تمامی نمودارها با موفقیت در پوشه '{output_dir}' ذخیره شدند.")