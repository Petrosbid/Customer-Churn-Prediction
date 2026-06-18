
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

# --- 1. بارگذاری و پاک‌سازی داده‌ها ---
file_names = ['customer_churn_dataset-training-master.csv', 'customer_churn_dataset-testing-master.csv']
csv_path = None

for f_name in file_names:
    if os.path.exists(f_name):
        csv_path = f_name
        break
    elif os.path.exists(os.path.join('dataset', f_name)):
        csv_path = os.path.join('dataset', f_name)
        break

if csv_path is None:
    raise FileNotFoundError("خطا: فایل داده‌های آموزش یافت نشد! لطفاً فایل csv را در پوشه پروژه قرار دهید.")

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)
df.dropna(inplace=True)

# کپی برای بصری‌سازی پیش از تغییر ساختار داده‌ها
df_for_viz = df.copy()

# حذف ستون شناسه مشتری چون ویژگی مفیدی برای مدل‌سازی نیست
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# جدا کردن ستون هدف با فرمت صریح عددی صحیح
y = df['Churn'].values.astype(int)

# دامی‌سازی متغیرهای متنی (One-Hot Encoding) روی ویژگی‌ها
X_raw = df.drop(columns=['Churn'])
X = pd.get_dummies(X_raw)

print("Data preprocessing completed.")

# --- 2. آماده‌سازی داده‌ها برای مدل‌سازی ---
model_columns = list(X.columns)
joblib.dump(model_columns, 'models/model_columns.joblib')

# تقسیم داده‌ها به آموزش و تست (۸۰٪ آموزش، ۲۰٪ تست)
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی ویژگی‌های عددی
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

print("Data prepared for training.")

# --- 3. آموزش مدل با GridSearchCV ---
print("Starting Grid Search CV... (Please wait)")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\n--- Best parameters found ---")
print(grid_search.best_params_)
print("-----------------------------------")

# ارزیابی مدل
y_pred = best_model.predict(X_test).astype(int)
print("\n--- Evaluation results of the best model (XGBoost) ---")
print(f"دقت مدل (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nگزارش کامل طبقه‌بندی:")
print(classification_report(y_test, y_pred))
print("-----------------------------------------")

# --- 4. ذخیره مدل و Scaler ---
joblib.dump(best_model, 'models/churn_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print("Model and Scaler successfully saved.")

# --- 5. ساخت و ذخیره نمودارها ---
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# نمودار ۱: توزیع کلاس هدف ریزش
plt.figure()
sns.countplot(x='Churn', data=df_for_viz).get_figure().savefig(f'{output_dir}/churn_distribution.png')
plt.close()

# نمودار ۲: هیستوگرام میزان حضور مشتری (Tenure) به تفکیک ریزش
plt.figure()
sns.histplot(data=df_for_viz, x='Tenure', hue='Churn', multiple='stack', kde=True).get_figure().savefig(f'{output_dir}/tenure_distribution.png')
plt.close()

# نمودار ۳: منحنی ROC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

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

# نمودار ۴: نمودار پراکندگی بر اساس سابقه و میزان کل هزینه پرداختی
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_test_orig['Tenure'], X_test_orig['Total Spend'], c=y_pred_proba, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Predicted Churn Probability')
plt.xlabel('Tenure (سابقه به ماه)')
plt.ylabel('Total Spend (کل هزینه پرداختی)')
plt.title('Decision Boundary Scatter Plot')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'decision_boundary_scatter.png'))
plt.close()

print(f"All plots successfully saved in '{output_dir}' folder.")
print("Advanced script successfully completed.")