# train_model_advanced.py (نسخه اصلاح شده)

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
df = pd.read_csv('inputs/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# **اصلاح ۱: ایجاد کپی برای بصری‌سازی قبل از هر تغییری**
df_for_viz = df.copy()

# ادامه پیش‌پردازش
df = df.drop('customerID', axis=1)

# **اصلاح ۲: به‌روزرسانی کد replace برای حذف هشدار پانداز**
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

df_dummies = pd.get_dummies(df)

print("Data preprocessing completed.")

# --- 2. آماده‌سازی داده‌ها برای مدل‌سازی ---
y = df_dummies['Churn'].values
X = df_dummies.drop(columns=['Churn'])

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.joblib')

# نگه داشتن نسخه اصلی X_test برای بصری‌سازی مرز تصمیم
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

print("Data prepared for training.")

# --- 3. آموزش مدل پیشرفته XGBoost با GridSearchCV ---
print("Starting advanced model training... (This may take a few minutes)")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# **اصلاح ۳: حذف پارامتر منسوخ شده از XGBClassifier**
grid_search = GridSearchCV(estimator=XGBClassifier(eval_metric='logloss', random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n--- Best parameters found ---")
print(grid_search.best_params_)
print("-----------------------------------")

y_pred = best_model.predict(X_test)
print("\n--- Evaluation results of the best model (XGBoost) ---")
print(f"دقت مدل: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("-----------------------------------------")

# --- 4. ذخیره مدل و Scaler ---
joblib.dump(best_model, 'churn_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Advanced model, scaler and columns successfully saved.")

# --- 5. ساخت و ذخیره نمودارها ---
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Now this section runs without errors since df_for_viz is defined
sns.countplot(x='Churn', data=df_for_viz).get_figure().savefig(f'{output_dir}/churn_distribution.png')
plt.close()
sns.histplot(data=df_for_viz, x='tenure', hue='Churn', multiple='stack', kde=True).get_figure().savefig(f'{output_dir}/tenure_distribution.png')
plt.close()

# New plots
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

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_test_orig['tenure'], X_test_orig['MonthlyCharges'], c=y_pred_proba, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Predicted Churn Probability')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.title('Decision Boundary Scatter Plot of Tenure and Monthly Charges')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'decision_boundary_scatter.png'))
plt.close()

print(f"All plots successfully saved in '{output_dir}' folder.")
print("Advanced script successfully completed.")