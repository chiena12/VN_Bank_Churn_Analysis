import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(page_title="Bank Churn Analytics - RAM Optimized", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #dee2e6; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Dự Đoán Khách Hàng Rời Bỏ (Tối Ưu Hệ Thống)")

# ---------------------------------------------------------
# HÀM LOAD & TỐI ƯU DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def load_and_optimize_data():
    # Tải dữ liệu
    path = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")
    file_path = f"{path}/{os.listdir(path)[0]}"
    df = pd.read_csv(file_path)
    
    # 1. Lấy mẫu dữ liệu (Sampling) để tránh tràn RAM Streamlit (Giữ 30% dữ liệu)
    # Sử dụng stratify để giữ nguyên tỷ lệ Churn
    df, _ = train_test_split(df, train_size=0.3, stratify=df['exit'], random_state=42)
    
    # 2. Feature Engineering cơ bản
    df['total_product'] = df['nums_card'] + df['nums_service']
    df['balance_income_ratio'] = df['balance'] / (df['monthly_ir'] + 1)
    
    # 3. Ép kiểu dữ liệu để tiết kiệm RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
        
    # 4. Encoding
    cat_cols = ['occupation', 'customer_segment', 'loyalty_level']
    df_ml = pd.get_dummies(df.drop(['id', 'full_name', 'address', 'last_active_date', 'created_date'], axis=1), 
                           columns=cat_cols, drop_first=True)
    df_ml['gender'] = df_ml['gender'].map({'Male': 1, 'Female': 0})
    
    return df, df_ml

df_raw, df_ml = load_and_optimize_data()

# Chuẩn bị tập dữ liệu
X = df_ml.drop('exit', axis=1)
y = df_ml['exit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# HUẤN LUYỆN MÔ HÌNH (SỬ DỤNG CACHE RESOURCE)
# ---------------------------------------------------------
@st.cache_resource
def train_models_fast(_X_train, _y_train):
    # XGBoost
    xgb = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    xgb.fit(_X_train, _y_train)
    
    # LightGBM
    lgbm = LGBMClassifier(n_estimators=50, learning_rate=0.1, num_leaves=20, verbosity=-1, random_state=42)
    lgbm.fit(_X_train, _y_train)
    
    # Stacking
    estimators = [('xgb', xgb), ('lgbm', lgbm)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking.fit(_X_train, _y_train)
    
    return xgb, lgbm, stacking

xgb_model, lgbm_model, stack_model = train_models_fast(X_train, y_train)

# ---------------------------------------------------------
# BỐ CỤC DASHBOARD
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình hệ thống")
st.sidebar.write(f"Dữ liệu đã nạp: **{len(df_raw)} dòng**")
st.sidebar.info("App đã được tối ưu hóa bộ nhớ để chạy trên Cloud.")

tabs = st.tabs(["📊 EDA", "⚖️ So sánh Mô hình", "🔍 Giải thích SHAP", "💡 Chiến lược"])

# --- TAB 1: EDA ---
with tabs[0]:
    st.header("Phân tích dữ liệu tinh gọn")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df_raw, x='balance', hue='exit', kde=True, ax=ax)
        ax.set_title("Phân bổ Số dư theo trạng thái Rời bỏ")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_raw, x='customer_segment', y='risk_score', hue='exit', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- TAB 2: SO SÁNH MÔ HÌNH (ROC-AUC) ---
with tabs[1]:
    st.header("Đánh giá Performance (XGB vs LGBM vs Stacking)")
    
    models = {"XGBoost": xgb_model, "LightGBM": lgbm_model, "Stacking": stack_model}
    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    
    cols = st.columns(3)
    for i, (name, model) in enumerate(models.items()):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        score = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {score:.3f})')
        cols[i].metric(name, f"{score:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.legend()
    st.pyplot(fig_roc)

# --- TAB 3: SHAP (GIẢI THÍCH MÔ HÌNH) ---
with tabs[2]:
    st.header("Tính minh bạch của mô hình (XAI)")
    st.write("Sử dụng SHAP để hiểu các yếu tố tác động mạnh nhất.")
    
    # Chỉ lấy 50 mẫu để tính SHAP tránh tràn RAM
    X_sample = X_test.iloc[:50]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Độ quan trọng tổng quát")
        fig_s1 = plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig_s1)
    with col_s2:
        st.subheader("Chi tiết tác động")
        fig_s2 = plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig_s2)

# --- TAB 4: KẾT LUẬN ---
with tabs[3]:
    st.header("Kết luận & Đề xuất")
    st.success("""
    - **Mô hình tốt nhất:** Stacking Ensemble (Kết hợp XGB & LGBM).
    - **Yếu tố then chốt:** Risk Score, Balance và Customer Segment là 3 biến dự báo quan trọng nhất.
    - **Hành động:** Tập trung chăm sóc khách hàng có Risk Score cao nhưng vẫn còn số dư (Balance) lớn.
    """)