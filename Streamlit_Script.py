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
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN & STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Vietnam Bank Churn Analytics", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fdfdfd; }
    .stMetric { border: 1px solid #ececec; padding: 15px; border-radius: 10px; background-color: white; }
    h1, h2 { color: #0e1133; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Phân Tích & Dự Báo Khách Hàng Rời Bỏ Ngân Hàng")
st.markdown("---")

# ---------------------------------------------------------
# 2. HÀM LOAD & TỐI ƯU DỮ LIỆU (FIXED CATEGORICAL BUG)
# ---------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    path = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")
    file_path = f"{path}/{os.listdir(path)[0]}"
    df = pd.read_csv(file_path)
    
    # 1. Lấy mẫu 30% để đảm bảo không tràn RAM 1GB của Streamlit Cloud
    df, _ = train_test_split(df, train_size=0.3, stratify=df['exit'], random_state=42)
    
    # 2. Feature Engineering
    df['total_product'] = (df['nums_card'] + df['nums_service']).astype('int32')
    df['balance_income_ratio'] = (df['balance'] / (df['monthly_ir'] + 1)).astype('float32')
    
    # 3. Loại bỏ cột không mang giá trị dự báo hoặc quá nặng
    cols_to_drop = ['id', 'full_name', 'address', 'last_active_date', 'created_date', 'origin_province']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 4. FIX BUG: Ép kiểu Object sang Category cho XGBoost/LGBM
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype('category')
        
    # 5. Downcasting kiểu số để tiết kiệm RAM
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype('float32')
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = df_clean[col].astype('int32')
        
    return df, df_clean

df_raw, df_ml = load_and_prep_data()

# Tách dữ liệu cho Modeling
X = df_ml.drop('exit', axis=1)
y = df_ml['exit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. HUẤN LUYỆN MÔ HÌNH (CACHE RESOURCE)
# ---------------------------------------------------------
@st.cache_resource
def train_ensemble_models(_X_train, _y_train):
    # XGBoost với Native Categorical Support
    xgb = XGBClassifier(n_estimators=60, learning_rate=0.08, max_depth=5, 
                        enable_categorical=True, tree_method="hist", random_state=42)
    xgb.fit(_X_train, _y_train)
    
    # LightGBM
    lgbm = LGBMClassifier(n_estimators=60, learning_rate=0.08, num_leaves=25, 
                          verbosity=-1, random_state=42)
    lgbm.fit(_X_train, _y_train)
    
    # Stacking
    stack = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm)],
        final_estimator=LogisticRegression()
    )
    stack.fit(_X_train, _y_train)
    
    return xgb, lgbm, stack

xgb_m, lgbm_m, stack_m = train_ensemble_models(X_train, y_train)

# ---------------------------------------------------------
# 4. DASHBOARD TABS
# ---------------------------------------------------------
tabs = st.tabs(["📊 EDA & Tương Quan", "⚖️ So Sánh Model", "🔍 Giải Thích SHAP", "💡 Chiến Lược"])

# --- TAB 1: EDA ĐẦY ĐỦ ---
with tabs[0]:
    st.header("1. Phân Tích Khám Phá Dữ Liệu (EDA)")
    
    col_eda1, col_eda2 = st.columns([1, 1.5])
    
    with col_eda1:
        st.subheader("Tỷ lệ Rời bỏ (Target Distribution)")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.countplot(data=df_raw, x='exit', palette='magma', ax=ax1)
        st.pyplot(fig1)
        st.info("Dữ liệu cho thấy tỷ lệ khách hàng rời bỏ chiếm khoảng 18-20%.")

    with col_eda2:
        st.subheader("Ma Trận Tương Quan (Numeric Features)")
        # Lọc chỉ các cột số để vẽ Heatmap
        numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_ml[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    st.divider()
    
    st.subheader("Phân tích Đặc trưng vs Trạng thái Churn")
    col_eda3, col_eda4 = st.columns(2)
    with col_eda3:
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df_raw, x='exit', y='risk_score', palette='Set2', ax=ax3)
        ax3.set_title("Risk Score vs Exit")
        st.pyplot(fig3)
    with col_eda4:
        fig4, ax4 = plt.subplots()
        sns.kdeplot(data=df_raw, x='balance', hue='exit', fill=True, ax=ax4)
        ax4.set_title("Phân bổ Số dư tài khoản (Balance)")
        st.pyplot(fig4)

# --- TAB 2: SO SÁNH MODEL ---
with tabs[1]:
    st.header("2. Đánh giá Hiệu năng Mô hình")
    
    models = {"XGBoost": xgb_m, "LightGBM": lgbm_m, "Stacking Ensemble": stack_m}
    
    m_col1, m_col2, m_col3 = st.columns(3)
    metrics_list = [m_col1, m_col2, m_col3]
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 5))
    
    for i, (name, model) in enumerate(models.items()):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {score:.3f})', lw=2)
        metrics_list[i].metric(name, f"{score:.3f}", delta="AUC Score")

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title("Biểu đồ ROC-AUC So Sánh")
    ax_roc.legend()
    st.pyplot(fig_roc)
    
    st.success("Mô hình Stacking thường cho kết quả AUC cao nhất nhờ sự kết hợp đa dạng các thuật toán Boosting.")

# --- TAB 3: SHAP ---
with tabs[2]:
    st.header("3. Giải mã Mô hình (Explainable AI)")
    st.markdown("Tại sao khách hàng rời đi? SHAP sẽ giúp chúng ta trả lời.")
    
    # Tính toán SHAP (giới hạn 50 mẫu để app không crash)
    X_shap = X_test.iloc[:50]
    explainer = shap.TreeExplainer(xgb_m)
    shap_vals = explainer.shap_values(X_shap)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Tầm quan trọng của các biến")
        fig_s1 = plt.figure()
        shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False)
        st.pyplot(fig_s1)
        plt.clf()
    with col_s2:
        st.subheader("Hướng tác động của đặc trưng")
        fig_s2 = plt.figure()
        shap.summary_plot(shap_vals, X_shap, show=False)
        st.pyplot(fig_s2)
        plt.clf()

# --- TAB 4: KẾT LUẬN ---
with tabs[3]:
    st.header("4. Đề xuất Chiến lược Kinh doanh")
    
    st.markdown("""
    ### 📌 Quan sát từ dữ liệu:
    - **Risk Score:** Là yếu tố dự báo mạnh nhất. Khách hàng có điểm rủi ro cao cần được quan tâm đặc biệt.
    - **Balance:** Khách hàng có số dư thấp thường có xu hướng rời bỏ cao hơn.
    - **Total Products:** Khách hàng sử dụng càng nhiều dịch vụ (Thẻ + Dịch vụ số) thì tỷ lệ rời bỏ càng giảm.
    
    ### 🚀 Hành động thực tế:
    1. **Chiến dịch Giữ chân (Retention):** Tự động lọc danh sách khách hàng có xác suất rời bỏ > 0.6 từ mô hình để gửi ưu đãi phí thường niên.
    2. **Upsell Sản phẩm:** Khuyến khích nhóm khách hàng chỉ dùng 1 sản phẩm mở thêm tài khoản thanh toán hoặc thẻ tín dụng để tăng sự gắn kết.
    3. **Quản trị Rủi ro:** Tích hợp trực tiếp điểm số từ mô hình Stacking vào hệ thống CRM của Ngân hàng.
    """)
    st.balloons()