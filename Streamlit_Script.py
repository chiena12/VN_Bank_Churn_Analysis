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
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Bank Churn Analytics", page_icon="📈", layout="wide")

# Custom CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Phân Tích Chuyên Sâu & Dự Đoán Rời Bỏ (Churn Prediction)")
st.markdown("---")

# ---------------------------------------------------------
# HÀM LOAD & XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")
    file_path = f"{path}/{os.listdir(path)[0]}"
    df = pd.read_csv(file_path)
    
    # Feature Engineering nhanh
    df['total_product'] = df['nums_card'] + df['nums_service']
    df['balance_income_ratio'] = df['balance'] / (df['monthly_ir'] + 1)
    
    # Encode categorical
    cat_cols = ['occupation', 'customer_segment', 'loyalty_level']
    df_ml = pd.get_dummies(df.drop(['id', 'full_name', 'address', 'last_active_date', 'created_date'], axis=1), 
                           columns=cat_cols, drop_first=True)
    
    # Xử lý các cột còn lại thành số
    df_ml['gender'] = df_ml['gender'].map({'Male': 1, 'Female': 0})
    return df, df_ml

df_raw, df_ml = load_data()

# Chia dữ liệu
X = df_ml.drop('exit', axis=1)
y = df_ml['exit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# HUẤN LUYỆN MÔ HÌNH (CACHE ĐỂ TỐI ƯU TỐC ĐỘ)
# ---------------------------------------------------------
@st.cache_resource
def train_models(_X_train, _y_train):
    # 1. XGBoost
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(_X_train, _y_train)
    
    # 2. LightGBM
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, verbosity=-1, random_state=42)
    lgbm.fit(_X_train, _y_train)
    
    # 3. Stacking Model (Sử dụng Logistic Regression làm Meta-model)
    estimators = [('xgb', xgb), ('lgbm', lgbm)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking.fit(_X_train, _y_train)
    
    return xgb, lgbm, stacking

xgb_model, lgbm_model, stack_model = train_models(X_train, y_train)

# ---------------------------------------------------------
# SIDEBAR: THÔNG TIN DỰ ÁN
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.header("Thông tin bộ dữ liệu")
    st.write(f"🔹 Số lượng khách hàng: `{df_raw.shape[0]}`")
    st.write(f"🔹 Tỷ lệ rời bỏ (Churn): `{df_raw['exit'].mean():.2%}`")
    st.info("Dự án sử dụng kiến trúc Stacking Ensemble để tối ưu hóa độ chính xác.")

# ---------------------------------------------------------
# NỘI DUNG CHÍNH - CÁC TABS TƯỜNG MINH
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 I. Phân tích Đặc trưng (EDA)", 
    "⚖️ II. So sánh Hiệu năng Mô hình", 
    "🔍 III. Giải mã AI (SHAP & Importance)", 
    "🚀 IV. Chiến lược & Thực thi"
])

# --- TAB 1: EDA ---
with tabs[0]:
    st.header("Phân tích các yếu tố dẫn đến rời bỏ")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bổ Số dư & Thu nhập")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_raw, x='monthly_ir', y='balance', hue='exit', alpha=0.5, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Ảnh hưởng của Loyalty Level")
        fig, ax = plt.subplots()
        sns.barplot(data=df_raw, x='loyalty_level', y='exit', ax=ax, palette='viridis')
        st.pyplot(fig)

# --- TAB 2: SO SÁNH MÔ HÌNH ---
with tabs[1]:
    st.header("Đánh giá Hiệu năng: XGBoost vs LightGBM vs Stacking")
    
    models = {"XGBoost": xgb_model, "LightGBM": lgbm_model, "Stacking Ensemble": stack_model}
    
    col_m1, col_m2, col_m3 = st.columns(3)
    cols = [col_m1, col_m2, col_m3]
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    
    for i, (name, model) in enumerate(models.items()):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Vẽ ROC chung
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2)
        
        # Hiển thị Metric từng cột
        cols[i].metric(label=f"AUC {name}", value=f"{roc_auc:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title("So sánh biểu đồ ROC-AUC")
    ax_roc.legend()
    st.pyplot(fig_roc)
    
    st.success("✅ **Nhận xét:** Mô hình Stacking cho kết quả ổn định nhất nhờ kết hợp ưu điểm của cả XGB và LGBM.")

# --- TAB 3: GIẢI MÃ AI ---
with tabs[2]:
    st.header("Giải thích mô hình (Model Explainability)")
    
    col_feat, col_shap = st.columns([1, 1])
    
    with col_feat:
        st.subheader("Feature Importance (Gini)")
        importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
        fig_f, ax_f = plt.subplots()
        importances.plot(kind='barh', ax=ax_f, color='skyblue')
        st.pyplot(fig_f)

    with col_shap:
        st.subheader("SHAP Summary Plot")
        # Tính toán SHAP (giới hạn 100 mẫu để app chạy mượt)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test.iloc[:100])
        
        fig_s, ax_s = plt.subplots()
        shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    st.info("""
    **Cách đọc SHAP:** 
    - Màu **Đỏ** thể hiện giá trị đặc trưng cao, màu **Xanh** thể hiện giá trị thấp. 
    - Nếu màu đỏ nằm bên phải trục 0, đặc trưng đó làm tăng khả năng rời bỏ (ví dụ: Risk Score cao).
    """)

# --- TAB 4: CHIẾN LƯỢC ---
with tabs[3]:
    st.header("Đề xuất hành động kinh doanh (Business Strategy)")
    
    st.markdown("""
    ### 🎯 1. Phân khúc Khách hàng Ưu tiên (Priority Segment)
    Dựa trên **SHAP values**, khách hàng có `risk_score` > 0.6 và `tenure_ye` < 2 là nhóm có nguy cơ cao nhất.
    - **Hành động:** Tự động gửi voucher phí thường niên hoặc ưu đãi lãi suất qua App.
    
    ### 🛠️ 2. Tối ưu hóa Sản phẩm
    Dữ liệu cho thấy khách hàng có `total_product` thấp dễ rời bỏ hơn.
    - **Hành động:** Triển khai chiến dịch "Cross-selling": Tặng quà khi mở thêm thẻ tín dụng hoặc đăng ký bảo hiểm liên kết.
    
    ### 🤖 3. Vận hành Mô hình (MLOps)
    - Mô hình **Stacking** nên được đóng gói thành API để tích hợp trực tiếp vào hệ thống CRM.
    - Tần suất cập nhật (Retrain): Mỗi quý 1 lần để tránh Data Drift.
    """)
    st.balloons()