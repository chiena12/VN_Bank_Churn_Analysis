import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import shap
import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# =========================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# =========================================================
st.set_page_config(
    page_title="Vietnam Bank Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS để Dashboard trông hiện đại hơn
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1e3a8a;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e5e7eb;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #1e3a8a !important; color: white !important; }
    h1 { color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif; }
    h2, h3 { color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. XỬ LÝ DỮ LIỆU TỐI ƯU (RAM & TYPE FIXING)
# =========================================================
@st.cache_data
def load_and_preprocess_full():
    """
    Hàm tải dữ liệu, lấy mẫu và xử lý lỗi Categorical cho XGBoost.
    """
    # Tải dataset từ Kaggle
    path = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")
    file_path = f"{path}/{os.listdir(path)[0]}"
    df = pd.read_csv(file_path)
    
    # 1. Stratified Sampling (30%): Giữ đúng tỷ lệ Churn nhưng không làm tràn RAM Cloud
    df_sample, _ = train_test_split(df, train_size=0.3, stratify=df['exit'], random_state=42)
    
    # 2. Feature Engineering (Dựa trên yêu cầu của bạn)
    df_sample['total_product'] = (df_sample['nums_card'] + df_sample['nums_service']).astype('int32')
    df_sample['balance_income_ratio'] = (df_sample['balance'] / (df_sample['monthly_ir'] + 1)).astype('float32')
    
    # 3. Clean-up: Loại bỏ các cột định danh không có giá trị dự báo
    drop_cols = ['id', 'full_name', 'address', 'last_active_date', 'created_date', 'origin_province']
    df_clean = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns])
    
    # 4. FIX CATEGORICAL BUG: Chuyển string sang Category cho XGBoost Native Support
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype('category')
    
    # 5. Downcasting kiểu số để tiết kiệm bộ nhớ
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype('float32')
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = df_clean[col].astype('int32')
        
    return df_sample, df_clean

df_raw, df_ml = load_and_preprocess_full()

# Tách đặc trưng và nhãn
X = df_ml.drop('exit', axis=1)
y = df_ml['exit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 3. HUẤN LUYỆN MÔ HÌNH (ENSEMBLE & STACKING)
# =========================================================
@st.cache_resource
def train_complex_models(_X_train, _y_train):
    """
    Huấn luyện XGBoost, LightGBM và Stacking Model.
    """
    # XGBoost: Cấu hình fix lỗi Categorical và tối ưu tốc độ
    xgb = XGBClassifier(
        n_estimators=80, 
        learning_rate=0.07, 
        max_depth=6, 
        enable_categorical=True, 
        tree_method="hist", 
        random_state=42, 
        n_jobs=-1
    )
    xgb.fit(_X_train, _y_train)
    
    # LightGBM: Hiệu suất cao với dữ liệu phân loại
    lgbm = LGBMClassifier(
        n_estimators=80, 
        learning_rate=0.07, 
        num_leaves=31, 
        verbosity=-1, 
        random_state=42
    )
    lgbm.fit(_X_train, _y_train)
    
    # Stacking Model: Sử dụng Logistic Regression làm Meta-model
    estimators = [('xgb', xgb), ('lgbm', lgbm)]
    stacking = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=3 # Cross-validation cho meta-model
    )
    stacking.fit(_X_train, _y_train)
    
    return xgb, lgbm, stacking

with st.spinner("🚀 Đang khởi tạo mô hình AI chuyên sâu..."):
    xgb_model, lgbm_model, stack_model = train_complex_models(X_train, y_train)

# =========================================================
# 4. SIDEBAR - THÔNG TIN CHIẾN LƯỢC
# =========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=120)
    st.header("Thông tin Dự án")
    st.write(f"👤 **Author :** Chiến")
    st.write(f"🏢 **Đơn vị:** FTU - International Business Economics")
    st.markdown("---")
    st.subheader("Cấu hình Dữ liệu")
    st.write(f"✅ Tổng mẫu: `{len(df_raw)}` dòng")
    st.write(f"✅ Tỷ lệ Churn: `{y.mean():.1%}`")
    st.success("App đã được tối ưu hóa cho Streamlit Cloud (1GB RAM)")

# =========================================================
# 5. MAIN CONTENT - HỆ THỐNG TABS TƯỜNG MINH
# =========================================================
st.title("🏦 Phân Tích Các Yếu Tố Ảnh Hưởng Đến Khả Năng Rời Bỏ (Exit)")
st.markdown("Dự án sử dụng học máy nâng cao để phân tích hành vi khách hàng và đề xuất chiến lược giữ chân.")

tabs = st.tabs([
    "📂 Tổng quan & Insight", 
    "🎨 Phân tích EDA (Subplots)", 
    "📊 Hiệu năng Mô hình", 
    "🔬 Giải thích SHAP", 
    "🎯 Chiến lược Business"
])

# ---------------------------------------------------------
# TAB 1: TỔNG QUAN & KẾT QUẢ SQL
# ---------------------------------------------------------
with tabs[0]:
    st.header("📌 Mục tiêu & Insight chính")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        - **Mục tiêu:** Xác định các key features ảnh hưởng đến việc khách hàng đóng tài khoản.
        - **Dữ liệu:** Vietnam Bank Churn Dataset 2025.
        - **Phương pháp:** Phân tích SQL kết hợp Stacking Ensemble Learning.
        """)
    with col2:
        st.metric("Top Feature", "Customer Segment", delta="High Impact")

    st.subheader("💡 Key Insights từ SQL (Project Summary)")
    st.info("""
    * **Phân khúc (Customer Segment):** Là yếu tố then chốt nhất ở mọi mô hình.
    * **Nghề nghiệp:** Nhóm khách hàng tự doanh có rủi ro rời bỏ cao hơn nhóm nhân viên văn phòng.
    * **Độ trung thành:** Sự gắn kết qua các sản phẩm phụ (cross-selling) giúp giảm tỷ lệ exit rõ rệt.
    """)

# ---------------------------------------------------------
# TAB 2: CHI TIẾT EDA (TÁI HIỆN SUBPLOTS)
# ---------------------------------------------------------
with tabs[1]:
    st.header("🎨 Phân Tích Trực Quan Hóa (EDA Dashboard)")
    st.write("Tái hiện lại hệ thống subplots từ file Python.ipynb của bạn:")

    # HỆ THỐNG SUBPLOTS 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4)

    # 1. Phân bổ Risk Score theo Exit
    sns.kdeplot(data=df_raw, x='risk_score', hue='exit', fill=True, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title("1. Phân bổ Điểm Rủi Ro (Risk Score) theo Exit", fontsize=12, fontweight='bold')

    # 2. Tương quan Customer Segment vs Exit
    sns.countplot(data=df_raw, x='customer_segment', hue='exit', ax=axes[0, 1], palette='magma')
    axes[0, 1].set_title("2. Phân khúc Khách hàng vs Tỷ lệ Rời bỏ", fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Phân bổ Số dư tài khoản (Balance)
    sns.boxplot(data=df_raw, x='exit', y='balance', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title("3. Tương quan Số dư (Balance) và Exit", fontsize=12, fontweight='bold')

    # 4. Heatmap tương quan các biến số
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
    corr_matrix = df_ml[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("4. Ma Trận Tương Quan Biến Số", fontsize=12, fontweight='bold')

    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: HIỆU NĂNG MÔ HÌNH (ROC-AUC)
# ---------------------------------------------------------
with tabs[2]:
    st.header("⚖️ Đánh Giá Hiệu Năng Mô Hình (Benchmarking)")
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.subheader("Chỉ số AUC")
        model_dict = {"XGBoost": xgb_model, "LightGBM": lgbm_model, "Stacking Ensemble": stack_model}
        for name, model in model_dict.items():
            probs = model.predict_proba(X_test)[:, 1]
            score = auc(*roc_curve(y_test, probs)[:2])
            st.metric(name, f"{score:.4f}")

    with col_m2:
        st.subheader("Biểu đồ ROC-AUC Comparison")
        fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
        for name, model in model_dict.items():
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})', lw=2)
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend()
        st.pyplot(fig_roc)

# ---------------------------------------------------------
# TAB 4: GIẢI THÍCH SHAP (XAI)
# ---------------------------------------------------------
with tabs[3]:
    st.header("🔬 Explainable ML")
    st.write("SHAP giúp chúng ta hiểu lý do đằng sau các dự đoán của mô hình XGBoost.")
    
    # Tính SHAP cho 50 mẫu (để app chạy nhanh trên Cloud)
    X_shap = X_test.iloc[:50]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)
    
# Tạo khung hình lớn hơn để biểu đồ summary hiện đầy đủ và đẹp
    fig_shap, ax_shap = plt.subplots(figsize=(12, 8))
    
    # Vẽ summary_plot (mặc định là Beeswarm plot)
    # Nếu muốn đổi sang dạng thanh ngang đơn giản, thêm tham số plot_type="bar"
    shap.summary_plot(shap_values, X_shap, show=False)
    
    # Tăng kích thước font cho các trục để dễ đọc trên web
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Hiển thị lên Streamlit
    st.pyplot(fig_shap)
    
    # Quan trọng: Giải phóng bộ nhớ sau khi vẽ
    plt.clf()
    plt.close()

    st.info("""
    **Cách đọc biểu đồ:**
    - **Trục tung (Y):** Các biến số được sắp xếp theo thứ tự quan trọng giảm dần từ trên xuống dưới.
    - **Màu sắc:** Màu đỏ đại diện cho giá trị biến cao, màu xanh đại diện cho giá trị biến thấp.
    - **Trục hoành (X):** SHAP Value dương (về bên phải) làm tăng khả năng khách hàng rời bỏ (Exit=1).
    """)

# ---------------------------------------------------------
# TAB 5: CHIẾN LƯỢC KINH DOANH
# ---------------------------------------------------------
with tabs[4]:
    st.header("🎯 Đề Xuất Chiến Lược Giữ Chân Khách Hàng")
    
    st.success("Dựa trên kết quả mô hình Stacking, chúng tôi đưa ra các khuyến nghị sau:")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("""
        ### 📍 1. Tối ưu theo Customer Segment
        - Nhóm khách hàng **High Risk** cần được đưa vào chương trình chăm sóc đặc biệt (Loyalty Program).
        - Triển khai các gói ưu đãi phí thường niên cho nhóm có Risk Score > 0.7.
        
        ### 📍 2. Tăng cường Độ trung thành
        - Khách hàng có ít hơn 2 sản phẩm (`total_product`) có xu hướng rời bỏ cao.
        - Chiến lược: **Cross-selling** thêm các dịch vụ bảo hiểm hoặc thẻ tín dụng với lãi suất ưu đãi.
        """)
    
    with col_b2:
        st.markdown("""
        ### 📍 3. Hệ thống Cảnh báo Sớm (Early Warning)
        - Tích hợp mô hình Stacking vào hệ thống CRM để chấm điểm rủi ro hàng ngày.
        - Tự động hóa thông báo cho chuyên viên quan hệ khách hàng (RM) khi điểm rủi ro vượt ngưỡng 0.8.
        
        ### 📍 4. Vận hành & MLOps
        - Sử dụng **LightGBM** làm mô hình chính nếu yêu cầu tốc độ xử lý hàng triệu dòng dữ liệu thời gian thực.
        - Cập nhật (Retrain) mô hình hàng quý để tránh hiện tượng trôi dạt dữ liệu (Data Drift).
        """)
    
    st.balloons()