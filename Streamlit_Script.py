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
    st.write(f"👤 **Author :** Đức Chiến")
    st.markdown("---")
    st.subheader("Cấu hình Dữ liệu")
    st.write(f"✅ Tổng mẫu: `{len(df_raw)}` dòng")
    st.write(f"✅ Tỷ lệ Churn: `{y.mean():.1%}`")

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
    st.header("🚀 Quy trình thực hiện Dự án (Project Workflow)")
    
    # Sử dụng cột để tạo cái nhìn tổng quan nhanh
    col_goal, col_tech = st.columns([2, 1])
    with col_goal:
        st.markdown("""
        **Mục tiêu chính:** Xây dựng hệ thống dự báo sớm (Early Warning System) giúp ngân hàng 
        giảm thiểu tỷ lệ khách hàng rời bỏ thông qua dữ liệu hành vi.
        """)
    with col_tech:
        st.info("**Tech Stack:** SQL, Python, LightGBM, Stacking")

    # Chia các bước lớn bằng expander hoặc markdown chuyên nghiệp
    st.subheader("🛠️ Các giai đoạn triển khai")

    with st.expander("1. Phân tích Insight & Khám phá dữ liệu (SQL Analytics)", expanded=True):
        st.markdown("""
        - **Data Source:** Truy vấn trực tiếp trên tập dữ liệu 80,000 khách hàng.
        - **Phân tích:** Sử dụng **T-SQL** để bóc tách tỷ lệ Exit theo từng nhóm nhân khẩu học.
        - **Kết quả:** Xác định **Customer Segment** và **Nghề nghiệp** là hai yếu tố then chốt gây ra rủi ro rời bỏ.
        """)

    with st.expander("2. Tiền xử lý & Kỹ thuật đặc trưng (Feature Engineering)"):
        st.markdown("""
        - **Cleaning:** Xử lý các cột dữ liệu thời gian (`last_active_date`, `created_date`) và làm sạch các giá trị nhiễu.
        - **Transformation:** Chia nhóm tuổi (Gen Z, Millennials, Gen X, Boomers) để tăng độ nhạy cho mô hình.
        - **Scale:** Xử lý mất cân bằng dữ liệu (Imbalanced Data) bằng cách điều chỉnh trọng số lớp (class weight).
        """)

    with st.expander("3. Huấn luyện mô hình & Tối ưu hóa (Model Training)"):
        st.markdown("""
        - **Mô hình:** Triển khai các thuật toán Boosting mạnh mẽ như **LightGBM** và **XGBoost**.
        - **Stacking Ensemble:** Kết hợp các mô hình đơn lẻ thông qua **StackingClassifier** để tối ưu hóa độ chính xác và khả năng tổng quát hóa.
        - **Tuning:** Sử dụng Randomized Search CV để tìm ra bộ siêu tham số (Hyperparameters) tốt nhất.
        - **Metric:** Tập trung vào chỉ số **Recall** và **AUC-ROC** để đảm bảo không bỏ sót khách hàng có rủi ro cao.
        """)

    st.success("✅ **Kết luận:** Mô hình LightGBM được chọn làm nhân tố chính nhờ tốc độ xử lý nhanh và độ ổn định cao trên dữ liệu lớn.")
# ---------------------------------------------------------
# TAB 2: CHI TIẾT EDA (TÁI HIỆN SUBPLOTS)
# ---------------------------------------------------------
with tabs[1]:
    st.header("🎨 Phân Tích Trực Quan Hóa (EDA Dashboard)")

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
    st.header("🔬 Explainable ML (SHAP Analysis)")
    st.write("Phân tích giá trị SHAP để hiểu rõ các yếu tố dẫn đến quyết định rời bỏ của khách hàng.")
    
    # 1. Lấy mẫu dữ liệu nhỏ để tránh lỗi RAM trên Streamlit Cloud
    X_shap = X_test.head(100) # Tăng lên 100 mẫu để biểu đồ mật độ (Beeswarm) nhìn đẹp hơn
    
    # 2. Khởi tạo Explainer và tính toán SHAP values
    # Sử dụng check_additivity=False để tránh lỗi nhỏ về sai số dấu phẩy động trên Cloud
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)
    
    # 3. FIX LỖI HIỂN THỊ: Tạo figure và xử lý plot
    # Lưu ý quan trọng: SHAP summary_plot tự tạo figure nội bộ, 
    # nên ta cần thiết lập kích thước qua plt.figure trước.
    fig_shap = plt.figure(figsize=(12, 8))
    
    # Vẽ summary_plot (Dạng Beeswarm - dấu chấm màu xanh đỏ như trong file Python của bạn)
    # Tham số show=False là bắt buộc để không bị treo app
    shap.summary_plot(shap_values, X_shap, show=False)
    
    # Tinh chỉnh thẩm mỹ để giống y hệt bản gốc
    plt.title("XGBoost Global Feature Importance (SHAP)", fontsize=15, pad=20)
    plt.xlabel("SHAP value (tác động đến xác suất rời bỏ)", fontsize=12)
    plt.grid(alpha=0.3) # Thêm lưới mờ cho chuyên nghiệp
    
    # 4. Hiển thị lên Streamlit
    st.pyplot(fig_shap, clear_figure=True)
    
    # 5. Giải phóng bộ nhớ
    plt.close(fig_shap)

    st.info("""
    **💡 Cụ thể:**
    - **Màu Đỏ (High):** Giá trị biến cao. 
    - **Màu Xanh (Low):** Giá trị biến thấp.
    - Vị trí của cụm giá trị nói lên ảnh hưởng đối với output của bài toán.
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
    
