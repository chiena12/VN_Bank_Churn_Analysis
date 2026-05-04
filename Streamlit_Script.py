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
        st.info("**Tech Stack:** SQL, Python, Streamlit")

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
    sns.boxplot(data=df_raw, x='exit', y='credit_sco', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title("3. Tương quan Điểm Tín Dụng (Credit Score) và Exit", fontsize=12, fontweight='bold')

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

with tabs[3]:
    st.header("🎯 CHIẾN LƯỢC GIỮ CHÂN KHÁCH HÀNG (RETENTION STRATEGY)")
    
    # 1. Phân tích từ dữ liệu (Insight Driven)
    st.subheader("1. Những phát hiện quan trọng từ dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Nhóm Gen Z (27% exit_ratio):** Đây là nhóm có tỷ lệ rời bỏ cao nhất trong tập dữ liệu. Điều này phản ánh tâm lý ưa thích trải nghiệm mới và ít tính cam kết hơn các thế hệ trước.")
    with col2:
        st.success("**Nhóm Boomers & Gen X:** Tỷ lệ rời bỏ duy trì ở mức thấp (~15-16%). Đây là nhóm khách hàng trung thành, ưu tiên sự ổn định và tin tưởng[cite: 3].")

    st.divider()

    # 2. Chiến lược cụ thể cho từng nhóm tuổi
    st.subheader("2. Đề xuất chiến lược theo nhóm khách hàng")
    
    with st.expander("🚀 Chiến lược cho Gen Z & Millenials (Nhóm rủi ro cao)"):
        st.write("""
        *   **Cá nhân hóa trải nghiệm số:** Tăng cường tính năng trên App Mobile vì nhóm này có hành vi 'digital_behavior' cao[cite: 3].
        *   **Gamification:** Xây dựng hệ thống đổi điểm thưởng, săn voucher ngay trên ứng dụng để tăng 'engagement_score'.
        *   **Sản phẩm linh hoạt:** Cung cấp các gói vay tiêu dùng nhanh hoặc thẻ tín dụng ảo với thủ tục đơn giản.
        """)

    with st.expander("🛡️ Chiến lược cho Gen X & Boomers (Nhóm bền vững)"):
        st.write("""
        *   **Chăm sóc đặc quyền:** Tập trung vào các gói bảo hiểm, hưu trí hoặc quản lý tài sản (Wealth Management) cho phân khúc 'Priority'[cite: 3].
        *   **Hỗ trợ đa kênh:** Kết hợp hỗ trợ trực tiếp tại quầy và điện thoại, vì nhóm này vẫn duy trì hành vi 'offline'[cite: 3].
        *   **Ưu đãi lòng trung thành:** Tăng lãi suất tiết kiệm bậc thang dựa trên `tenure_ye` (số năm gắn bó).
        """)

    st.divider()

    # 3. Tối ưu hóa phân khúc (Customer Segment)
    st.subheader("3. Hành động dựa trên phân khúc")
    
    seg_col1, seg_col2, seg_col3 = st.columns(3)
    with seg_col1:
        st.markdown("### 💎 Priority")
        st.write("Duy trì trạng thái 'Active Member' bằng các ưu đãi phòng chờ sân bay, thẻ đen[cite: 3].")
    with seg_col2:
        st.markdown("### 📈 Emerging")
        st.write("Kích thích sử dụng thêm dịch vụ (Cross-sell) để nâng hạng lên Priority[cite: 3].")
    with seg_col3:
        st.markdown("### 👥 Mass")
        st.write("Tự động hóa chăm sóc qua chatbot để tối ưu chi phí nhưng vẫn đảm bảo kết nối[cite: 3].")

    st.divider()

    # 4. Đề xuất chiến lược tổng quát
    st.subheader("4. Đề xuất chiến lược phù hợp với phân tích")
    st.warning("Dựa trên phân tích SHAP và rủi ro (Risk Score), ngân hàng cần lưu ý[cite: 3]:")
    
    st.write("""
    *   **Giám sát biến động số dư (Balance):** Dữ liệu cho thấy những khách hàng có số dư thay đổi đột ngột thường có xu hướng rời bỏ. Cần thiết lập hệ thống cảnh báo sớm (Early Warning System)[cite: 3].
    *   **Nâng cao chỉ số Engagement:** Tập trung vào những khách hàng có `engagement_score` thấp (dưới 40) bằng các chiến dịch Email Marketing cá nhân hóa[cite: 3].
    *   **Ưu tiên khách hàng có rủi ro thấp (Low Risk Segment):** Tập trung nguồn lực giữ chân nhóm `Low Risk` nhưng đang có dấu hiệu giảm tương tác để bảo vệ nguồn doanh thu ổn định[cite: 3].
    """)

        
    
