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
# 1. CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(page_title="Bank Churn Analytics - Bug Fixed", page_icon="🏦", layout="wide")

st.title("🏦 Dự Đoán Khách Hàng Rời Bỏ (Bản Sửa Lỗi Hệ Thống)")

# ---------------------------------------------------------
# 2. HÀM LOAD & XỬ LÝ DỮ LIỆU (FIXED TYPES)
# ---------------------------------------------------------
@st.cache_data
def load_and_fix_data():
    path = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")
    file_path = f"{path}/{os.listdir(path)[0]}"
    df = pd.read_csv(file_path)
    
    # Lấy mẫu 30% để tránh tràn RAM Streamlit Cloud
    df, _ = train_test_split(df, train_size=0.3, stratify=df['exit'], random_state=42)
    
    # Feature Engineering
    df['total_product'] = (df['nums_card'] + df['nums_service']).astype('int32')
    df['balance_income_ratio'] = (df['balance'] / (df['monthly_ir'] + 1)).astype('float32')
    
    # Loại bỏ các cột định danh không dùng làm đặc trưng
    # Bao gồm cả 'origin_province' vì nó chứa quá nhiều giá trị text gây nhiễu
    cols_to_drop = ['id', 'full_name', 'address', 'last_active_date', 'created_date', 'origin_province']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # XỬ LÝ ĐỊNH DẠNG (FIX BUG): Chuyển tất cả cột Object sang Category
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Ép kiểu số để tiết kiệm RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
        
    return df

df_clean = load_and_fix_data()

# Chia dữ liệu
X = df_clean.drop('exit', axis=1)
y = df_clean['exit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. HUẤN LUYỆN MÔ HÌNH (SỬ DỤNG NATIVE CATEGORICAL SUPPORT)
# ---------------------------------------------------------
@st.cache_resource
def train_models_robust(_X_train, _y_train):
    # XGBoost: Bật enable_categorical=True để xử lý các cột category
    xgb = XGBClassifier(
        n_estimators=50, 
        learning_rate=0.1, 
        max_depth=5, 
        enable_categorical=True, # QUAN TRỌNG: Sửa lỗi ValueError
        tree_method="hist",      # Tối ưu cho dữ liệu phân loại
        random_state=42, 
        n_jobs=-1
    )
    xgb.fit(_X_train, _y_train)
    
    # LightGBM: Tự động hỗ trợ categorical nếu định dạng cột là 'category'
    lgbm = LGBMClassifier(
        n_estimators=50, 
        learning_rate=0.1, 
        num_leaves=20, 
        verbosity=-1, 
        random_state=42
    )
    lgbm.fit(_X_train, _y_train)
    
    # Stacking (Dùng kết quả của 2 ông trên để Meta-model học)
    estimators = [('xgb', xgb), ('lgbm', lgbm)]
    stacking = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        passthrough=False 
    )
    stacking.fit(_X_train, _y_train)
    
    return xgb, lgbm, stacking

xgb_model, lgbm_model, stack_model = train_models_robust(X_train, y_train)

# ---------------------------------------------------------
# 4. GIAO DIỆN DASHBOARD
# ---------------------------------------------------------
st.sidebar.success("Dữ liệu đã được chuẩn hóa định dạng!")
st.sidebar.write(f"Dòng dữ liệu sử dụng: **{len(df_clean)}**")

tabs = st.tabs(["📈 Hiệu năng", "🧠 Giải thích (SHAP)", "📋 Kết luận"])

with tabs[0]:
    st.header("So sánh chỉ số ROC-AUC")
    models = {"XGBoost": xgb_model, "LightGBM": lgbm_model, "Stacking": stack_model}
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 5))
    metrics_cols = st.columns(3)
    
    for i, (name, model) in enumerate(models.items()):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        score = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {score:.3f})')
        metrics_cols[i].metric(name, f"{score:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("Tỷ lệ Dương tính giả (FPR)")
    ax_roc.set_ylabel("Tỷ lệ Dương tính thật (TPR)")
    ax_roc.legend()
    st.pyplot(fig_roc)

with tabs[1]:
    st.header("Phân tích các yếu tố ảnh hưởng (SHAP)")
    # Giới hạn 30 mẫu để tính cực nhanh trên Cloud
    X_sample = X_test.iloc[:30]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    fig_shap = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig_shap)

with tabs[2]:
    st.header("Giải pháp kinh doanh")
    st.info("Mô hình đã xử lý thành công dữ liệu hỗn hợp (Số & Phân loại) nhờ tính năng Native Categorical của XGBoost.")
    st.markdown("""
    1. **Nhóm rủi ro cao:** Cần chú ý các khách hàng có `risk_segment` là Medium/High kết hợp với số dư tài khoản sụt giảm.
    2. **Đề xuất:** Triển khai ưu đãi cho nhóm `loyalty_level` thấp để tăng tỷ lệ giữ chân (Retention).
    """)