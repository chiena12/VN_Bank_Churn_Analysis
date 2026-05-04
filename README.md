# 🏦 Vietnam Bank Churn Analysis & Early Warning System
This project integrates T-SQL Analytics with Ensemble Learning to build an automated risk-forecasting system, analyzing behavioral patterns of 80,000 customers in the Vietnamese banking sector.

# 🌟 Key Features

## 📊 Business Intelligence & SQL Analytics:
* Customer Profiling: Executed complex queries on 80,000 records to map exit ratios across diverse demographics.  
* Risk Segmentation: Identified high-risk cohorts, revealing that Gen Z and Mass (Retail) segments exhibit the highest churn rate at 27%.  
* Behavioral Drivers: Analyzed the correlation between 10+ occupation types and 4 wealth segments to pinpoint key churn triggers.  
## 🤖 Advanced Predictive Modeling:
* Ensemble Stacking: Architected a Stacking Classifier combining LightGBM and XGBoost with a Logistic Regression meta-learner to maximize the detection of at-risk customers.  
* Imbalance Mitigation: Optimized model performance for the ~18% minority churn class using class-weight adjustments and AUC-ROC scoring.  
* Hyperparameter Optimization: Implemented Randomized Search CV to fine-tune boosting architectures for stability on large-scale datasets.  
## ⚙️ Quantitative Engineering:
* Feature Engineering: Transformed raw demographic data into high-value economic features such as age_group (Gen Z to Boomers) and asset-based tiers.  
* Data Pipeline: Developed a streamlined workflow spanning Kaggle API data ingestion, T-SQL Exploratory Data Analysis (EDA), and Python-based model deployment.  
## 📈 Analysis Results
* Model Performance: LightGBM was identified as the optimal model for production, providing a superior balance between inference speed and AUC-ROC stability.  
* Risk Metrics: Integrated Recall and Confusion Matrix analysis to ensure the system prioritizes catching potential "exits" before they occur.  
* Strategic Insights: While High-Net-Worth individuals show near-total loyalty, the analysis recommends a strategic pivot toward aggressive retention for the volatile "Mass" and "Potential" segments.  