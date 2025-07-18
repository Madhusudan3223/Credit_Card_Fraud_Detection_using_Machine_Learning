# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning techniques. Given the highly imbalanced nature of the dataset (with very few fraudulent transactions), the model is trained using appropriate resampling and evaluation strategies.

---

## 🎯 Objective

- Build a machine learning model to classify **fraudulent vs. non-fraudulent** transactions  
- Handle **data imbalance** using resampling techniques  
- Evaluate model performance using appropriate metrics like precision, recall, and F1-score

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions  
- **Features**: 30 (Anonymized PCA components + `Amount`, `Time`)  
- **Target**: `Class` (1 = Fraud, 0 = Non-Fraud)  
- ⚠️ **Only 492 fraudulent transactions (~0.17%)**

---

## 🧠 Methodology

1. **Data Exploration & Preprocessing**  
   - Checked class distribution  
   - Scaled the `Amount` and `Time` features using `StandardScaler`

2. **Handling Imbalance**  
   - Used **undersampling** (equal number of fraud and non-fraud samples for training)

3. **Modeling**  
   - Used **Logistic Regression** for binary classification  
   - Trained on balanced data, tested on original data split

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix and ROC Curve

---

## 🔍 Key Results

- **Training Accuracy**: ~94.79%  
- **Test Accuracy**: ~91.87%  
- Strong **Recall** for fraud class (important in real-world fraud detection)  
- Balanced performance using simple logistic regression with minimal computation

---

## 🛠 Libraries Used

- `pandas`, `numpy`  
- `matplotlib`, `seaborn`  
- `scikit-learn` (StandardScaler, LogisticRegression, classification_report)

---

## ✅ Skills Demonstrated

- Working with **imbalanced datasets**  
- Data preprocessing and feature scaling  
- Binary classification using **Logistic Regression**  
- Model evaluation beyond just accuracy (focus on **Recall/F1**)  
- Use of **undersampling** to balance datasets

---

## 📈 Visuals Included

- Class distribution before and after undersampling  
- Confusion matrix  
- Evaluation metrics report (Precision, Recall, F1-Score)

---

## 🧑‍💻 Author

**Madhusudan Mandal**  
📧 madhumandal49@gmail.com  
🔗 [GitHub](https://github.com/Madhusudan3223)

---

## 🌟 Show Your Support

If you liked this project, please ⭐️ the repo and connect with me for collaboration or feedback!

