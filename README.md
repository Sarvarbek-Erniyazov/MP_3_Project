# 📝 IELTS Writing Score Prediction Project

---

## 🎯 1. Project Overview
This project predicts **IELTS Writing Task scores** automatically using student essays.  
It leverages **NLP** and **machine learning regression models** to estimate the *Overall Score* accurately, reducing human grading effort and increasing consistency.

---

## 🎯 2. Target Column
* **Target:** `Overall`  
* **Description:** Final IELTS score assigned by examiners (0–9).  
* **Importance:** All preprocessing, feature engineering, and modeling pipelines are designed to predict this target.

---

## 📚 3. Dataset Description
* **Source:** [Kaggle – IELTS Writing Scored Essays Dataset](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset)  
* **Size:** 1,435 essays  

| Column | Description |
|--------|-------------|
| 📝 Task_Type | 1 or 2 – essay type |
| ❓ Question | Prompt/question for the essay |
| ✍️ Essay | Student-written essay text |
| 🎯 Overall | Target score (our prediction) |
| 💬 Examiner_Commen | Mostly missing, dropped |
| Sub-scores (`Task_Response`, `Coherence_Cohesion`, `Lexical_Resource`, `Range_Accuracy`) | Completely missing, dropped |

* **Missing Data Analysis:**  

| Column             | Missing Values |
|-------------------|----------------|
| Examiner_Commen    | 1373           |
| Task_Response      | 1435           |
| Coherence_Cohesion | 1435           |
| Lexical_Resource   | 1435           |
| Range_Accuracy     | 1435           |

> ✅ Many columns were mostly empty, so we dropped them. Only `Essay` and `Task_Type` are used as primary features.

---

## ⚠️ 4. Problems & Solutions

| Problem                | Cause                  | Solution                                           |
|------------------------|-----------------------|--------------------------------------------------|
| Missing sub-scores      | Dataset incomplete    | Dropped and focused on `Overall`                |
| Missing examiner comments | Mostly NaN          | Dropped column                                   |
| Text inconsistencies    | Upper/lowercase, punctuation, numbers | Text preprocessing: lowercase, remove punctuation/numbers, remove stopwords |
| Feature extraction      | Raw text not usable for ML | Added `Essay_len` (word count), TF-IDF vectorization |
| High dimensionality     | TF-IDF produces thousands of features | Combined TF-IDF with numeric feature (`Task_Type`) |
| Model evaluation        | Ensure accurate prediction | Train/test split, multiple regression models, best model selection (RandomForest) |
| Model tuning            | Improve performance    | Grid search hyperparameter tuning               |

---

## 🔄 5. Step-by-Step Workflow

### **Step 1: Data Loading**
- Scripts: `src/data_loader.py`, `scripts/load_data.py`  
- Logging-enabled for traceability  
- Example log: ✅ Loaded ielts_writing_dataset.csv | Shape: (1435, 9)

### **Step 2: Data Preprocessing**
- Notebook: `02_preprocessing.ipynb`  
- Dropped unnecessary/missing columns, cleaned text, engineered features, TF-IDF + Task_Type, exported to `.pkl`

### **Step 3: Modeling**
- Notebook: `03_modeling.ipynb`  
- Models: Linear Regression, Random Forest, Gradient Boosting  
- Train/Test: 80/20, Features: (1435 samples, 5001 features)  
- Best model: `RandomForestRegressor` → `models/best_model.pkl`

### **Step 4: Hyperparameter Tuning**
- Notebook: `04_tuning.ipynb`  
- Grid search applied for RandomForest

### **Step 5: Logging System**

| Log File              | Purpose                                |
|-----------------------|----------------------------------------|
| `data_loader.log`     | Dataset loading                        |
| `preprocessing.log`   | Preprocessing & feature extraction     |
| `modeling.log`        | Training, evaluation, best model selection |
| `tuning.log`          | Hyperparameter tuning results          |
| `shap_analysis.log`   | SHAP feature importance analysis       |
| `tune_models.log`     | Hyperparameter tuning logs             |

---

## 🏆 6. Achievements & Insights
* End-to-end ML pipeline implemented
* Data cleaned, preprocessed, vectorized
* Three regression models compared → RandomForest chosen
* Hyperparameter tuning applied successfully
* Logging system integrated for all steps
* Essay length moderately correlates with overall score
* Future improvements: n-grams, embeddings, LSTM/Transformers

---

## ✅ 7. Conclusion
Fully modular, reproducible NLP regression pipeline. Solves missing data, text preprocessing, high-dimensional feature challenges. Produces a logged, traceable, deployable model.

---

## 📂 8. Artifacts / Outputs

| Path                           | Description                               |
|--------------------------------|-------------------------------------------|
| `data/processed/preprocessed_features.pkl` | Preprocessed feature matrix for ML models |
| `data/processed/target_overall.pkl`       | Target variable (Overall)                 |
| `models/best_rf_model.pkl`                 | Best Random Forest model                  |
| `models/best_xgb_model.pkl`                | Best XGBoost model                        |
| `logs/data_loader.log`                     | Dataset loading log                        |
| `logs/preprocessing.log`                   | Preprocessing log                          |
| `logs/modeling.log`                        | Training/evaluation log                     |
| `logs/tuning.log`                          | Hyperparameter tuning log                  |
| `logs/shap_analysis.log`                   | SHAP feature importance log                |
| `logs/tune_models.log`                     | Hyperparameter tuning log                  |
