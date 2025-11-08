# Mental Health Status Multi-Class Classification

This repository contains a Natural Language Processing project focused on classifying mental health statuses based on user-generated statements. The dataset is a comprehensive and carefully curated collection sourced from various social platforms, designed to support the development of conversational agents, sentiment analysis systems, and mental health research applications.

---

## 📌 Problem Statement

Understanding mental health expressions in text is essential for early detection, support systems, and automated mental health assistance. However, language surrounding mental well-being is nuanced and contextual. This project aims to:

- Build a robust text classification model capable of identifying mental health status from user statements.
- Aid research related to language patterns in mental health discourse.
- Evaluate model performance using **weighted F1-score** to compensate for class imbalance.

---

## 🧠 Features

The dataset contains the following features:

| Feature Name            | Description                                                                 | Type        |
|-------------------------|-----------------------------------------------------------------------------|-------------|
| `unique_id`             | A unique identifier assigned to each data entry                             | Numerical   |
| `statement`             | The text content or user-generated post being analyzed                      | Text        |
| `status`  | The annotated mental health category label (one of seven classes)           | Categorical |

---

## 📂 Project Structure

```bash
mental-health-sentiment-analysis-nlp-classification/
├── data/
│   ├── raw/                        # Original collected statements dataset
│   │   ├── Combined Data.csv
│   └── cleaned/                  # Cleaned and preprocessed text data
│       ├── data_cleaned.csv
├── notebooks/
│   └── eda.ipynb                   # Setup, EDA, and preprocessing steps
│   └── model.ipynb                 # Model training and evaluation
├── models/
│   └── best_multimodal_mental_bert.pt        # Saved trained classification model
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

## 🔁 Workflow

This project follows a typical machine learning workflow:

1. Data Collection and Preparation
   - Downloaded from Kaggle (see [Dataset & Credits](#-dataset--credits) section).
   - Create train and test set from splitting the data.

2. Data Preprocessing
   - Handled missing values, duplicates, and corrected formatting inconsistencies
   - Performed text cleaning: casefolding, demojizing, removing URLs, special characters, and stopwords.
   - Performed feature engineering and label encoding for `Target`.

3. Exploratory Data Analysis (EDA)
   - Analyzed `Target` distribution.
   - Analyzed word frequency and common phrases in each class.
   - Visualized correlations between features and the target.

4. Model Training
   - Tried multiple regression models: SVM, Random Forest, LightGBM, CatBoost, and XGBoost.
   - Implemented deep learning models: LSTM + GRU and Transformer-based (BERT).

5. Model Evaluation
   - Evaluated models using Weighted F1 Score, appropriate for imbalanced class distributions.
   - Created confusion matrix and detailed classification reports.
   - Best-performing model: **BERT (mental-bert)**.

## 📈 Model Performance

Several classification models were evaluated to categorize user statements into one of the seven mental health status labels.  
Model performance was measured using the **Weighted F1 Score**, which is suitable for imbalanced multi-class classification.  
The summarized results are shown below:

| Model                    | Weighted F1 Score |
|------------------------|------------------|
| Random Forest             | 70.68            |
| CatBoost                  | 77.49            |
| XGBoost                   | 77.67            |
| LightGBM                  | 77.92            |
| SVM                       | 73.46            |
| LSTM + GRU                | 74.38            |
| **BERT (mental-bert)**    | 80.89            |

The **Transformer-based model** achieved the **highest Weighted F1 Score**, demonstrating superior ability to capture semantic and contextual cues in text.  
Therefore, it was selected as the **final model** for inference.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[Mental Health Sentiment Analysis](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

We would like to acknowledge and thank the dataset creator for making this resource publicly available for research and educational use.

---

## 🚀 How to Run

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/abidalfrz/mental-health-status-classification.git
cd mental-health-status-classification
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Virtual Environment as a Kernel (If using Jupyter Notebooks)

```bash
python -m ipykernel install --user --name name-kernel --display-name "display-name-kernel"
```

### 5. Run the Jupyter Notebook

Make sure you have Jupyter installed and select the kernel that you just created, then run the notebooks:

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/models.ipynb
```

You can explore:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Generating final predictions

