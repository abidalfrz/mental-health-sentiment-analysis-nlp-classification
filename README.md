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
| `Statement`             | The text content or user-generated post being analyzed                      | Text        |
| `Mental_Health_Status`  | The annotated mental health category label (one of seven classes)           | Categorical |

---