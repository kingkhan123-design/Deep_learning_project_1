# Deep_learning_project_1

# 🧠 Customer Churn Prediction using Deep Learning

This project uses deep learning techniques to predict customer churn — identifying customers who are likely to stop using a service. Accurate churn prediction helps businesses improve customer retention and reduce revenue loss.

---

## 📊 Project Overview

Churn prediction is a key part of customer relationship management. In this project, we use a customer dataset and apply a deep learning classification model built with TensorFlow/Keras to predict churn.

---

## 🔍 Problem Statement

**Goal:**  
Predict whether a customer will churn based on their demographic, behavioral, and account-related features.

**Why it matters:**  
- Helps businesses proactively retain high-risk customers
- Reduces marketing and acquisition costs
- Enhances customer satisfaction and lifetime value

---

## 📁 Dataset

The dataset includes customer information such as:

| Feature                    | Description                             |
|---------------------------|-----------------------------------------|
| `customerID`              | Unique customer ID                      |
| `gender`                  | Male/Female                             |
| `AGE`                     | Whether the customer is a senior citizen (0/1) |
| `Partner`, `Dependents`   | Marital/family status                   |
| `tenure`                  | Number of months the customer has stayed |
| `PhoneService`, `InternetService`, etc. | Service types            |
| `Contract`, `PaymentMethod` | Subscription and billing methods       |
| `MonthlyCharges`, `TotalCharges` | Financial information            |
| `Churn`                   | Target label (Yes/No)                   |



---

## 🧠 Model Architecture

A deep neural network (DNN) was used for classification.

### Model Summary:

- **Input Layer:** Normalized input features
- **Hidden Layers:** 2–3 Dense layers with ReLU activation
- **Dropout Layers:** To prevent overfitting
- **Output Layer:** Sigmoid activation for binary classification

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
