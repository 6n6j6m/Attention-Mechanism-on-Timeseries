# 📈 ProjectIndividu: Stock Price Prediction (BBCA)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorch-lightning&logoColor=white)](https://www.pytorchlightning.ai/)

This project is an in-depth study on applying the **Attention Mechanism** to time-series forecasting using BBCA stock closing prices. The primary objective is to transition from experimental research in Jupyter Notebooks to a **modular, production-ready code structure**.

---

## 📂 Project Structure
- `src/`: Core modules for data loading, scaling, and sequencing (Modular).
- `models/`: PyTorch model architecture definitions (`model_build.py`) and `.pth` weight storage.
- `data/`: Historical BBCA stock price dataset (CSV).
- `notebooks/`: Collection of `.ipynb` files used for initial research and prototyping.
- `train.py`: Main script for model training integrated with PyTorch Lightning.
- `predict.py`: Script for evaluation, inference, and result visualization.
- `config.json`: Centralized configuration for hyperparameters, paths, and device settings.

---

## 🧠 Concept & Architecture
The project explores how a hybrid architecture can improve a model's ability to remember long-term context without losing focus on short-term volatility.

### Hybrid Model: CNN-BiLSTM with Attention
The model integrates three main components to capture hierarchical spatial-temporal features:
1.  **CNN (1D Convolutional)**: Used to extract local or spatial patterns within stock price movements.
2.  **BiLSTM (Bidirectional LSTM)**: Captures long-term temporal dependencies by modeling relationships in both forward and backward directions.
3.  **Attention Layer**: A mechanism that dynamically assigns weights to each time step, allowing the model to focus on the most relevant temporal features for prediction.

---

## 📊 Evaluation Metrics
Models are trained using **Grid Search** to identify optimal hyperparameters. Performance is measured using:

1.  **Mean Squared Error (MSE)**:
    $$MSE = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

2.  **Root Mean Squared Error (RMSE)**:
    $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

---

## 💡 Struggles & Findings

During the development process, several challenges provided significant learning opportunities:

### 1. Data Sparsity & Continuity
A major hurdle was the presence of **missing values** within the dataset. This resulted in a non-continuous time sequence, which theoretically deviates from ideal time-series concepts. This underscores the importance of data characteristics, as model performance is heavily influenced by the underlying structure of the asset.

### 2. The Critical Role of Statistical Analysis (EDA)
I recognized a limitation in this project: **skipping the deep Exploratory Data Analysis (EDA)** and time-series statistical testing (such as stationarity, seasonality, or autocorrelation tests). Relying solely on model complexity (like Attention) without fully understanding the data context prevents the model from reaching its optimal potential.

### 3. Effectiveness of the Attention Mechanism
Despite the data challenges, the **Attention Mechanism** proved highly effective in helping the model interpret temporal features. It acts as a dynamic filter, enabling the model to focus on specific time steps with higher significance relative to the target prediction.

### 4. The Concept of Q, K, V
While technical implementations of attention vary across domains, the core concepts of **Query (Q), Key (K), and Value (V)** remain the fundamental bridge connecting various attention-based architectures.

---

## 📝 Future Work
- **Monitoring Platform**: Potentially developing a web application to monitor real-time stock predictions.
- **Advanced Architectures**: Comparing results against Transformer-based approaches or novel attention variants.

---
