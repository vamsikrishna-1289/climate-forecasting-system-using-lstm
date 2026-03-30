# 🌍 Climate Forecasting System using LSTM

An AI-powered climate forecasting system that predicts future temperature and precipitation trends using deep learning. This project leverages LSTM (Long Short-Term Memory) networks along with PCA-based dimensionality reduction to model complex climate patterns efficiently.

---

## 🚀 Overview

Climate prediction is a critical challenge due to the temporal and nonlinear nature of environmental data. This system addresses the problem using a sequence-based deep learning approach, enabling accurate forecasting of future climate conditions.

The application includes:

* Data preprocessing pipeline
* LSTM-based prediction model
* PCA for dimensionality reduction
* Flask web interface for interaction

---

## 🧠 Methodology

### 🔹 Data Preprocessing

* Handles missing values using forward fill
* Converts date columns into time-indexed format
* Applies one-hot encoding for categorical features
* Normalizes features using MinMaxScaler
* Reduces dimensionality using PCA

---

### 🔹 Model Architecture

* Sequence-based time series modeling
* Multi-layer LSTM network
* Dropout layers to prevent overfitting
* Dense output layer for multi-feature prediction
* Loss Function: Mean Squared Error (MSE)

---

### 🔹 Prediction Mechanism

* Uses sliding window sequences
* Performs recursive forecasting for future steps
* Applies inverse PCA transformation to restore original scale

---

### 🔹 Web Application

* Upload custom dataset
* Train model dynamically
* Generate predictions via API
* View results interactively

---

## 📊 Outputs

The system generates:

* Historical climate trend analysis
* Actual vs predicted temperature graphs
* Actual vs predicted precipitation graphs
* Future climate forecasts

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras (LSTM)
* Scikit-learn (PCA, Scaling)
* Pandas, NumPy
* Flask (Backend)
* Matplotlib, Seaborn

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

---

## 📌 Applications

* Climate trend forecasting
* Environmental monitoring systems
* Agricultural planning
* Disaster risk prediction
* AI-based weather analytics

---

## 🔮 Future Enhancements

* Transformer-based forecasting models
* Real-time API integration (weather datasets)
* Region-specific predictions
* Cloud deployment (AWS / GCP)
* Interactive dashboards

---

## 👤 Author

**Vamsi Krishna Gondu**

AI Research Aspirant

B.Tech Computer Science and Engineering
(Specialization: Artificial Intelligence & Intelligent Process Automation)

KL University, India

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project with proper attribution.
