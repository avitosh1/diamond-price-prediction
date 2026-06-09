# 💎 Diamond Dynamics

## AI-Powered Diamond Price Prediction & Market Segmentation System

Diamond Dynamics is an end-to-end Machine Learning project that combines **Diamond Price Prediction** and **Market Segmentation** into a single interactive Streamlit application. The system uses supervised learning to estimate diamond prices and unsupervised learning to identify meaningful market segments.

---

# 📌 Project Overview

The project was developed to:

- Predict diamond prices based on physical and quality characteristics.
- Compare multiple machine learning algorithms and select the best-performing model.
- Segment diamonds into meaningful market categories using clustering.
- Provide business insights through interactive analytics.
- Deploy the complete solution as a Streamlit web application.

---

# 🎯 Problem Statement

Diamond pricing depends on multiple factors such as:

- Carat
- Cut
- Color
- Clarity
- Dimensions
- Volume

Determining the value of a diamond manually can be difficult and inconsistent.

This project uses Machine Learning to:

1. Predict the expected market price of a diamond.
2. Classify diamonds into market segments.
3. Visualize key patterns and trends for decision-making.

---

# 🛠 Technology Stack

## Programming Language
- Python

## Data Analysis
- Pandas
- NumPy

## Machine Learning
- Scikit-Learn
- XGBoost

## Visualization
- Matplotlib
- Seaborn

## Deployment
- Streamlit

## Model Persistence
- Joblib

---

# 📂 Project Structure

```text
DiamondDynamics/
│
├── data/
│   └── processed_diamonds.csv
│
├── models/
│   ├── best_model.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── best_cluster_model.pkl
│   ├── cluster_encoder.pkl
│   ├── cluster_scaler.pkl
│   ├── cluster_names.pkl
│   ├── best_model_name.pkl
│   └── model_metrics.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── clustering.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Features

| Feature | Description |
|----------|------------|
| carat | Diamond weight |
| cut | Cut quality |
| color | Color grade |
| clarity | Clarity grade |
| depth | Total depth percentage |
| table | Width of the top facet |
| y | Width dimension |
| z | Depth dimension |
| volume | Derived volume feature |
| price_inr | Target variable |

---

# 🤖 Machine Learning Models Evaluated

The following regression models were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors (KNN)
5. XGBoost Regressor

---

# 🏆 Final Model Performance

| Model | MAE | RMSE | R² |
|---------|---------|---------|---------|
| XGBoost | 0.0617 | 0.0842 | 0.9914 |
| Random Forest | 0.0679 | 0.0919 | 0.9897 |
| Decision Tree | 0.0875 | 0.1260 | 0.9807 |
| KNN | 0.0997 | 0.1317 | 0.9789 |
| Linear Regression | 0.1088 | 0.1404 | 0.9761 |

## Selected Model

### XGBoost Regressor

Reasons:

- Highest R² Score
- Lowest RMSE
- Lowest MAE
- Best generalization performance

---

# 🎯 Market Segmentation

K-Means Clustering was applied using:

```python
[
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "y",
    "z",
    "volume"
]
```

## Cluster Evaluation

The optimal number of clusters was selected using:

- Elbow Method
- Silhouette Score Analysis

## Final Segments

### Premium Diamonds
- Higher average prices
- Larger carat sizes
- Luxury market category

### Affordable Diamonds
- Budget-friendly segment
- Lower average prices
- Suitable for mass-market buyers

---

# 🌐 Streamlit Application

## Dashboard

Displays:

- Dataset Summary
- Model Performance
- Best Model Information
- Cluster Overview
- Dataset Preview

## Price Prediction

Features:

- Manual Input
- Sample Diamond Generator
- INR Prediction
- USD Prediction
- Market Segment Prediction

## Analytics

Includes:

- Correlation Heatmap
- Price Distribution
- Carat vs Price Analysis
- Cluster Distribution
- PCA Visualization
- Business Insights

---

# 🔄 System Workflow

```text
User Inputs Diamond Features
            │
            ▼
     Data Preprocessing
            │
            ▼
      XGBoost Prediction
            │
            ├──► Price in INR
            │
            └──► Price in USD
            │
            ▼
      K-Means Clustering
            │
            ▼
     Market Segment Output
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/DiamondDynamics.git
```

```bash
cd DiamondDynamics
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

# 📈 Key Insights

- Carat is the strongest predictor of diamond price.
- Diamond price increases significantly with size.
- XGBoost achieved superior performance compared to other models.
- Market segmentation revealed two natural customer groups.
- Clustering can support inventory planning and marketing strategies.

---

# 🎓 Learning Outcomes

This project demonstrates:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Regression Modeling
- Model Evaluation
- K-Means Clustering
- PCA Visualization
- Streamlit Deployment
- Business Analytics

---

# 🔮 Future Improvements

- Deep Learning Models
- Real-Time Currency Conversion
- Cloud Deployment
- Advanced Customer Segmentation
- Automated Report Generation
- Interactive Business Intelligence Dashboard

---

# 👨‍💻 Author

Diamond Dynamics Project

AI-Powered Diamond Price Prediction & Market Segmentation System

---

# 📄 License

This project is intended for educational and academic purposes.
