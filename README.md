# Smart Energy Demand Forecasting

A data mining and machine learning project for analyzing and forecasting city-level electricity demand. The project combines time series forecasting, clustering, and anomaly detection techniques, all served via an interactive Flask-based dashboard.

## 🚀 Features

- ⚙️ **Preprocessing & Feature Engineering**  
  - Missing value handling
  - Time-based features (hour, day, month, season)
  - Normalization and scaling

- 🔎 **Anomaly Detection**  
  - Z-Score, IQR, Isolation Forest

- 📊 **Clustering & Dimensionality Reduction**  
  - KMeans, DBSCAN, Hierarchical Clustering
  - PCA and t-SNE visualizations
  - Silhouette score evaluation

- 📈 **Forecasting Models**  
  - Linear Regression, Random Forest, XGBoost
  - Stacked Ensemble
  - ARIMA for time series
  - LSTM Neural Network

- 🖥️ **Web Dashboard (Flask)**  
  - City-wise demand forecast using selectable models
  - Interactive clustering visualizations
  - Help and user instructions

## 🗂️ Project Structure

├── app.py # Flask web application
├── dm_project_py.py # Core data mining logic and model training
├── templates/
│ └── index.html # Dashboard front-end (not provided here)
├── models/ # Pre-trained model files (e.g. .pkl, .h5)
├── merged_dataset1.csv # Input dataset (not included in repo)


## 🧠 Requirements

- Python 3.8+
- Flask
- Pandas, NumPy, Scikit-learn
- XGBoost, Keras, TensorFlow
- Matplotlib, Seaborn
- Statsmodels

Install with:

```bash
pip install -r requirements.txt
A sample requirements.txt can be generated via:

bash
Copy
Edit
pip freeze > requirements.txt
📦 Running the App
Place merged_dataset1.csv and pre-trained models in the root or appropriate folders.

Run the Flask app:

bash
Copy
Edit
python app.py
Open http://localhost:5000 in your browser.

##  📊 Sample Results
Forecast visualizations with multiple models

Clustering maps by PCA or t-SNE

Cluster interpretation and feature means

Anomaly flags across demand data

##  📝 Project Goals
Practice data cleaning, preprocessing, and feature engineering

Apply multiple clustering techniques for unsupervised learning

Compare classic and deep learning models for time series forecasting

Build an end-to-end ML dashboard

## 👨‍💻 Author
📘 This project was completed as part of a university Data Mining course.

## 📄 License
This repository is for academic and educational use only.
