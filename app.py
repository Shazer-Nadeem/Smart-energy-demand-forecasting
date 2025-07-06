from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from keras.models import load_model
import dm_project as dmp

app = Flask(__name__)

# Load data
raw_data = pd.read_csv('merged_dataset1.csv')
raw_data = dmp.standardize_columns(raw_data)
raw_data = dmp.handle_missing(raw_data)
raw_data = dmp.add_time_features(raw_data)

# Load pre-trained models
MODELS = {
    'rf': joblib.load('random_forest.pkl'),
    'xgb': joblib.load('xgboost.pkl'),
    'linear': joblib.load('linear_regression.pkl'),
    'arima': joblib.load('arima_model.pkl'),
    'ann': joblib.load('ann_model.pkl'),
    'stacked': joblib.load('stacked_model.pkl'),
    'lstm': load_model('lstm_model.h5')
}

@app.route('/')
def dashboard():
    cities = sorted(raw_data['city'].unique().tolist())
    return render_template('index.html', cities=cities)

@app.route('/get_forecast', methods=['POST'])
def get_forecast():
    try:
        city = request.form.get('city')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        lookback = int(request.form.get('lookback', 24))
        model_types = request.form.getlist('model_type')

        mask = (raw_data['city'] == city)
        df = raw_data[mask].copy()
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        if df.empty:
            return jsonify({'error': 'No data available for the selected parameters'})

        feature_cols = ['temperature', 'humidity', 'windSpeed', 'hour', 'day_of_week', 'month', 'season']
        df, scaler = dmp.scale_features(df, [c for c in feature_cols if c in df.columns])
        X = df[feature_cols]
        y = df['demand_mwh']

        predicted_all = {}
        for model_type in model_types:
            if model_type == 'arima':
                model = MODELS['arima']
                preds = model.forecast(steps=len(X))
                predicted_all[model_type] = preds.tolist()
            elif model_type == 'lstm':
                model = MODELS['lstm']
                preds = dmp.lstm_predict(model, X)
                predicted_all[model_type] = preds.tolist()
            elif model_type in MODELS:
                model = MODELS[model_type]
                preds = model.predict(X)
                predicted_all[model_type] = preds.tolist()

        response = {
            'dates': df['timestamp'].astype(str).tolist(),
            'actual': y.tolist(),
            'predicted_all': predicted_all
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_clusters', methods=['POST'])
def get_clusters():
    try:
        city = request.form.get('city')
        n_clusters = int(request.form.get('n_clusters', 4))
        cluster_algo = request.form.get('cluster_algo', 'kmeans')
        cluster_vis = request.form.get('cluster_vis', 'pca')

        mask = (raw_data['city'] == city)
        df = raw_data[mask].copy()
        feature_cols = ['temperature', 'humidity', 'windSpeed', 'demand_mwh', 'hour', 'day_of_week', 'season']
        df, scaler = dmp.scale_features(df, [c for c in feature_cols if c in df.columns])
        X = df[feature_cols]

        if cluster_vis == 'pca':
            coords = dmp.run_pca(X)
        elif cluster_vis == 'tsne':
            coords = dmp.run_tsne(X)
        else:
            coords = df[['temperature', 'humidity']].values
        x_vis, y_vis = coords[:,0], coords[:,1]

        if cluster_algo == 'kmeans':
            labels, _ = dmp.kmeans_clustering(X, n_clusters)
        elif cluster_algo == 'dbscan':
            labels, _ = dmp.dbscan_clustering(X)
        elif cluster_algo == 'hierarchical':
            labels, _ = dmp.hierarchical_clustering(X, n_clusters)
        else:
            labels = np.zeros(X.shape[0])

        sil_score = dmp.compute_silhouette(X, labels)
        df['cluster'] = labels
        cluster_means = df.groupby('cluster')[feature_cols].mean().reset_index().to_dict(orient='records')

        response = {
            'x_vis': x_vis.tolist(),
            'y_vis': y_vis.tolist(),
            'clusters': labels.tolist(),
            'dates': df['timestamp'].astype(str).tolist(),
            'cluster_means': cluster_means,
            'silhouette': sil_score
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_help')
def get_help():
    help_text = """
    <h4>How to Use the Dashboard</h4>
    <ul>
      <li><b>City/Date:</b> Select a city and date range for analysis.</li>
      <li><b>Models:</b> Choose one or more models for forecasting. You can compare their performance using the metrics table.</li>
      <li><b>Clustering:</b> Adjust the number of clusters (k), algorithm, and visualization method. The silhouette score helps evaluate cluster quality.</li>
      <li><b>Results:</b> View interactive plots and tables. Hover for details.</li>
      <li><b>Help:</b> This section provides instructions for all controls.</li>
    </ul>
    """
    return jsonify({'help': help_text})

if __name__ == '__main__':
    app.run(debug=True)
