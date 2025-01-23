import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet  # Actualización del nombre correcto del paquete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import logging
import streamlit as st

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================
# Función para cargar datos
# ==========================
def load_data(file_path, static_path):
    try:
        data = pd.read_csv(file_path, parse_dates=["Date_time"])
        stations = pd.read_csv(static_path)
        logging.info(f"Dataset '{file_path}' loaded successfully with shape {data.shape}")
        
        # Merge datos geográficos
        for station in stations["STATION"]:
            data[f"{station}_LAT"] = stations.loc[stations["STATION"] == station, "LAT"].values[0]
            data[f"{station}_LON"] = stations.loc[stations["STATION"] == station, "LON"].values[0]
        return data
    except Exception as e:
        logging.error(f"Error loading dataset '{file_path}': {e}")
        raise

# ==========================
# Preprocesar datos
# ==========================
def preprocess_data(data):
    logging.info("Starting data preprocessing...")
    data.set_index("Date_time", inplace=True)
    data = data.interpolate(method="time")
    logging.info("Missing data imputed using time-based interpolation.")
    return data

# ==========================
# Crear features
# ==========================
def feature_engineering(data, pm_columns):
    logging.info("Starting feature engineering...")
    lags = [1, 7, 30]
    for lag in lags:
        for col in pm_columns:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    for col in pm_columns:
        data[f"{col}_rolling_mean_7"] = data[col].rolling(window=7).mean()
        data[f"{col}_rolling_std_7"] = data[col].rolling(window=7).std()

    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["year"] = data.index.year

    logging.info("Feature engineering completed.")
    return data.dropna()

# ==========================
# Entrenar modelos con TSCV
# ==========================
def train_models_with_tscv(X, y):
    results = {}
    models = {
        "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15]
        } if name == "XGBRegressor" else {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30]
        }

        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        logging.info(f"{name} - Best Params: {grid_search.best_params_}")

        y_pred = best_model.predict(X)
        results[name] = {
            "model": best_model,
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred)
        }
        logging.info(f"{name} - RMSE: {results[name]['rmse']}, MAE: {results[name]['mae']}, R2: {results[name]['r2']}")

    return results

# ==========================
# Implementar Prophet
# ==========================
def run_prophet(data, station):
    prophet_data = data[["Date_time", station]].rename(columns={"Date_time": "ds", station: "y"})
    model = Prophet()
    model.fit(prophet_data)
    
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    return forecast, model

# ==========================
# Graficar resultados
# ==========================
def plot_results(results, y_test, target_station):
    plt.figure(figsize=(14, 7))
    for name, result in results.items():
        y_pred = result["model"].predict(y_test)
        plt.plot(y_test.index, y_pred, label=f"{name} Prediction")

    plt.title(f"Prediction for {target_station}")
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.show()

# ==========================
# Streamlit App
# ==========================
def build_streamlit_app():
    st.title("Air Pollution Forecasting")

    data = load_data("processed_data_imputed.csv", "estaciones.csv")
    pm_columns = ["BELISARIO_PM25", "CARAPUNGO_PM25", "CENTRO_PM25", "COTOCOLLAO_PM25", "EL_CAMAL_PM25"]
    data = preprocess_data(data)
    data = feature_engineering(data, pm_columns)

    station = st.selectbox("Select a station:", pm_columns)

    st.write("### Model Training")
    X = data.drop(columns=pm_columns)
    y = data[station]
    results = train_models_with_tscv(X, y)
    st.write(results)

    st.write("### Prophet Forecast")
    forecast, _ = run_prophet(data, station)
    st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))

if __name__ == "__main__":
    build_streamlit_app()