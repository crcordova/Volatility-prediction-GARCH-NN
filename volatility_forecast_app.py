import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("🔮 Predicción de Volatilidad con GARCH + LSTM")

# Input del usuario
ticker = st.text_input("Ingresa el ticker del activo (ej. ^GSPC, AAPL, BTC-USD):", "^GSPC")

if st.button("Predecir Volatilidad"):

    try:
        # Descargar datos
        data = yf.download(ticker, start='2018-01-01')
        returns = 100 * data['Close'].pct_change().dropna()

        # Calcular volatilidad con GARCH
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        garch_vol = garch_result.conditional_volatility

        # Preparar datos para NN
        look_back = 10
        X, y = [], []
        for i in range(look_back, len(garch_vol)):
            X.append(garch_vol[i - look_back:i])
            y.append(garch_vol[i])
        X, y = np.array(X), np.array(y)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        # Modelo LSTM
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=20, batch_size=32, verbose=0)
        history = model.fit(X_scaled, y_scaled, epochs=20, batch_size=32, verbose=0)

         # Calcular métricas
        y_pred_train = model.predict(X_scaled, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred_train)
        y_train_inv = scaler_y.inverse_transform(y_scaled)
        mse = mean_squared_error(y_train_inv, y_pred_inv)

        # Predicción a 10 días
        last_sequence = garch_vol[-look_back:].values
        forecast = []
        for _ in range(10):
            input_seq = scaler_X.transform(last_sequence.reshape(1, -1)).reshape((1, look_back, 1))
            pred_scaled = model.predict(input_seq, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
            forecast.append(pred)
            last_sequence = np.append(last_sequence[1:], pred)

        forecast_dates = pd.date_range(start=garch_vol.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')
        forecast_series = pd.Series(forecast, index=forecast_dates)

        # Graficar con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=garch_vol.index, y=garch_vol, name="GARCH Volatility", line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, name="Forecast (NN)", line=dict(color="orange", dash="dash")))
        fig.update_layout(title=f"Predicción de Volatilidad para {ticker}", xaxis_title="Fecha", yaxis_title="Volatilidad")

        st.plotly_chart(fig, use_container_width=True)

        # Mostrar métrica de desempeño
        st.subheader("📉 Métricas del modelo")
        st.write(f"**MSE en entrenamiento:** {mse:.4f}")

        # Gráfico de loss
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
        loss_fig.update_layout(title="Pérdida (Loss) durante el entrenamiento", xaxis_title="Época", yaxis_title="MSE")
        st.plotly_chart(loss_fig, use_container_width=True)

        # Crear y ofrecer CSV
        st.subheader("📥 Descargar CSV de resultados")
        combined_df = pd.DataFrame(index=garch_vol.index.union(forecast_series.index))
        combined_df['GARCH Volatility'] = garch_vol
        combined_df['Forecast Volatility (NN)'] = forecast_series
        csv = combined_df.to_csv().encode('utf-8')
        st.download_button("Descargar CSV", data=csv, file_name=f"volatility_forecast_{ticker}.csv", mime='text/csv')

        st.success("✅ Predicción completada exitosamente.")

    except Exception as e:
        st.error(f"❌ Error al procesar el activo: {e}")