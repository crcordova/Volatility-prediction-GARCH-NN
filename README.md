# 🔮 Forecast de Volatilidad con GARCH + LSTM

Esta aplicación permite predecir la **volatilidad futura** de cualquier activo financiero utilizando una combinación de modelos **GARCH(1,1)** y una red neuronal **LSTM**. El modelo se entrena automáticamente con los datos más recientes del activo seleccionado y genera una predicción para los próximos **10 días hábiles**.

## ¿Qué hace esta aplicación?

- Descarga los datos históricos del activo desde Yahoo Finance.
- Calcula la volatilidad diaria mediante un modelo GARCH.
- Entrena una red LSTM para aprender el comportamiento histórico de la volatilidad.
- Predice la volatilidad para los próximos 10 días.
- Visualiza los resultados con gráficos interactivos.
- Permite descargar los resultados en formato CSV.
- Muestra las métricas del modelo, incluyendo el error de entrenamiento (MSE) y la evolución de la pérdida.

---

## ⚙️ Requisitos

- Python 3.8+
- pip

---

## 🚀 Instrucciones para ejecutar la app

1. **Clona el repositorio**:

```bash
git clone https://github.com/tu-usuario/volatility-forecast-app.git
cd volatility-forecast-app
```
2. **Crear enviorement**:
```bash
python -m venv .venv
source .venv/bin/activate     # En Windows: .venv\Scripts\activate
```
3. **Instalar libreriast**:
 ```bash
pip install -r requirements.txt
```
4. **Ejecutar**:
```bash
streamlit run volatility_forecast_app.py
```
