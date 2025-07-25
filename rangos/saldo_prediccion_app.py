import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from datetime import timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Saldo", layout="wide")
st.title("📊 Predicción de Saldo por Sucursal (2 semanas)")

# Función para cargar y procesar datos
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['id_sucursal', 'fecha'])
    return df

# Función para entrenar modelos y hacer predicciones
def train_and_predict(df):
    # Codificación y preparación de características
    le = LabelEncoder()
    df['tipo_encoded'] = le.fit_transform(df['tipo'])
    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['año'] = df['fecha'].dt.year

    features = ['ingresos', 'egresos', 'tipo_encoded', 'dia', 'mes', 'año']
    target = 'saldo'
    predicciones = []

    # Barra de progreso
    progress_bar = st.progress(0)
    total_sucursales = len(df['id_sucursal'].unique())
    
    for i, (id_sucursal, data) in enumerate(df.groupby('id_sucursal')):
        progress_bar.progress((i + 1) / total_sucursales)
        
        if len(data) < 10:
            continue  # evitar entrenar con poca data

        # Determinar tipo y saldo actual
        tipo_predominante = data['tipo'].mode()[0]
        tipo_encoded = 1 if tipo_predominante == 'pagadora' else 0
        saldo_actual = data.iloc[-1]['saldo']

        # Preparar datos de entrenamiento
        X = data[features]
        y = data[target]

        # Entrenar modelo
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Validación cruzada para error
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae = np.mean(np.abs(cv_scores))
        
        # Preparar predicción a 14 días
        last_row = data.iloc[-1]
        future_date = last_row['fecha'] + timedelta(days=14)

        future_data = pd.DataFrame({
            'ingresos': [last_row['ingresos']],
            'egresos': [last_row['egresos']],
            'tipo_encoded': [tipo_encoded],
            'dia': [future_date.day],
            'mes': [future_date.month],
            'año': [future_date.year]
        })

        # Hacer predicción
        pred_saldo = model.predict(future_data)[0]
        margen_error = mae
        porcentaje_error = (margen_error / pred_saldo) * 100 if pred_saldo != 0 else 0
        diferencia = pred_saldo - saldo_actual
        cambio_porcentual = (diferencia / saldo_actual) * 100 if saldo_actual != 0 else 0

        # Guardar resultados
        predicciones.append({
            'id_sucursal': id_sucursal,
            'tipo_sucursal': tipo_predominante,
            'saldo_actual': saldo_actual,
            'fecha_predicha': future_date.strftime("%Y-%m-%d"),
            'saldo_predicho': round(pred_saldo, 2),
            'diferencia': round(diferencia, 2),
            'cambio_%': round(cambio_porcentual, 2),
            'margen_error_absoluto': round(mae, 2),
            'margen_error_%': round(porcentaje_error, 2)
        })

    return pd.DataFrame(predicciones)

# Función para mostrar resultados
def display_results(pred_df):
    # Reordenar columnas
    column_order = ['id_sucursal', 'tipo_sucursal', 'saldo_actual', 'fecha_predicha',
                   'saldo_predicho', 'diferencia', 'cambio_%', 
                   'margen_error_absoluto', 'margen_error_%']
    pred_df = pred_df[column_order]

    # Formatear tabla
    def color_tipo(val):
        color = 'lightcoral' if val == 'pagadora' else 'lightgreen'
        return f'background-color: {color}'
    
    def color_cambio(val):
        if val > 0:
            return 'color: green; font-weight: bold'
        elif val < 0:
            return 'color: red; font-weight: bold'
        return ''
    
    styled_df = pred_df.style.format({
        'saldo_actual': '{:,.2f}',
        'saldo_predicho': '{:,.2f}',
        'diferencia': '{:,.2f}',
        'cambio_%': '{:,.2f}%',
        'margen_error_absoluto': '{:,.2f}',
        'margen_error_%': '{:,.2f}%'
    }).applymap(color_tipo, subset=['tipo_sucursal']
              ).applymap(color_cambio, subset=['diferencia', 'cambio_%'])
    
    st.dataframe(styled_df)

    # Gráfica comparativa
    st.subheader("📈 Gráfica Comparativa: Actual vs Predicho")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(len(pred_df))
    
    bars1 = ax.bar(index - bar_width/2, pred_df['saldo_actual'], bar_width,
                 color='skyblue', label='Saldo Actual')
    
    bars2 = ax.bar(index + bar_width/2, pred_df['saldo_predicho'], bar_width,
                 color='orange', label='Saldo Predicho',
                 yerr=pred_df['margen_error_absoluto'], capsize=5)
    
    ax.set_xlabel('Sucursal')
    ax.set_ylabel('Saldo')
    ax.set_title('Comparación entre Saldo Actual y Saldo Predicho')
    ax.set_xticks(index)
    
    # Etiquetas de ejes corregidas
    labels = []
    for _, row in pred_df.iterrows():
        label = f"{row['id_sucursal']} {str(row['tipo_sucursal'])[0].upper()}"
        labels.append(label)
    ax.set_xticklabels(labels)
    
    ax.legend()
    plt.xticks(rotation=45)
    
    # Añadir etiquetas de valor
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig)

# Interfaz principal
uploaded_file = st.file_uploader("📥 Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    try:
        with st.spinner('Procesando datos...'):
            df = load_data(uploaded_file)
            
            # Validar estructura del archivo
            required_columns = ['id_sucursal', 'tipo', 'fecha', 'ingresos', 'egresos', 'saldo']
            if not all(col in df.columns for col in required_columns):
                st.error("El archivo CSV no tiene la estructura requerida.")
                st.stop()
            
            st.info("Entrenando modelo para cada sucursal...")
            pred_df = train_and_predict(df)
            
            st.subheader("📋 Tabla de Predicciones")
            display_results(pred_df)
            
            st.success("✅ Predicción completada correctamente.")
            
            # Botón para descargar resultados
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📤 Descargar resultados como CSV",
                data=csv,
                file_name='predicciones_saldo.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
else:
    st.info("Por favor, sube un archivo CSV con los campos: id_sucursal, tipo, fecha, ingresos, egresos, saldo.")