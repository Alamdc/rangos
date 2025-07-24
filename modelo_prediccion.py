import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinterdnd2 import TkinterDnD, DND_FILES

class SucursalPredictorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predictor de Saldo - Sistema Sucursales CDMX")
        self.geometry("900x700")
        
        # Variables de estado
        self.modelo = None
        self.datos_cargados = None
        
        # Configuración de estilo
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        
        # Crear interfaz
        self.crear_widgets()
    
    def cargar_archivos(self, event=None):
        """Carga archivos CSV arrastrados o seleccionados"""
        file_paths = []
        
        if event:
            # Si viene de drag and drop
            file_paths = [f.strip("{}") for f in event.data.split()]
        else:
            # Si viene del botón de selección
            file_paths = filedialog.askopenfilenames(
                title="Seleccionar archivos CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        
        if not file_paths:
            return
        
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path, parse_dates=['fecha'])
                # Verificar columnas esenciales
                if not all(col in df.columns for col in ['id_sucursal', 'tipo', 'fecha', 'ingresos', 'egresos', 'saldo']):
                    raise ValueError(f"Archivo {Path(file_path).name} no tiene las columnas requeridas")
                dfs.append(df)
            except Exception as e:
                messagebox.showwarning("Advertencia", f"Error al leer {Path(file_path).name}:\n{str(e)}")
                continue
        
        if not dfs:
            messagebox.showerror("Error", "Ningún archivo CSV pudo ser leído")
            return
        
        self.datos_cargados = pd.concat(dfs).sort_values('fecha')
        self.preparar_datos()
        
        messagebox.showinfo("Éxito", f"Datos cargados correctamente\n{len(self.datos_cargados)} registros")
        self.btn_entrenar['state'] = 'normal'
        self.lbl_archivos.config(text=f"Archivos cargados: {len(file_paths)}")

    def preparar_datos(self):
        """Preprocesamiento específico para tus archivos"""
        df = self.datos_cargados.copy()
        
        # Convertir id_sucursal a string para consistencia
        df['id_sucursal'] = df['id_sucursal'].astype(str)
        
        # Variable objetivo (saldo 14 días después)
        df['saldo_futuro'] = df.groupby('id_sucursal')['saldo'].shift(-14)
        
        # Codificar tipo
        df['tipo'] = LabelEncoder().fit_transform(df['tipo'])
        
        # Features adicionales
        df['diff'] = df['ingresos'] - df['egresos']
        df['ratio_ing_egr'] = df['ingresos'] / (df['egresos'] + 1)
        
        # Promedios móviles
        for window in [3, 7, 14]:
            df[f'saldo_rolling_{window}'] = df.groupby('id_sucursal')['saldo'].transform(lambda x: x.rolling(window).mean())
            df[f'ingresos_rolling_{window}'] = df.groupby('id_sucursal')['ingresos'].transform(lambda x: x.rolling(window).mean())
            df[f'egresos_rolling_{window}'] = df.groupby('id_sucursal')['egresos'].transform(lambda x: x.rolling(window).mean())
        
        # Características temporales
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_mes'] = df['fecha'].dt.day
        df['mes'] = df['fecha'].dt.month
        
        # Eliminar NAs
        self.datos_cargados = df.dropna()

    def entrenar_modelo(self):
        """Entrenamiento del modelo con validación"""
        if self.datos_cargados is None:
            messagebox.showerror("Error", "Primero carga los datos")
            return
        
        try:
            features = [
                'tipo', 'ingresos', 'egresos', 'saldo', 'diff', 'ratio_ing_egr',
                'dia_semana', 'dia_mes', 'mes',
                'saldo_rolling_3', 'saldo_rolling_7', 'saldo_rolling_14',
                'ingresos_rolling_3', 'ingresos_rolling_7', 'ingresos_rolling_14',
                'egresos_rolling_3', 'egresos_rolling_7', 'egresos_rolling_14'
            ]
            
            X = self.datos_cargados[features]
            y = self.datos_cargados['saldo_futuro']
            
            # División temporal (80% entrenamiento, 20% prueba)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.modelo = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                early_stopping_rounds=20,
                random_state=42
            )
            
            self.modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            # Evaluación
            pred = self.modelo.predict(X_test)
            mae = np.mean(np.abs(pred - y_test))
            error_relativo = (mae / y_test.mean()) * 100
            
            messagebox.showinfo(
                "Entrenamiento Completado",
                f"Modelo entrenado exitosamente\n"
                f"Error absoluto medio: ${mae:,.2f}\n"
                f"Error relativo: {error_relativo:.2f}%"
            )
            
            self.btn_predecir['state'] = 'normal'
            
        except Exception as e:
            messagebox.showerror("Error", f"Fallo en entrenamiento:\n{str(e)}")

    def predecir_proximo_mes(self):
        """Predicción para el próximo mes"""
        if not self.modelo:
            messagebox.showerror("Error", "Primero entrena el modelo")
            return
        
        try:
            # Obtener los últimos 30 días para la predicción
            ultimos_datos = self.datos_cargados.tail(30).copy()
            
            features = [
                'tipo', 'ingresos', 'egresos', 'saldo', 'diff', 'ratio_ing_egr',
                'dia_semana', 'dia_mes', 'mes',
                'saldo_rolling_3', 'saldo_rolling_7', 'saldo_rolling_14',
                'ingresos_rolling_3', 'ingresos_rolling_7', 'ingresos_rolling_14',
                'egresos_rolling_3', 'egresos_rolling_7', 'egresos_rolling_14'
            ]
            
            X = ultimos_datos[features]
            predicciones = self.modelo.predict(X)
            
            # Mostrar resultados
            self.mostrar_resultados(ultimos_datos['saldo'].values, predicciones)
            
            # Calcular promedio para el próximo mes
            prediccion_promedio = np.mean(predicciones)
            messagebox.showinfo(
                "Predicción",
                f"Saldo promedio predicho para el próximo mes:\n"
                f"${prediccion_promedio:,.2f} MXN"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción:\n{str(e)}")

    def mostrar_resultados(self, real, pred):
        """Gráfico interactivo de resultados"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(real, label='Real', color='blue', marker='o')
        ax.plot(pred, label='Predicción', color='orange', linestyle='--', marker='x')
        ax.set_title('Comparación: Valores Reales vs Predicción')
        ax.set_xlabel('Días')
        ax.set_ylabel('Saldo (MXN)')
        ax.legend()
        ax.grid(True)
        
        # Mostrar en la interfaz
        if hasattr(self, 'frame_grafico'):
            self.frame_grafico.destroy()
            
        self.frame_grafico = ttk.Frame(self)
        self.frame_grafico.place(relx=0.05, rely=0.5, relwidth=0.9, relheight=0.45)
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def crear_widgets(self):
        """Interfaz gráfica con soporte para arrastrar y soltar"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Área de arrastrar y soltar
        drop_frame = ttk.LabelFrame(
            main_frame, 
            text="Arrastra y suelta archivos CSV aquí", 
            relief=tk.RIDGE
        )
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', self.cargar_archivos)
        
        # Configurar el área de drop
        lbl_drop = ttk.Label(
            drop_frame, 
            text="Arrastra uno o más archivos CSV aquí\n\nO", 
            justify=tk.CENTER
        )
        lbl_drop.pack(pady=40)
        
        btn_seleccionar = ttk.Button(
            drop_frame, 
            text="Seleccionar archivos", 
            command=self.cargar_archivos
        )
        btn_seleccionar.pack(pady=10)
        
        self.lbl_archivos = ttk.Label(drop_frame, text="Ningún archivo cargado")
        self.lbl_archivos.pack(pady=10)
        
        # Botones de proceso
        frame_botones = ttk.Frame(main_frame)
        frame_botones.pack(pady=20)
        
        self.btn_entrenar = ttk.Button(
            frame_botones, 
            text="Entrenar Modelo", 
            state=tk.DISABLED,
            command=self.entrenar_modelo
        )
        self.btn_entrenar.pack(side=tk.LEFT, padx=10)
        
        self.btn_predecir = ttk.Button(
            frame_botones, 
            text="Predecir Próximo Mes", 
            state=tk.DISABLED,
            command=self.predecir_proximo_mes
        )
        self.btn_predecir.pack(side=tk.LEFT, padx=10)
        
        # Información
        lbl_info = ttk.Label(
            main_frame,
            text="Instrucciones:\n1. Arrastra o selecciona archivos CSV con los datos\n2. Entrena el modelo\n3. Genera predicción\n\n"
                 "Los archivos deben contener las columnas requeridas: id_sucursal, tipo, fecha, ingresos, egresos, saldo",
            justify=tk.LEFT
        )
        lbl_info.pack(pady=10)

if __name__ == "__main__":
    app = SucursalPredictorApp()
    app.mainloop()