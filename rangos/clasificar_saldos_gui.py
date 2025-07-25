import pandas as pd
from tkinter import Tk, filedialog, messagebox, ttk, Frame, Scrollbar, BOTH, RIGHT, LEFT, Y

# Rangos por tipo de sucursal
rangos = {
    "captadora": {
        "A": (10_000, 35_000),
        "B": (15_000, 50_000),
        "C": (25_000, 80_000),
        "D": (40_000, 140_000),
        "E": (60_000, 300_000)
    },
    "pagadora": {
        "A": (15_000, 40_000),
        "B": (30_000, 80_000),
        "C": (60_000, 160_000),
        "D": (120_000, 300_000),
        "E": (240_000, 600_000)
    }
}

# Asignar rango a saldo predicho
def asignar_rango(tipo, saldo_predicho):
    tipo = tipo.lower()
    if tipo not in rangos:
        return "Tipo desconocido"
    for rango, (minimo, maximo) in rangos[tipo].items():
        if minimo <= saldo_predicho <= maximo:
            return rango
    return "Fuera de rango"

# Evaluar necesidad de transferencia
def necesita_transferencia(tipo, saldo_actual, saldo_predicho):
    tipo = tipo.lower()
    if saldo_actual < 0:
        return "Sí, urgente"
    if tipo == "captadora" and saldo_predicho > saldo_actual:
        return "Sí"
    elif tipo == "pagadora" and saldo_predicho < saldo_actual:
        return "Sí"
    return "No"

# Mostrar los datos en tabla visual
def mostrar_tabla(df):
    ventana = Tk()
    ventana.title("Sucursales - Análisis de Transferencia de Efectivo")

    marco = Frame(ventana)
    marco.pack(fill=BOTH, expand=True)

    columnas = ["id_sucursal", "tipo_sucursal", "saldo_actual", "saldo_predicho", "rango_predicho", "necesita_transferencia"]

    tree = ttk.Treeview(marco, columns=columnas, show='headings')
    for col in columnas:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=130)

    scrollbar = Scrollbar(marco, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=RIGHT, fill=Y)
    tree.pack(side=LEFT, fill=BOTH, expand=True)

    for _, row in df.iterrows():
        tree.insert("", "end", values=[
            row["id_sucursal"],
            row["tipo_sucursal"],
            row["saldo_actual"],
            row["saldo_predicho"],
            row["rango_predicho"],
            row["necesita_transferencia"]
        ])

    ventana.mainloop()

# Cargar archivo y procesar
def procesar_archivo():
    root = Tk()
    root.withdraw()

    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo CSV",
        filetypes=[("Archivos CSV", "*.csv")]
    )

    if not archivo:
        messagebox.showinfo("Sin archivo", "No se seleccionó ningún archivo.")
        return

    try:
        df = pd.read_csv(archivo)

        # Verificación de columnas
        columnas_necesarias = ["id_sucursal", "tipo_sucursal", "saldo_actual", "saldo_predicho"]
        for col in columnas_necesarias:
            if col not in df.columns:
                raise ValueError(f"Falta la columna '{col}' en el archivo CSV.")

        # Clasificación y evaluación
        df["rango_predicho"] = df.apply(
            lambda row: asignar_rango(row["tipo_sucursal"], row["saldo_predicho"]), axis=1
        )
        df["necesita_transferencia"] = df.apply(
            lambda row: necesita_transferencia(row["tipo_sucursal"], row["saldo_actual"], row["saldo_predicho"]), axis=1
        )

        mostrar_tabla(df)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo procesar el archivo:\n{str(e)}")

if __name__ == "__main__":
    procesar_archivo()

