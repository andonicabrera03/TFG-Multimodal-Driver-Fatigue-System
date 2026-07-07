"""
00_Analisis_Dataset.py

Script para la auditoría y análisis estadístico del dataset UTA-RLDD.
Recorre la jerarquía de directorios generada tras el preprocesamiento 
(Fold -> Sujeto -> Clase), cuantifica las muestras válidas por cada estado 
de fatiga (KSS) y genera una gráfica de balanceo de clases.

Autor: Andoni Cabrera Fernández
"""

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # =========================================================================
    # CONFIGURACIÓN DEL ENTORNO
    # =========================================================================
    # TODO: Configurar con la ruta local absoluta o relativa del dataset
    ruta_salida_base = r""  
    nombre_grafico = "descripcion_dataset.png"

    # Contadores de muestras por estado fisiológico
    clase_0 = 0   # Alerta base (KSS 1-3)
    clase_5 = 0   # Baja Vigilancia (KSS 6-7)
    clase_10 = 0  # Somnolencia Severa / Micro-sueño (KSS 8-9)

    print("Iniciando auditoría topológica de muestras extraídas...")

    # Validación de seguridad: Comprobar que la ruta configurada existe
    if not ruta_salida_base or not os.path.exists(ruta_salida_base):
        print("\n[ERROR] Directorio no encontrado o no configurado.")
        print("Por favor, verifica la variable 'ruta_salida_base' antes de ejecutar.")
        return

    # =========================================================================
    # EXTRACCIÓN DE MÉTRICAS POBLACIONALES
    # =========================================================================
    # Jerarquía esperada: ruta_salida_base/Fold_X/Sujeto_Y/Clase_Z/*.jpg
    for fold in os.listdir(ruta_salida_base):
        ruta_fold = os.path.join(ruta_salida_base, fold)
        
        if os.path.isdir(ruta_fold):
            for sujeto in os.listdir(ruta_fold):
                ruta_sujeto = os.path.join(ruta_fold, sujeto)
                
                if os.path.isdir(ruta_sujeto):
                    # Contabilización mediante búsqueda de patrones (.jpg)
                    clase_0 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_0", "*.jpg")))
                    clase_5 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_5", "*.jpg")))
                    clase_10 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_10", "*.jpg")))

    total = clase_0 + clase_5 + clase_10

    # =========================================================================
    # RESUMEN ESTADÍSTICO EN CONSOLA
    # =========================================================================
    print("\n" + "=" * 50)
    print(" RESULTADOS DE LA AUDITORÍA DEL DATASET (UTA-RLDD)")
    print("=" * 50)
    print(f" Total Clase 0 (Alerta base):         {clase_0:,}".replace(',', '.'))
    print(f" Total Clase 5 (Baja Vigilancia):     {clase_5:,}".replace(',', '.'))
    print(f" Total Clase 10 (Somnolencia):        {clase_10:,}".replace(',', '.'))
    print("-" * 50)
    print(f" TOTAL DE IMÁGENES VÁLIDAS:           {total:,}".replace(',', '.'))
    print("=" * 50 + "\n")

    # =========================================================================
    # GENERACIÓN DEL GRÁFICO DE DISTRIBUCIÓN
    # =========================================================================
    if total > 0:
        print(f"Generando gráfico de balanceo de clases: '{nombre_grafico}'...")

        clases = ['Clase 0\n(Alerta)', 'Clase 5\n(Baja Vigilancia)', 'Clase 10\n(Somnolencia)']
        cantidades = [clase_0, clase_5, clase_10]

        # Configuración estética con Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Generación del gráfico de barras
        ax = sns.barplot(x=clases, y=cantidades, palette="viridis", hue=clases, legend=False)

        # Inserción dinámica de etiquetas numéricas sobre las barras
        for i, v in enumerate(cantidades):
            ax.text(i, v + (total * 0.01), f"{v:,}".replace(',', '.'), 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Formateo de títulos y ejes
        plt.title('Distribución de Muestras por Nivel de Fatiga', fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
        plt.xlabel('Estado Fisiológico del Conductor', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(nombre_grafico, dpi=300)
        
        print("Proceso finalizado correctamente. Gráfico exportado con éxito.")
    else:
        print("No se han detectado imágenes en los subdirectorios. El gráfico no fue generado.")

if __name__ == "__main__":
    main()