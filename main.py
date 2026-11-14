"""
Análisis de radiación UV en Neiva (2001-2024)
Datos fuente: NASA POWER (ALLSKY_SFC_UV_INDEX)
Autor: Proyecto de grado - Programa de Física
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA Y FILTRADO ESTRICTO DE DATOS
# ============================================================================

# Ruta del archivo
archivo_entrada = 'RADIACION UV NEIVA.csv'

def leer_y_filtrar_datos(ruta_archivo):
    """Lee el archivo y filtra estrictamente años 2001-2024"""
    
    # Leer todo el archivo como texto
    with open(ruta_archivo, 'r') as f:
        lineas = f.readlines()
    
    # Buscar la línea donde empiezan los datos
    inicio_datos = 0
    for i, linea in enumerate(lineas):
        if 'PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN' in linea:
            inicio_datos = i
            break
    
    # Leer los datos
    df = pd.read_csv(ruta_archivo, skiprows=inicio_datos)
    
    # Filtrar solo el parámetro UV
    df = df[df['PARAMETER'] == 'ALLSKY_SFC_UV_INDEX'].copy()
    
    # === FILTRADO ESTRICTO: SOLO AÑOS 2001 A 2024 ===
    df = df[(df['YEAR'] >= 2001) & (df['YEAR'] <= 2024)]
    
    # Eliminar filas con valores -999.0
    meses = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    for mes in meses:
        df = df[df[mes] != -999.0]
    
    return df, meses

# Cargar y filtrar datos
df_uv, meses = leer_y_filtrar_datos(archivo_entrada)

print("=" * 60)
print("DATOS CARGADOS Y FILTRADOS")
print("=" * 60)
print(f"Años incluidos: {df_uv['YEAR'].min()} - {df_uv['YEAR'].max()}")
print(f"Número de años: {len(df_uv)}")
print(f"\nAños presentes:\n{df_uv['YEAR'].values}")
print(f"\nPrimeras filas:")
print(df_uv[['YEAR'] + meses[:3]].head())

# ============================================================================
# 2. REESTRUCTURACIÓN A FORMATO LARGO
# ============================================================================

df_largo = pd.melt(
    df_uv, 
    id_vars=['YEAR'], 
    value_vars=meses,
    var_name='MES', 
    value_name='UV_INDEX'
)

mes_a_numero = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
                'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
df_largo['MES_NUM'] = df_largo['MES'].map(mes_a_numero)
df_largo = df_largo.sort_values(['YEAR', 'MES_NUM']).reset_index(drop=True)

# ============================================================================
# 3. CÁLCULO DE PROMEDIOS ANUALES
# ============================================================================

promedios_anuales = df_largo.groupby('YEAR')['UV_INDEX'].mean().reset_index()
promedios_anuales.columns = ['AÑO', 'PROMEDIO_ANUAL']

promedio_total = promedios_anuales['PROMEDIO_ANUAL'].mean()
std_total = promedios_anuales['PROMEDIO_ANUAL'].std()

# Identificar años anómalos
promedios_anuales['ANOMALIA'] = promedios_anuales['PROMEDIO_ANUAL'] - promedio_total
promedios_anuales['TIPO'] = 'Normal'
promedios_anuales.loc[promedios_anuales['ANOMALIA'] > std_total, 'TIPO'] = 'Alto'
promedios_anuales.loc[promedios_anuales['ANOMALIA'] < -std_total, 'TIPO'] = 'Bajo'

print("\n" + "=" * 60)
print("PROMEDIOS ANUALES")
print("=" * 60)
print(f"Promedio total (2001-2024): {promedio_total:.4f}")
print(f"Años anómalos:")
print(promedios_anuales[promedios_anuales['TIPO'] != 'Normal'].to_string(index=False))

# ============================================================================
# 4. TENDENCIA LINEAL
# ============================================================================

x = promedios_anuales['AÑO'].values
y = promedios_anuales['PROMEDIO_ANUAL'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
tendencia_lineal = slope * x + intercept

# ============================================================================
# 5. GRÁFICA DE TENDENCIA ANUAL (CON AÑOS CORRECTOS)
# ============================================================================

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))

# Serie mensual (fondo)
ax.plot(df_largo['YEAR'] + (df_largo['MES_NUM'] - 0.5)/12, 
        df_largo['UV_INDEX'], 
        color='lightgray', linewidth=0.5, alpha=0.5, label='Datos mensuales')

# Calcular desviación estándar mensual por año
std_mensual = df_largo.groupby('YEAR')['UV_INDEX'].std().reset_index()
std_mensual.columns = ['AÑO', 'STD_MENSUAL']
promedios_con_std = promedios_anuales.merge(std_mensual, on='AÑO')

# Barras de error del promedio anual
ax.errorbar(promedios_con_std['AÑO'].values, 
            promedios_con_std['PROMEDIO_ANUAL'].values,
            yerr=promedios_con_std['STD_MENSUAL'].values,
            fmt='o', capsize=5, capthick=2, markersize=8,
            color='black', ecolor='gray', alpha=0.8, 
            label='Promedio anual ± 1σ')

# Resaltar años anómalos
altos = promedios_con_std[promedios_con_std['TIPO'] == 'Alto']
bajos = promedios_con_std[promedios_con_std['TIPO'] == 'Bajo']

ax.scatter(altos['AÑO'], altos['PROMEDIO_ANUAL'], 
           color='red', s=150, zorder=5, label='Años anómalamente altos', 
           edgecolors='darkred', linewidth=2)
ax.scatter(bajos['AÑO'], bajos['PROMEDIO_ANUAL'], 
           color='green', s=150, zorder=5, label='Años anómalamente bajos', 
           edgecolors='darkgreen', linewidth=2)

# Línea de tendencia
ax.plot(x, tendencia_lineal, 'r--', linewidth=2, 
        label=f'Tendencia: {slope:.5f} UVI/año (p={p_value:.3f})')

# Línea del promedio total
ax.axhline(y=promedio_total, color='blue', linestyle=':', linewidth=1.5, 
           label=f'Promedio total: {promedio_total:.3f}')

# ===== CONFIGURACIÓN CORRECTA DEL EJE X =====
# Usar LOS AÑOS DE TUS DATOS como etiquetas
años_unicos = sorted(promedios_anuales['AÑO'].unique())

# Mostrar marcas cada 2 años (para que no se vea tan saturado)
años_a_mostrar = años_unicos[::2]  # [2001, 2003, 2005, ...]
ax.set_xticks(años_a_mostrar)
ax.set_xticklabels([str(a) for a in años_a_mostrar])

# También asegurar que los límites del eje sean correctos
ax.set_xlim(2000.5, 2024.5)

# Etiquetas
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Índice UV (ALLSKY_SFC_UV_INDEX)', fontsize=12)
ax.set_title('Variación de la radiación UV en Neiva (2001-2024)\nDatos NASA POWER', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Anotaciones de los años anómalos
for _, row in altos.iterrows():
    ax.annotate(f'{row["PROMEDIO_ANUAL"]:.2f}', 
                xy=(row['AÑO'], row['PROMEDIO_ANUAL']),
                xytext=(5, 10), textcoords='offset points', 
                fontsize=10, color='red', fontweight='bold')

for _, row in bajos.iterrows():
    ax.annotate(f'{row["PROMEDIO_ANUAL"]:.2f}', 
                xy=(row['AÑO'], row['PROMEDIO_ANUAL']),
                xytext=(5, -15), textcoords='offset points', 
                fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('uv_neiva_tendencia_anual.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico 1 guardado: uv_neiva_tendencia_anual.png")
plt.show()

# ============================================================================
# 6. GRÁFICA DE ESTACIONALIDAD
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

promedio_mensual = df_largo.groupby('MES_NUM')['UV_INDEX'].mean()
std_mensual_global = df_largo.groupby('MES_NUM')['UV_INDEX'].std()

meses_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

ax2.bar(meses_es, promedio_mensual, yerr=std_mensual_global, 
        capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)
ax2.set_xlabel('Mes', fontsize=12)
ax2.set_ylabel('Índice UV promedio (2001-2024)', fontsize=12)
ax2.set_title('Ciclo estacional de la radiación UV en Neiva', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=promedio_total, color='red', linestyle='--', 
            label=f'Promedio anual: {promedio_total:.3f}')
ax2.legend()

plt.tight_layout()
plt.savefig('uv_neiva_estacionalidad.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 2 guardado: uv_neiva_estacionalidad.png")
plt.show()

# ============================================================================
# 7. GUARDAR CSV LIMPIOS
# ============================================================================

# Datos mensuales limpios
df_mensual_limpio = df_largo[['YEAR', 'MES', 'MES_NUM', 'UV_INDEX']].copy()
df_mensual_limpio.columns = ['AÑO', 'MES', 'NUMERO_MES', 'UV_INDEX']
df_mensual_limpio.to_csv('datos_uv_neiva_limpio.csv', index=False, encoding='utf-8')

# Promedios anuales
promedios_anuales.to_csv('promedios_anuales_uv_neiva.csv', index=False, encoding='utf-8')

print("\n✅ Archivos CSV guardados:")
print("   - datos_uv_neiva_limpio.csv")
print("   - promedios_anuales_uv_neiva.csv")

# ============================================================================
# 8. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 60)
print("RESUMEN EJECUTIVO")
print("=" * 60)
print(f"""
Período analizado: 2001 - 2024 (24 años)
Promedio anual total: {promedio_total:.4f}

🔴 AÑOS ANÓMALAMENTE ALTOS:
{altos[['AÑO', 'PROMEDIO_ANUAL', 'ANOMALIA']].to_string(index=False) if len(altos) > 0 else 'Ninguno'}

🟢 AÑOS ANÓMALAMENTE BAJOS:
{bajos[['AÑO', 'PROMEDIO_ANUAL', 'ANOMALIA']].to_string(index=False) if len(bajos) > 0 else 'Ninguno'}

📈 TENDENCIA: {slope:.6f} unidades/año (p={p_value:.4f})
{'→ Tendencia NO significativa' if p_value >= 0.05 else '→ Tendencia significativa'}
""")