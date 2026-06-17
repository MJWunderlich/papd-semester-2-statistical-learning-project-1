"""
Pipeline de preparación de datos HORARIOS para modelado de precios de energía.
Produce data a frecuencia horaria con features sin/cos y sin temperatura.

Uso: python prepare_data_hourly.py
Salida: data/HourlyData-2023.csv
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import re
import datetime as dt

# ─────────────────────────────────────────────────────────────────────────────
# Funciones de features
# ─────────────────────────────────────────────────────────────────────────────

def add_date_features(data):
    data['dia_anual']   = data['dia'].dt.dayofyear
    data['dia_semanal'] = data['dia'].dt.dayofweek
    data['dia_mensual'] = data['dia'].dt.day
    data['mes']         = data['dia'].dt.month
    data['year']        = data['dia'].dt.year
    return data

def add_sinusoidal_date_column(data, phase_shift_days=-60):
    data['dia_anual_norm'] = (2 * np.pi * (phase_shift_days + data['dia_anual'])) / 365
    data['dia_anual_sin']  = np.round(np.sin(data['dia_anual_norm'].values), 6)
    data['dia_anual_cos']  = np.round(np.cos(data['dia_anual_norm'].values), 6)
    data.drop(columns=['dia_anual_norm'], inplace=True)
    return data

def add_sinusoidal_hour_column(data):
    data['hora_norm'] = (2 * np.pi * data['hora']) / 24
    data['hora_sin']  = np.round(np.sin(data['hora_norm'].values), 6)
    data['hora_cos']  = np.round(np.cos(data['hora_norm'].values), 6)
    data.drop(columns=['hora_norm'], inplace=True)
    return data

def add_precipitation_column(data, lat=15.4696, lon=-90.3767):
    date_min = data['dia'].dt.date.min()
    date_max = data['dia'].dt.date.max()

    # Fetch 2 extra years for precip_lag_1y warmup
    start_date = (date_min - dt.timedelta(days=365 * 2)).isoformat()
    end_date   = date_max.isoformat()

    filename = f'data/temperature_{start_date}_{end_date}.json'
    if os.path.exists(filename):
        result = json.load(open(filename))
    else:
        print(f"  Fetching precipitation data ({start_date} → {end_date})...")
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "daily":      "temperature_2m_mean,precipitation_sum",
            "timezone":   "America/Guatemala"
        })
        result = response.json()
        json.dump(result, open(filename, 'w'))

    weather_df = pd.DataFrame({
        'dia':              pd.to_datetime(result['daily']['time']),
        'precipitation_mm': result['daily']['precipitation_sum'],
    })

    weather_df = weather_df.sort_values('dia').reset_index(drop=True)
    weather_df['precip_lag_1y'] = weather_df['precipitation_mm'].shift(365).rolling(365).sum()

    # Merge daily precip onto hourly data
    data = data.merge(weather_df[['dia', 'precip_lag_1y']], on='dia', how='left')
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Limpieza
# ─────────────────────────────────────────────────────────────────────────────

def clean_hours(data):
    view = data[(data['hora'] > 23) | (data['hora'] < 0)]
    if len(view) == 0:
        print("   *** No hay horas fuera de rango ***")
    for i in range(len(view)):
        hora = view.iloc[i]['hora']
        nueva_hora = int(hora / 10 ** np.floor(np.log10(hora) - 1))
        data.loc[view.iloc[i].name, 'hora'] = nueva_hora
        print(f'  Hora original: {hora} → Nueva hora: {nueva_hora}')
    data['hora'] = pd.to_numeric(data['hora'], errors='coerce').astype(int)

def clean_price_values(data):
    view = (
        data['precio'].astype(str).str.contains(r'[^\d.]', regex=True) |
        (data['precio'].astype(str).str.count(r'\.') > 1)
    )
    view = data[view]
    if len(view) == 0:
        print("   *** No hay precios con formato incorrecto ***")
    for i in range(len(view)):
        precio = view.iloc[i]['precio']
        nuevo_precio = re.sub(r'[^\d.]', '', str(precio))
        data.loc[view.iloc[i].name, 'precio'] = float(nuevo_precio)
        print(f'  Precio original: {precio} → Nuevo precio: {nuevo_precio}')
    data['precio'] = data['precio'].astype(float)

def clean_price_outliers(data):
    iq01 = data['precio'].quantile(0.01)
    iq99 = data['precio'].quantile(0.99)
    iq   = data[data['precio'].between(iq01, iq99)]
    outlier_threshold = iq['precio'].max() + iq['precio'].std() * 3
    view = data[data['precio'] >= outlier_threshold]
    if len(view) == 0:
        print("   *** No hay precios outliers claramente erróneos ***")
    for i in range(len(view)):
        idx    = view.iloc[i].name
        precio = view.iloc[i]['precio']
        neighbor = outlier_threshold
        offset   = 1
        while neighbor >= outlier_threshold:
            if idx + offset not in data.index:
                neighbor = data['precio'].median()
                break
            neighbor = data.loc[idx + offset]['precio']
            offset  += 1
        magnitude    = np.round(np.log10(precio / neighbor))
        nuevo_precio = precio / (10 ** magnitude)
        print(f'  Precio: {precio:.2f} → Vecino: {neighbor:.4f} → Magnitud: {magnitude} → Nuevo: {nuevo_precio:.4f}')
        data.loc[idx, 'precio'] = nuevo_precio

def clean_data(data, label):
    print(f"\n=== LIMPIAR DATOS PARA AÑO {label} ===")
    clean_hours(data)
    clean_price_values(data)
    clean_price_outliers(data)
    # Cap outliers at 300
    n_capped = (data['precio'] > 300).sum()
    data['precio'] = data['precio'].clip(upper=300)
    if n_capped > 0:
        print(f"  Capped {n_capped} values above 300 Q/MWh")

def load_eegsa_price_data(file_path, clean=False):
    data = pd.read_excel(file_path)
    data = data.iloc[3:]
    data.columns = ['dia', 'hora', 'precio']
    if clean:
        data['precio'] = (
            data['precio'].astype(str)
            .str.strip()
            .str.replace(r'[^\d.]', '', regex=True)
            .apply(lambda x: pd.to_numeric(x, errors='coerce'))
        )
    data['dia'] = pd.to_datetime(data['dia'], format='%d/%m/%Y')
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('../data', exist_ok=True)

    # 1. Cargar y limpiar 2023 only
    print("\n=== CARGANDO DATOS 2023 ===")
    data = load_eegsa_price_data('../data/Precios2023.xls', clean=True)
    clean_data(data, 2023)
    data = data.reset_index(drop=True)

    print(f"Rows after cleaning: {len(data)}")

    # 2. Features de fecha y hora
    print("\n=== AGREGANDO FEATURES ===")
    data = add_date_features(data)
    data = add_sinusoidal_hour_column(data)
    data = add_sinusoidal_date_column(data, phase_shift_days=-60)

    # 3. Precipitación lagged (no temperatura)
    print("--- Precipitación (Cobán, Alta Verapaz) ---")
    data = add_precipitation_column(data, lat=15.4696, lon=-90.3767)

    # 4. Dummies
    print("\n=== CREANDO DUMMIES ===")
    hora_dummies = pd.get_dummies(data['hora'],        prefix='hora',        drop_first=True).astype(int)
    week_dummies = pd.get_dummies(data['dia_semanal'], prefix='dia_semanal', drop_first=True).astype(int)
    data = pd.concat([data, hora_dummies, week_dummies], axis=1)

    # 5. Drop columns not needed for modeling
    drop_cols = ['dia_anual', 'dia_mensual', 'mes', 'year']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    print(f"\nFinal shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"NaN counts:\n{data.isna().sum()[data.isna().sum() > 0]}")

    # 6. Guardar
    output_path = '../data/HourlyData-2023.csv'
    data.to_csv(output_path, index=False)
    print(f"\n=== GUARDADO: {output_path} ===")
    print(data.head())
