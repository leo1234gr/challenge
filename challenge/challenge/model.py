import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance

warnings.filterwarnings('ignore')

# Cargar los datos
data = pd.read_csv('data/data.csv')

# Información del dataset
data.info()

# Función para obtener el periodo del día
def get_period_day(date: str) -> str:
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("04:59", '%H:%M').time()

    if morning_min <= date_time <= morning_max:
        return 'mañana'
    elif afternoon_min <= date_time <= afternoon_max:
        return 'tarde'
    elif evening_min <= date_time <= evening_max or night_min <= date_time <= night_max:
        return 'noche'

# Aplicar la función para obtener el periodo del día
data['period_day'] = data['Fecha-I'].apply(get_period_day)

# Función para determinar si es temporada alta
def is_high_season(fecha: str) -> int:
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
        (fecha >= range2_min and fecha <= range2_max) or
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

# Aplicar la función para determinar si es temporada alta
data['high_season'] = data['Fecha-I'].apply(is_high_season)

# Función para calcular la diferencia en minutos
def get_min_diff(row: pd.Series) -> float:
    fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%dT%H:%M:%S%z')
    fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff
from datetime import datetime

def get_min_diff(row):
    # Ajustar el formato de fecha para que coincida con la cadena de entrada
    fecha_o = datetime.strptime(row['Fecha-0'], )
    fecha_d = datetime.strptime(row['Fecha-D'], )
    return (fecha_d - fecha_o).total_seconds() / 60

# Resto del código...

# Aplicar la función para calcular la diferencia en minutos
data['min_diff'] = data.apply(get_min_diff, axis=1)

# Definir el umbral de retraso en minutos
threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

# Función para obtener la tasa de retraso por columna
def get_rate_from_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()

    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(delays[name] / total * 100, 2)
        else:
            rates[name] = 0

    return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])

# Definición de la clase DelayModel
class DelayModel:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def some_method(self):
        # Implementación del método
        pass

# Clase de prueba para DelayModel
class TestModel:
    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel(param1='value1', param2='value2')

    def test_model_fit(self):
        # Implementación del test
        pass

    def test_model_predict(self):
        # Implementación del test
        pass

    def test_model_preprocess_for_serving(self):
        # Implementación del test
        pass

    def test_model_preprocess_for_training(self):
        # Implementación del test
        pass

# Visualización de la tasa de retraso por destino
def plot_delay_rate_by_destination(data: pd.DataFrame) -> None:
    destination_rate = get_rate_from_column(data, 'SIGLADES')
    destination_rate_values = data['SIGLADES'].value_counts().index
    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(destination_rate_values, destination_rate['Tasa (%)'], alpha=0.75)
    plt.title('Delay Rate by Destination')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Destination', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

# Visualización de la tasa de retraso por aerolínea
def plot_delay_rate_by_airline(data: pd.DataFrame) -> None:
    airlines_rate = get_rate_from_column(data, 'OPERA')
    airlines_rate_values = data['OPERA'].value_counts().index
    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(airlines_rate_values, airlines_rate['Tasa (%)'], alpha=0.75)
    plt.title('Delay Rate by Airline')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Airline', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

# Visualización de la tasa de retraso por mes
def plot_delay_rate_by_month(data: pd.DataFrame) -> None:
    month_rate = get_rate_from_column(data, 'MES')
    month_rate_value = data['MES'].value_counts().index
    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(month_rate_value, month_rate['Tasa (%)'], color='blue', alpha=0.75)
    plt.title('Delay Rate by Month')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 10)
    plt.show()

# Visualización de la tasa de retraso por día de la semana
def plot_delay_rate_by_day(data: pd.DataFrame) -> None:
    days_rate = get_rate_from_column(data, 'DIANOM')
    days_rate_value = data['DIANOM'].value_counts().index
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 5))
    sns.barplot(days_rate_value, days_rate['Tasa (%)'], color='blue', alpha=0.75)
    plt.title('Delay Rate by Day')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('Days', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 7)
    plt.show()

# Visualización de la tasa de retraso por temporada alta
def plot_delay_rate_by_season(data: pd.DataFrame) -> None:
    high_season_rate = get_rate_from_column(data, 'high_season')
    high_season_rate_values = data['high_season'].value_counts().index
    plt.figure(figsize=(5, 2))
    sns.set(style="darkgrid")
    sns.barplot(["no", "yes"], high_season_rate['Tasa (%)'])
    plt.title('Delay Rate by Season')
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel('High Season', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 6)
