import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler

# FUNZIONE PER LETTURA FILE DATASET DA GITHUB

def read_data_from_github(train_url, test_url, rul_url):
    """
    Legge i dati dai file presenti su GitHub e restituisce i relativi dataframes.
    """
    # legge i dati di training dal file su GitHub
    df_train = pd.read_csv(train_url, sep=" ", header=None)

    # legge i dati di test dal file su GitHub
    df_test = pd.read_csv(test_url, sep=" ", header=None)

    # legge i valori RUL dal file su GitHub
    df_rul = pd.read_csv(rul_url, sep=" ", header=None)

    # restituisce i dataframes
    return df_train, df_test, df_rul


# RIMOZIONE COLONNE NaN
def remove_nan_columns(df_train, df_test, df_rul):
    """
    Rimuove le colonne con valori NaN dai dataframes di training e di test e una colonna specifica dal dataframe RUL.
    """
    # rimuove le colonne con valori NaN dal dataframe di training
    df_train.dropna(axis=1, inplace=True)

    # rimuove le colonne con valori NaN dal dataframe di test
    df_test.dropna(axis=1, inplace=True)

    # rimuove la colonna specifica dal dataframe RUL
    df_rul.drop(columns=[1], axis=1, inplace=True)

    # restituisce i dataframes modificati
    return df_train, df_test, df_rul
  
# RINOMINO COLONNE CON LABELS
def rename_columns(df_train, df_test, new_column_names):
    """
    Rinomina le colonne di due dataframe utilizzando una lista di nuove etichette.
    """
    # rinomina le colonne del dataframe di training
    df_train.columns = new_column_names

    # rinomina le colonne del dataframe di test
    df_test.columns = new_column_names

    # restituisce i dataframe con le colonne rinominate
    return df_train, df_test
  
# RIMOZIONE SENSORI CON DEVIAZIONE STANDARD = 0
def remove_zero_std_columns(df):
    """
    Rimuove dal dataframe tutte le colonne che hanno deviazione standard pari a zero.
    """
    std = df.std()
    zero_std_cols = std[std == 0].index.tolist()
    df = df.drop(zero_std_cols, axis=1)
    return df

# RIMOZIONE COLONNE (QUALI COLONNE LO DECIDO DAL MAIN)
def remove_columns(df_train, df_test, columns_to_remove):
    """
    Rimuove le colonne specifiche da due dataframe.
    """
    # rimuove le colonne specifiche dal dataframe di training
    df_train = df_train.drop(columns_to_remove, axis=1)

    # rimuove le colonne specifiche dal dataframe di test
    df_test = df_test.drop(columns_to_remove, axis=1)

    # restituisce i dataframe modificati
    return df_train, df_test
  
############## STREAMLIT ##############

#FILTRO DEL DATASET PER UNIT_ID SELEZIONATA
def filter_by_unit(df):
    """
    # Creazione del menu sidebar per la selezione dell'unit_id
    """
    unit_ids = df['unit_ID'].unique()
    selected_unit_id = st.sidebar.selectbox('Seleziona unit_ID', unit_ids)

    # Filtro del dataframe per la unit_ID selezionata
    filtered_data = df[df['unit_ID'] == selected_unit_id]

    # Restituisce il DataFrame filtrato
    return filtered_data

# CONTEGGIO VOLI EFFETTUATI PER UNIT_ID
def count_cycles_by_unit(df):
    """
    Raggruppa il DataFrame in base all'unit_id e calcola il conteggio dei valori della colonna time_in_cycles per ogni gruppo
    """
    counts = df.groupby('unit_ID')['time_in_cycles'].count()

    # Crea una lista di stringhe di testo che mostrano il conteggio dei time_in_cycles per ogni unit_id
    results = [f"L'unità {unit_id} ha effettuato {count} voli." for unit_id, count in counts.items()]

    # Restituisce la lista di stringhe di testo
    return results


# PLOT PER UNIT_ID L'ANDAMENTO DEI SENSORI NEL TEMPO
def plot_sensor_data(df, filtered_data):

    # creazione del grafico
    fig, ax = plt.subplots(figsize=(10, 6))
    for sensor in df.columns[2:]:
        ax.plot(filtered_data['time_in_cycles'], filtered_data[sensor], label=sensor)
    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Sensor values')
    ax.set_title(f'Sensor data for unit')
    ax.legend()

    # visualizzazione del grafico
    st.pyplot(fig)

"""
# GRAFICO SENSORI MA UTILIZZANDO FUNZIONE DI STREAMLIT ST.LINECHART
def plot_sensor_data(df, filtered_data):
    # creazione del grafico
    sensor_cols = df.columns[2:]
    chart_data = filtered_data.set_index('time_in_cycles')[sensor_cols]
    chart_data.columns = chart_data.columns.str.replace('sensor', 'Sensor ')
    chart_data.columns = chart_data.columns.str.replace('_', ' ')
    chart = st.line_chart(chart_data)
    
    # Imposta manualmente il range sull'asse y
    y_min = filtered_data.iloc[:, 2:].values.min()
    y_max = filtered_data.iloc[:, 2:].values.max()
    y_range = y_max - y_min
    chart.y_axis.set_range(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Aggiungi titoli al grafico
    chart.title('Sensor data for unit')
    chart.xlabel('Time (cycles)')
    chart.ylabel('Sensor values')
    chart.legend(sensor_cols)
    
    # visualizzazione del grafico
    st.write(chart)
"""

############################# PROVA SEQ #######################################

# NORMALIZZAZIONE COLONNE TEST
def normalize_test_columns(df, cols_to_exclude):
    """
    Normalizza le colonne del dataset di test in modo che i valori siano compresi tra 0 e 1 utilizzando il Min-Max Scaler, 
    escludendo le colonne specificate da cols_to_exclude.
    """
    # Crea una copia del DataFrame di test
    df_test = df.copy()
    
    # Crea un oggetto MinMaxScaler
    min_max_scaler = MinMaxScaler()
    
    # Aggiungi una colonna "cycle_norm" con i valori di "time_in_cycles"
    df_test['cycle_norm'] = df_test['time_in_cycles']
    
    # Seleziona le colonne da normalizzare
    cols_to_normalize = df_test.columns.difference(cols_to_exclude + ['unit_ID', 'time_in_cycles'])
    
    # Normalizza le colonne del dataset di test
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(df_test[cols_to_normalize]), 
                                columns=cols_to_normalize,
                                index=df_test.index)
    
    # Combina le colonne normalizzate con le colonne escluse dal processo di normalizzazione
    test_join_df = df_test[df_test.columns.difference(cols_to_normalize)].join(norm_test_df)
    
    # Riordina le colonne del DataFrame
    df_test = test_join_df.reindex(columns=df_test.columns)
    
    # Reimposta l'indice delle righe del DataFrame
    df_test.reset_index(drop=True, inplace=True)
    
    return df_test
