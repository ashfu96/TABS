import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


########################################################
      ######## CARICAMENTO E PREPROCESING ########
########################################################

# FUNZIONE LETTURA DATASET
def load_data(data):
    return pd.read_csv(data, delimiter=" ", header=None)

    
# FILTRO DEL DATASET PER UNIT_ID SELEZIONATA
def filter_by_unit(df , selected_unit_id):
    """
    # Creazione del menu sidebar per la selezione dell'unit_id
    """    
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


########################################################
            ######## PLOT SENSORI ########
########################################################

# PLOT DEI 4 SENSORI TUTTI INSIEME
def plot_selected_columns(df_train, selected_unit_id, selected_columns):
    # Filtro il DataFrame per l'unità selezionata
    df_selected_unit = df_train[df_train['unit_ID'] == selected_unit_id]
    
    # Lista dei colori
    colors = ['b', 'g', 'r', 'c']
       
    # Crea la figura e la griglia per i subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    # Flatten degli array degli assi per indexing più facile
    axs = axs.flatten()
    
    # Plot ogni colonna
    for i, column in enumerate(selected_columns):
        axs[i].plot(df_selected_unit[column].values, color=colors[i % len(colors)], label=column)
        axs[i].set_title('Valore del sensore "{}" per l\' unità con ID "{}"'.format(column, selected_unit_id))
        axs[i].set_xlabel('Cicli effettuati')
        axs[i].set_ylabel('Valore')
        axs[i].legend()
    
    # rimozione subplot inutili
    for i in range(4, 4):
        fig.delaxes(axs[i])    
    # evitare sovrapposizioni
    plt.tight_layout()
    st.pyplot(fig)

 
######### SELEZIONE DEL SINGOLO SENSORE DALLA SIDEBAR
def plot_sensor(df, selected_unit_id, selected_column):
    # Filtra il DataFrame del test per l'unità selezionata
    df_selected_unit = df[df['unit_ID'] == selected_unit_id]

    # Visualizza il grafico della colonna selezionata per l'unità selezionata
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df_selected_unit[selected_column].values, label=selected_column)
    ax.set_title('Valori del sensore "{}" per l\'unità "{}"'.format(selected_column, selected_unit_id))
    ax.set_xlabel('Cicli effettuati')
    ax.set_ylabel('Valore')

    # Utilizza st.pyplot() per visualizzare il grafico all'interno dell'applicazione Streamlit
    st.header(f"Valori del sensore {selected_column} per l\'unità {selected_unit_id}")
    st.pyplot(fig)

#################################################################################    
#           HEALT-INDEX
#################################################################################

#  SLIDER PER IMPOSTAZIONI AVANZATE
def show_sliders():
    st.write("Modifica i pesi dei sensori:")
    weight1 = st.slider('T30 (w)', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    weight2 = st.slider('T50 (w) ', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    weight3 = st.slider('Nc (w)', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    weight4 = st.slider('NRc (w)', min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    return weight1, weight2, weight3, weight4

def calculate_and_plot_health_index(df, unit_id, weights):
    # Verifica che i pesi siano validi
    if len(weights) != 4:
        raise ValueError("weights_list deve avere quattro elementi.")

    # Normalizza le letture dei sensori per ciascuna unità
    df_normalized = df.groupby('unit_ID').transform(lambda x: (x - x.min()) / (x.max() - x.min()))

     # Calcola l'indice di salute
    df['health_index'] = np.dot(df_normalized[['T30', 'T50', 'Nc', 'NRc']], weights)

    # Filtra il DataFrame per l'ID dell'unità specificata
    df_unit = df[df['unit_ID'] == unit_id]

    # Crea una nuova figura
    plt.figure(figsize=(10, 6))

    # Plotta l'indice di salute per l'unità specificata
    plt.plot(df_unit.index, df_unit['health_index'], label=f'Unit {unit_id}')

    # Aggiungi titolo e etichette
    plt.title(f'Health Index nel tempo per l\' Unità nr° {unit_id} (l\'aumento dei parametri mostra la sofferenza del motore)')
    plt.xlabel('Tempo')
    plt.ylabel('Health Index')

    # Aggiungi una legenda
    plt.legend()    
    # Mostra il grafico
    st.pyplot()

########################################################
      ######## HOTELLING T-SQUARE ########
########################################################
   
def plot_hotelling_tsquare(df, selected_unit_id, sensors):

    # Filtra i dati per l'unità selezionata
    unit_data = df[df['unit_ID'] == selected_unit_id]

    # Seleziona le variabili di interesse per l'unità selezionata
    unit_data_selected = unit_data[sensors]
    unit_data_selected.reset_index(drop=True, inplace=True)
    
    # Calcola il vettore medio per le variabili selezionate
    mean_vector = np.mean(unit_data_selected, axis=0)

    # Calcola la matrice di covarianza per le variabili selezionate
    covariance_matrix = np.cov(unit_data_selected.values, rowvar=False)

    # Calcolo dell' Hotelling's T-square per ogni riga nell'unità selezionata (dove np.dot fa il prodotto scalare)
    unit_T_square = np.dot(np.dot((unit_data_selected - mean_vector), np.linalg.inv(covariance_matrix)), (unit_data_selected - mean_vector).T).diagonal()

    return  unit_T_square

def plot_hotelling_tsquare_comparison(df_train, df_test, selected_unit_id, sensors):
    # Crea figura e assi
    fig, ax = plt.subplots()
    # Plot Hotelling's T-square per il training
    unit_T_square_train = plot_hotelling_tsquare(df_train, selected_unit_id, sensors)

    # Plot Hotelling's T-square per il test
    unit_T_square_test = plot_hotelling_tsquare(df_test, selected_unit_id, sensors)
    unit_T_square_test = plot_hotelling_tsquare(df_test, selected_unit_id, sensors)

    # Plot del valore dell Hotelling's T-square e il valore critico
    ax.plot(unit_T_square_train, label="normal data")
    ax.plot(unit_T_square_test, label="actual data")
    ax.set_xlabel('Row Index')
    ax.set_ylabel("Hotelling's T-square")
    ax.set_title(f'Hotelling\'s T-square for Unit ID {selected_unit_id}')
    ax.legend()

    # Mostra il grafico
    st.pyplot(fig)

##############################################################################################
#            PREDIZIONI PRESE COME ULTIMO VALORE DELL'ARRAY    
   
def get_last_sequences_with_predictions(df, sequence_cols, sequence_length, model):
    
    unique_unit_ids = df['unit_ID'].unique()
    predictions = []
    
    for unit_id in unique_unit_ids:
        unit_df = df[df['unit_ID'] == unit_id]
        
        if len(unit_df) >= sequence_length:
            sequence = unit_df[sequence_cols].values[-sequence_length:]
            sequence = np.asarray([sequence])
            prediction = model.predict(sequence)[0]
            predictions.append(prediction)
        else:
            predictions.append(np.nan)  # Add NaN per le predizioni non disponibili
    
    predictions = np.asarray(predictions)
    result_df = pd.DataFrame({'unit_ID': unique_unit_ids, 'prediction': predictions})
    return result_df

#############################################################################
#       DOWNLOAD BUTTON
##############################################################################

@st.cache
def convert_df(df):
    
    #df_copia = df.copy()
    # Effettua il casting a intero dell'ultima colonna del dataframe
    #df_copia.iloc[:, -1] = df_copia.iloc[:, -1].astype(int)  
      
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=',')
    #return df.to_csv.encode('utf-8')


###################################################################
#     FINE
##################################################################








