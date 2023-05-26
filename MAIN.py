import myfunction
import streamlit as st
from tensorflow.keras.models import load_model

################
columns = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
sensors = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr','Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd','PCNfR_dmd', 'W31', 'W32']
settings = ['setting_1', 'setting_2','setting_3']
sequence_length = 50 
sequence_cols=sensors + settings
################

#CONFIGURAZIONE PAGINA
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page configuration
st.set_page_config(
    page_title="Dashboard page",
    page_icon=":computer:",
    layout="wide",  # Optional: Set the layout to wide
    initial_sidebar_state="expanded"  # Optional: Expand the sidebar by default
)

st.title("Manutenzione predittiva")
#st.image('https://calaero.edu/wp-content/uploads/2018/05/Airplane-Transponder.jpg',caption='CMAPPS - NASA', use_column_width=False)
st.image('https://www.researchgate.net/publication/348472709/figure/fig1/AS:979966627958790@1610653659534/Schematic-representation-of-the-CMAPSS-model-as-depicted-in-the-CMAPSS-documentation-23.ppm', caption='Turbofan Engine', use_column_width=False)

#################################################################################    
#           CARICAMENTO DATASET
#################################################################################

#TRAIN E RUL CARICATI IN BACKEND
url_TRAIN = "https://raw.githubusercontent.com/ashfu96/TABS/main/train_FD001.txt"
url_RUL = "https://raw.githubusercontent.com/ashfu96/TABS/main/RUL_FD001.txt"
url_TEST = "https://raw.githubusercontent.com/ashfu96/TABS/main/test_FD001.txt"

df_train = myfunction.load_data(url_TRAIN)
df_rul = myfunction.load_data(url_RUL)
comparison_test = myfunction.load_data(url_TEST)

#Rimozione colonne nulle
df_train.dropna(axis=1, inplace=True)
df_rul.dropna(axis=1, inplace=True)
comparison_test.dropna(axis=1, inplace=True)

#labelling colonne
df_train.columns = columns

# TEST CARICATO DALL'UTENTE
# crea il pulsante di caricamento file
test_data_file = st.file_uploader("Carica qui il dataset di test (txt)", type="txt")

# se l'utente ha caricato un file di testo valido
if test_data_file is not None:
    data_file = myfunction.load_data(test_data_file)
    #shape_file = data_file.shape
    
    df_test = data_file.copy()
    
    # rinomino colonne e rimuovo colonne nulle
    df_test.dropna(axis=1, inplace=True)
    df_test.columns = columns
    # rimuovo le colonne setting 1, setting 2 e setting 3 per scopi grafici
    df_no_setting = df_test.copy()
    df_no_setting = df_no_setting.drop(columns=['setting_1', 'setting_2', 'setting_3'])
    
    # show dataset labellato
    st.header("Dataset caricato e labellato:")
    st.dataframe(df_no_setting)
    
    #dimensioni dataset
    shape = df_no_setting.shape
    st.write("Le dimensioni del dataset labellato sono :", shape)
    
    # EXPANDER DATASET ORGINALE
    expander = st.expander("Vedi dataset originale")
    expander.write(data_file)
    shape_file = data_file.shape
    expander.write("Le dimensioni del dataset originale sono :")
    expander.write(shape_file)
   
#################################################################################    
#           SIDEBAR E INFO DOPO LA SELEZIONE UNITà
#################################################################################
    test=df_test
    
    # SIDEBAR PER SELEZIONE UNITà   
    unit_ids = test['unit_ID'].unique()
    # Selezione unit_ID su sidebar
    selected_unit_id = st.sidebar.selectbox('Seleziona unit_ID', unit_ids)
    # Filtra il DataFrame in base all'unità selezionata
    filtered_data = myfunction.filter_by_unit(test,selected_unit_id)

    # CONTEGGIO CICLI EFFETTUATI PER UNITà SELEZIONATA
    results = myfunction.count_cycles_by_unit(filtered_data)
    for result in results:
            #st.header(result)
            st.sidebar.write(result)
            st.sidebar.divider()
            
#################################################################################    
#           PLOT SENSORI CHIAVE
#################################################################################
    
    ### ***   PLOT 4 SENSORI   *** ###
    st.title("Visualizzazione informazioni per l'unità selezionata")
    #st.image('https://www.researchgate.net/publication/348472709/figure/fig1/AS:979966627958790@1610653659534/Schematic-representation-of-the-CMAPSS-model-as-depicted-in-the-CMAPSS-documentation-23.ppm', caption='Turbofan Engine', use_column_width=False)
    st.write("Analisi sensori con deviazione standard più alta")
    
    # Rimuovi le colonne specificate dal DataFrame
    df_dropped = test.drop(['time_in_cycles', 'unit_ID'], axis=1)
    # Calcola la deviazione standard di ogni colonna
    std_dev = df_dropped.std()
    ## Ordina le colonne per deviazione standard, in ordine decrescente
    sorted_columns = std_dev.sort_values(ascending=False)
    # Seleziona i nomi delle prime quattro colonne
    selected_columns = sorted_columns.index[:4]
   
    myfunction.plot_selected_columns(test, selected_unit_id, list(selected_columns))
    
    #EXPANDER LEGENDA SENSORI
    with st.expander("Vedi legenda sensori"):
        st.image("https://github.com/ashfu96/TABS/blob/main/legenda_sensori.png?raw=true") #use_column_width=False

    
    ### ***   PLOT SENSORE SELEZIONATO   *** ###
    st.divider()
    # Crea un menù a tendina nella sidebar per selezionare la colonna da visualizzare
    selected_sensor = st.sidebar.selectbox('Seleziona il sensore da visualizzare', selected_columns)
    st.sidebar.divider()
    # Genera il grafico in base alle selezioni dell'utente
    myfunction.plot_sensor(test, selected_unit_id, selected_sensor)


#################################################################################    
#           HEALT-INDEX
#################################################################################
    st.divider()
    st.header('Health-index dell\' unità')
    st.write("L\'health-index (o indice di salute) ci mostra il deterioramento del motore all\' aumentare dei paramteri")
    
    # AVANZATE SLIDER PESI SENSORI
    st.subheader('Modifica i pesi dei sensori:')
    weight1, weight2, weight3, weight4 = myfunction.show_sliders()

    weights = [weight1, weight2, weight3, weight4]
    test2=test.copy()
    myfunction.calculate_and_plot_health_index(test2, selected_unit_id, weights)
 
####################################################################################
#               HOTELLING T-SQUARE
####################################################################################
    st.title('Analisi statistica multivariata')
    st.write('Confronto generale tra dati normali e dati effettivi.')
    st.write('NOTA: i \" dati normali \" vengono assunti tali e sono i dati di train caricati in backend, i \" dati effettivi \" sono quelli caricati ')
    # NORMALIZZAZIONE COLONNE DATASET DI TEST + CREAZIONE cycle_norm
    cols_to_exclude = ['unit_ID','time_in_cycles']
    df_test_normalized = myfunction.normalize_test_columns(test, cols_to_exclude)
    #st.dataframe(df_test_normalized.head(10))
    
    myfunction.plot_hotelling_tsquare_comparison(df_train, df_test, selected_unit_id, selected_columns)

####################################################################################
#               PREDIZIONE
####################################################################################        
    #st.write(df_test_normalized.shape)
    sequence_columns = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                        'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32', 'setting_1', 'setting_2', 'setting_3','cycle_norm']
    
    # Carica e salva il modello
    model = load_model("model_lstm.h5")
    model.compile(loss='mean_squared_error', optimizer='nadam',metrics=['mae'])
    
    # Prendo le predizioni e le salvo in un dataset 
    result_df = myfunction.get_last_sequences_with_predictions(df_test_normalized, sequence_columns , sequence_length, model)
    result_df2 = result_df.copy()

#############################################################################
#       PREDIZIONE PER SINGOLA UNITà SELEZIONATA
##############################################################################

    st.divider()
    st.title("Predizione dei cicli di vita rimanenti")

    # Seleziona la riga del dataframe corrispondente all'unità selezionata e casting ad intero
    selected_row = result_df2.loc[result_df2['unit_ID'] == selected_unit_id]    
    
    # Verifica se il valore della colonna "prediction" è nullo
    if selected_row['prediction'].isnull().values[0]:
        st.markdown(f"<h1 style='color:red;font-size:32px;font-weight:bold;'>ERRORE: Nessuna predizione disponibile per l'unità selezionata.</h1>", unsafe_allow_html=True)
    else:
        # Estrai il valore della colonna "prediction"
        selected_row['prediction'] = selected_row['prediction'].astype(int)
        prediction_value = selected_row['prediction'].values[0]

        # Stampa il valore utilizzando st.markdown
        st.markdown(f"<h1 style='color:red;font-size:32px;font-weight:bold;'>La predizione per l'unità {selected_unit_id} è di {prediction_value} voli rimanenti.</h1>", unsafe_allow_html=True)
        #st.sidebar.write(f"La predizione per l'unità {selected_unit_id} è di {prediction_value} voli rimanenti.")    
    
       
#############################################################################
#       VISUALIZZAZIONE PREDIZIONI IN SUBSET CON TABS
##############################################################################

    # creazione dataframe con le prediction arrotondate ad intero
    not_null2 = result_df2[result_df2['prediction'].notnull()].copy()
    not_null2["prediction"] = not_null2["prediction"].astype(int)

    subset_df2 = result_df2[result_df2['prediction'].notnull()].copy()
    subset_df2["prediction"] = subset_df2["prediction"].astype(int)

    null2 = result_df2[result_df2['prediction'].isnull()].copy()
    null2["prediction"] = null2["prediction"].fillna('In control')
    
    # divisione in subset in base ai cicli predetti rimanenti, oridinati per unit_ID
    subset_df_part_1 = subset_df2[subset_df2['prediction'] <= 10].sort_values('unit_ID').copy()
    subset_df_part_2 = subset_df2[(subset_df2['prediction'] > 10) & (subset_df2['prediction'] <= 25)].sort_values('unit_ID').copy()
    subset_df_part_3 = subset_df2[subset_df2['prediction'] > 25].sort_values('unit_ID').copy()
    
    
    #       TABS PER LA VISUALIZZAZIONE
    
    st.divider()
    st.header("Vedi altre predizioni:")
    st.write("I cicli di vita rimanenti sono suddivisi in tre range, clicca per visualizzare")
    
    tab1, tab2, tab3, tab4, tab5  = st.tabs(["VEDI TUTTE", "MENO DI 10", "TRA 10 E 25", "SUPERORI A 25", "NON DISPONIBILI"])

    with tab1:
        st.markdown("TUTTE LE PREDIZIONI")
        st.write("Qui sono mostrate tutte le unità e i cicli di vita predetti")
        st.dataframe(not_null2.style.set_caption("Normal condition"))
    
    with tab2:
        st.markdown('<span style="font-size:40px; color:#FFFF00; font-weight: bold;">MANUTENZIONE URGENTE</span>', unsafe_allow_html=True)
        st.write("Qui sono mostrate le unità a cui restano cicli di vita inferiori a 10")
        st.dataframe(subset_df_part_1)
        
    with tab3:
        st.markdown("TRA 10 E 25")
        st.write("Qui sono mostrate le unità a cui restano cicli di vita superiori a 10 e inferiori a 25")
        st.dataframe(subset_df_part_2)

    with tab4:
        st.markdown("SUPERORI A 25")
        st.write("Qui sono mostrate le unità a cui restano cicli di vita superiori a 25")
        st.dataframe(subset_df_part_3)
        
    with tab5:
        st.markdown("PREDIZIONI NON DISPONIBILI")
        st.write("Qui sono mostrate le unità per le quali non è stato possibile effettuare la predizione")
        st.dataframe(null2.style.set_caption("Non disponibile"))
        
     
#############################################################################
#       DOWNLOAD BUTTON
##############################################################################

    st.divider()
    st.header("Scarica il dataset con le predizioni")
    st.write("Clicca su download per scaricare il file in formato .csv con tutte le predizioni effettuate")
    
    #arrotondo le predizioni ad int nel file
    df_csv = result_df.astype(int)
    
    #conversione file in csv e salataggio in una variabile (csv)
    csv = myfunction.convert_df(df_csv)

    #definizione del button
    st.download_button(
        label="Download Predizioni",
        data=csv,
        file_name='LSTM_PREDICTION.csv',
        mime='text/csv',
        key= 'button_one',
    )

# DOWNLOAD ANCHE DA SIDEBAR
    st.sidebar.caption("Scarica il file .csv con le predizioni")

    st.sidebar.download_button(
        label="Download Predizioni",
        data=csv,
        file_name='LSTM_PREDICTION.csv',
        mime='text/csv',
        key= 'button_two',
    )

#############################################################################
#       FINE MAIN
##############################################################################











