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

st.title("Manutenzione predittiva tramite LSTM (Long-short term memory)")
#st.image('https://calaero.edu/wp-content/uploads/2018/05/Airplane-Transponder.jpg',caption='CMAPPS - NASA', use_column_width=False)

#################################################################################    
#           CARICAMENTO DATASET
#################################################################################

#TRAIN E RUL CARICATI IN BACKEND
url_TRAIN = "https://raw.githubusercontent.com/ashfu96/ALB/main/train_FD001.txt"
url_RUL = "https://raw.githubusercontent.com/ashfu96/ALB/main/RUL_FD001.txt"
url_TEST = "https://raw.githubusercontent.com/ashfu96/ALB/main/test_FD001.txt"

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
    
    # show dataset labellato
    st.header("Dataset caricato e labellato:")
    st.dataframe(df_test)
    
    #dimensioni dataset
    shape = df_test.shape
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
    expander = st.expander("Vedi legenda sensori")
    expander.write("Il sensore Nc misura la velocità fisica del nucleo motore")
    expander.write("Il sensore NRc misura la velocità corretta del nucleo motore")
    expander.write("Il sensore T30 misura la temperatura totale all'uscita del compressore ad alta pressione (HPC)")
    expander.write("Il sensore T50 misura la temperatura totale all'uscita dalla turbina a bassa pressione (LPT)")
    
    ### ***   PLOT SENSORE SELEZIONATO   *** ###
    st.divider()
    # Crea un menù a tendina nella sidebar per selezionare la colonna da visualizzare
    selected_sensor = st.sidebar.selectbox('Seleziona il sensore da visualizzare', selected_columns)
    # Genera il grafico in base alle selezioni dell'utente
    myfunction.plot_sensor(test, selected_unit_id, selected_sensor)


#################################################################################    
#           HEALT-INDEX
#################################################################################
    st.divider()
    st.header('Health-index dell\' unità')
    st.write("L\'health-index (o indice di salute) ci mostra il deterioramento del motore all\' aumentare dei paramteri")
    
    # EXPANDER AVANZATE PESI SENSORI
    with st.expander("Avanzate"):
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
    
    # Load the saved model
    model = load_model("model_lstm.h5")
    model.compile(loss='mean_squared_error', optimizer='nadam',metrics=['mae'])
    
    st.divider()
    st.title("Prediction of Remain useful life")
    
#############################################################################
# PROVA TABS #
###############################################################################

    # Assuming you have a DataFrame called df_test
    result_df = myfunction.get_last_sequences_with_predictions(df_test_normalized, sequence_columns , sequence_length, model)
    result_df2 = result_df.copy()
    
    not_null = result_df[result_df['prediction'].notnull()].copy()
    not_null["prediction"]=not_null["prediction"].astype(int)
    not_null = not_null[not_null['prediction'] >= 20].copy()
    
    subset_df = result_df[result_df['prediction'].notnull()].copy()
    subset_df["prediction"]=subset_df["prediction"].astype(int)
    subset_df = subset_df[subset_df['prediction'] < 20].copy()
    
    null = result_df[result_df['prediction'].isnull()].copy()
    null["prediction"]=null["prediction"].fillna('In control')
    
    
    # Calculate the size of each part
    total_rows = len(subset_df)
    part_size = total_rows // 4
    remaining_rows = total_rows % 4
    subset_df = subset_df.sort_values('prediction')
    
    # Split the dataset into four parts
    subset_df_part1 = subset_df[:part_size]
    subset_df_part2 = subset_df[part_size:part_size*2]
    subset_df_part3 = subset_df[part_size*2:part_size*3]
    subset_df_part4 = subset_df[part_size*3:part_size*3+remaining_rows]

    st.title("Urgent maintainance")
    # Create columns to display the dataset parts side by side
    col1, col2, col3, col4 = st.beta_columns(4)

    #############################################################################
    # PROVA TABS #
    ###############################################################################

    tab1, tab2, tab3, tab4 = st.tabs(["COL1", "COL2", "COL3", "COL4"])    
    
    # Display the first part of the first dataset in the first column
    with tab1:
    #with col1:
        st.markdown("COL1")
        st.dataframe(subset_df_part1.style.set_caption(""))

    # Display the second part of the first dataset in the second column
    with tab2:
    #with col2:
        st.markdown("COL2")
        st.dataframe(subset_df_part2.style.set_caption(""))

    # Display the third part of the first dataset in the third column
    with tab3:
    #with col3:
        st.markdown("COL3")
        st.dataframe(subset_df_part3.style.set_caption(""))
    
    # Display the fourth part in the fourth column
    with tab4:
    #with col4:
        st.markdown("COL4")
        st.dataframe(subset_df_part4.style.set_caption(""))
            
    
    # Create columns to display the datasets side by side
    col1_, col2_ = st.beta_columns(2)
    
    # Display the second dataset in the second column
    with col1_:
        st.markdown("")
        st.dataframe(not_null.style.set_caption("Normal condition"))

    # Display the third dataset in the third column
    with col2_:
        st.markdown("")
        st.dataframe(null.style.set_caption("In control"))

################################################################
# TAB 2

    not_null2 = result_df2[result_df2['prediction'].notnull()].copy()
    not_null2["prediction"] = not_null2["prediction"].astype(int)

    subset_df2 = result_df2[result_df2['prediction'].notnull()].copy()
    subset_df2["prediction"] = subset_df2["prediction"].astype(int)

    null2 = result_df2[result_df2['prediction'].isnull()].copy()
    null2["prediction"] = null2["prediction"].fillna('In control')

    subset_df_part_1 = subset_df2[subset_df2['prediction'] <= 10].sort_values('prediction').copy()
    subset_df_part_2 = subset_df2[(subset_df2['prediction'] > 10) & (subset_df2['prediction'] <= 25)].sort_values('prediction').copy()
    subset_df_part_3 = subset_df2[subset_df2['prediction'] > 25].sort_values('prediction').copy()
    
    prova1, prova2, prova3  = st.tabs(["less_than_10", "between_11_25", "greater_than_25"])

    with prova1:
        st.markdown("less_than_10")
        st.dataframe(subset_df_part_1)
        
    with prova2:
        st.markdown("between_11_25")
        st.dataframe(subset_df_part_2)

    with prova3:
        st.markdown("greater_than_25")
        st.dataframe(subset_df_part_3)
