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

st.title("Predictive maintainace using LSTM (Long-short term memory")
st.image('https://calaero.edu/wp-content/uploads/2018/05/Airplane-Transponder.jpg',caption='CMAPPS - NASA', use_column_width=False)

# CARICAMENTO DATASET
# crea il pulsante di caricamento file
test_data_file = st.file_uploader("Carica qui il dataset di test (txt)", type="txt")

# se l'utente ha caricato un file di testo valido
if test_data_file is not None:
    df_test = myfunction.load_data(test_data_file)

    # visualizza la forma del DataFrame su schermo
    st.write("Dataset caricato:")
    
    # rinomino colonne
    df_test.columns = columns
    st.dataframe(df_test)
   
################################################################################

    st.divider()
    st.title("Visualizzazione dati sensori per unit_ID")
    st.write("Analisi sensori critici")
    test=df_test
    #st.image('https://www.researchgate.net/publication/348472709/figure/fig1/AS:979966627958790@1610653659534/Schematic-representation-of-the-CMAPSS-model-as-depicted-in-the-CMAPSS-documentation-23.ppm', caption='Turbofan Engine', use_column_width=False)
        
# SIDEBAR PER SELEZIONE UNITà   
    unit_ids = test['unit_ID'].unique()
    # Selezione unit_ID su sidebar
    selected_unit_id = st.sidebar.selectbox('Seleziona unit_ID', unit_ids)
    # Filtra il DataFrame in base all'unità selezionata
    filtered_data = myfunction.filter_by_unit(test,selected_unit_id)

# CONTEGGIO CICLI EFFETTUATI PER UNITà SELEZIONATA
    results = myfunction.count_cycles_by_unit(filtered_data)
    for result in results:
            st.header(result)

    # Drop the specified columns
    df_dropped = test.drop(['time_in_cycles', 'unit_ID'], axis=1)
    # Calculate the standard deviation of each column
    std_dev = df_dropped.std()
    # Sort the columns by their standard deviation, in descending order
    sorted_columns = std_dev.sort_values(ascending=False)
    # Get the names of the first four columns
    selected_columns = sorted_columns.index[:4]
    
    st.divider()
    myfunction.plot_selected_columns(test, selected_unit_id, list(selected_columns))

    st.write('Health analysis of the engine') 

# BOTTONE AVANZATE
if st.button('Impostazioni avanzate'):
    myfunction.show_sliders()

    weights = [weight1, weight2, weight3, weight4]
    test2=test.copy()
    myfunction.calculate_and_plot_health_index(test2, selected_unit_id, weights)
    
    
    
    
             
    st.title('Multivariate statistical analysis')
    st.write('overall comparison between normal and actual data')
    # NORMALIZZAZIONE COLONNE DATASET DI TEST + CREAZIONE cycle_norm
    cols_to_exclude = ['unit_ID','time_in_cycles']
    df_test_normalized = myfunction.normalize_test_columns(test, cols_to_exclude)
    #st.dataframe(df_test_normalized.head(10))
    myfunction.plot_hotelling_tsquare_comparison(df_train, df_test, selected_unit_id, selected_columns)
    

    
    

    st.write(df_test_normalized.shape)
    sequence_columns = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                        'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32', 'setting_1', 'setting_2', 'setting_3','cycle_norm']
    # Load the saved model
    model = load_model("model_lstm.h5")
    model.compile(loss='mean_squared_error', optimizer='nadam',metrics=['mae'])
    
    
    
    st.title("Prediction of Remain useful life")
    # Assuming you have a DataFrame called df_test
    result_df = myfunction.get_last_sequences_with_predictions(df_test_normalized, sequence_columns , sequence_length, model)
    
    
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

    
    # Display the first part of the first dataset in the first column
    with col1:
        st.markdown("")
        st.dataframe(subset_df_part1.style.set_caption(""))

    # Display the second part of the first dataset in the second column
    with col2:
        st.markdown("")
        st.dataframe(subset_df_part2.style.set_caption(""))

    # Display the third part of the first dataset in the third column
    with col3:
        st.markdown("")
        st.dataframe(subset_df_part3.style.set_caption(""))
    # Display the fourth part in the fourth column
    with col4:
        st.markdown("")
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
