import myfunction
import streamlit as st
#streamlit run ALB/MAIN.py

# IMPORTO DATASET
url_TRAIN = "https://raw.githubusercontent.com/ashfu96/ALB/main/train_FD001.txt"
url_TEST = "https://raw.githubusercontent.com/ashfu96/ALB/main/test_FD001.txt"
url_RUL = "https://raw.githubusercontent.com/ashfu96/ALB/main/RUL_FD001.txt"

df_train, df_test, df_rul = myfunction.read_data_from_github(url_TRAIN, url_TEST, url_RUL)

# RIMUOVO NaN
df_train, df_test, df_rul = myfunction.remove_nan_columns(df_train, df_test, df_rul)

# RINOMINO COLONNE CON LABELS
columns = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

df_train, df_test = myfunction.rename_columns(df_train, df_test, columns)

# RIMOZIONE SENSORI CON DEVIAZIONE STANDARD = 0
train = myfunction.remove_zero_std_columns(df_train)
test = myfunction.remove_zero_std_columns(df_test)

# RIMOZIONE COLONNE CHE NON MI SERVONO ORA
columns_to_remove = ['setting_1', 'setting_2']
train, test = myfunction.remove_columns(train, test, columns_to_remove)

############################# PROVA PRINT TEST PREPROCESS #########################################
st.write("Preprocessed Test Data:")
st.write(test.head())

####### STREAMLIT #######

st.title("Visualizzazione dati sensori per unit_ID")
st.write("Analisi dati delle unità")

# Filtra il DataFrame in base all'unità selezionata
filtered_data = myfunction.filter_by_unit(test)

# Mostra il conteggio dei cicli per l'unità selezionata
results = myfunction.count_cycles_by_unit(filtered_data)
for result in results:
           st.write(result)

# Mostra il plot dell'andamento dei sensori per l'unità selezionata
myfunction.plot_sensor_data(test, filtered_data)

#######################################################

# NORMALIZZAZIONE COLONNE DATASET DI TEST + CREAZIONE cycle_norm
cols_to_exclude = ['unit_ID','time_in_cycles']
df_test_normalized = myfunction.normalize_test_columns(test, cols_to_exclude)
#st.dataframe(df_test_normalized.head(10))
