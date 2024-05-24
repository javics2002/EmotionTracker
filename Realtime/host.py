import socket
import threading
import time
import pandas as pd
import mne
import joblib
import numpy as np
from scipy import signal
import math
import copy

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

exit = False
record = False
endedChunk = False
emotibit_data_list = []
dataframe_eeg = None
fif_saved = False
chunk_time = 10
SAMPLE_RATE = 250 # En Hz
chunk_size = math.floor(chunk_time * SAMPLE_RATE)

emotibit_raw_columns = ['time', 'PacketNumber', 'DataLength', 'TypeTag', 'ProtocolVersion', 'DataReliability', 'Data']
emotibit_final_columns = ['time', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'T1', 'MX', 'MY', 'MZ', 'HR', 'BI', 'PI', 'PR', 'PG', 'EA', 'EL', 'ER', 'SA', 'SR', 'SF']

def listen_eeg():
    global dataframe_eeg, chunk_time, endedChunk, fif_saved, record, chunk_time

    eeg = acquisition.EEG()

    # Mapeo de índices de electrodos a nombres
    cap = {
        0: "P8", 1: "O2", 2: "P4", 3: "C4", 4: "F8", 5: "F4", 6: "Oz", 7: "Cz",
        8: "Fz", 9: "Pz", 10: "F3", 11: "O1", 12: "P7", 13: "C3", 14: "P3", 15: "F7"
    }
    

    with EEGManager() as mgr:
        # Configurar el puerto Bluetooth o serie según el sistema operativo
        eeg.setup(mgr, port='COM4', cap=cap)

        while not exit:
            while record:

                start_time = time.time()
                print("start time eeg: " + str(start_time))
                # Iniciar adquisición de datos
                eeg.start_acquisition()
                
                time.sleep(chunk_time)

                print("end time " + str(time.time() - start_time))

                # get all eeg data and stop acquisition
                eeg.get_mne()
                eeg.stop_acquisition()

                # save EEG data to MNE fif format
                eeg.data.save('raw.fif')

                # Close brainaccess library
                eeg.close()

                # PASAR .FIF A CSV

                # Cargar los datos EEG desde el archivo .fif
                raw = mne.io.read_raw_fif('raw.fif', preload=True)

                # Obtener los datos EEG
                data = raw.get_data()

                # Obtener la información del mapeo de electrodos
                mapping = raw.ch_names

                # Convertir los datos en un DataFrame de Pandas
                dataframe_eeg = pd.DataFrame(data.T, columns=mapping)

                print("----------------------------------------------------------------------------------------------------")

                fif_saved = True


def listen_client():
    global exit, record, endedChunk, emotibit_data_list, chunk_time, emotibit_dataframe_preprocessed, eeg_dataframe_preprocessed, fif_saved, dataframe_eeg

    server_socket = socket.socket()
    port = 5000  # Puedes cambiar el puerto si es necesario
    server_socket.bind(('0.0.0.0', port))
    
    # Escucha hasta 1 conexión
    server_socket.listen(10)

    print(f"Servidor escuchando en el puerto {port}...")

    conn, address = server_socket.accept()
    print(f"Conexión de {address} establecida.")
    
    record = True

    startTime = time.time()
    print(startTime)

    # Recibe mensajes del cliente
    while not exit:
        while not endedChunk and record:
            if(time.time() - startTime >= chunk_time):
                conn.send('stop'.encode())
                endedChunk = True
                record = False
                fif_saved = False

                realtime_data = conn.recv(2000000)

                #print(realtime_data)
                print(time.time())

                realtime_data = realtime_data.decode('utf-8')

                # Dividir la cadena utilizando el punto y coma ';' como separador
                realtime_data_list = realtime_data.split(';')

                # Convertir cada elemento de la lista de nuevo a un objeto de bytes
                emotibit_data_list = [dato.encode('utf-8') for dato in realtime_data_list]

                while(not fif_saved):
                    time.sleep(0.1)

                emotibit_data_list_copy = copy.deepcopy(emotibit_data_list)
                dataframe_eeg_copy = copy.deepcopy(dataframe_eeg)
                
                # Crear hilo que preprocesa los datos, hace la prediccion y lo manda a unreal
                preprocess_predict_and_send_thread = threading.Thread(target=preprocess_predict_and_send, args=(emotibit_data_list_copy, dataframe_eeg_copy))
                preprocess_predict_and_send_thread.start()

                #Reseteo de variables
                startTime = time.time()
                record = True
                endedChunk = False
                fif_saved = False

                realtime_data = []
                emotibit_data_list = []
                dataframe_eeg = None

                conn.send('record'.encode())
    
    # Cierra la conexión
    conn.close()

def bytes_to_dataframe(byte_list):
    # Inicializa una lista vacía para almacenar las filas
    rows = []

    # Convierte cada elemento de la lista de bytes a cadena de texto y luego a una lista de campos
    for byte_data in byte_list:
        # Convierte bytes a cadena de texto
        str_data = byte_data.decode('utf-8')
        # Divide la cadena de texto en campos usando la coma como separador
        fields = str_data.split(',')
        # Añade los campos a la lista de filas
        rows.append(fields)

    # Crea un DataFrame de pandas a partir de la lista de listas de campos
    df = pd.DataFrame(rows)
    
    # Devuelve el DataFrame
    return df

def get_emotibit_raw(emotibit_dataframe):
    
    i = 0
    for emotibit_raw_column in emotibit_raw_columns:
        emotibit_dataframe = emotibit_dataframe.rename(columns={i: emotibit_raw_columns[i]})
        i += 1

    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'AK'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'RB'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'TL'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'RD'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'TH'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'EM'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'DC'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'DO'].index)
    emotibit_dataframe = emotibit_dataframe.drop(emotibit_dataframe[emotibit_dataframe['TypeTag'] == 'ER'].index)


    if emotibit_dataframe['time'].dtype == 'object':
        emotibit_dataframe['time'] = emotibit_dataframe['time'].astype(float)

    if emotibit_dataframe['DataLength'].dtype == 'object':
        emotibit_dataframe['DataLength'] = emotibit_dataframe['DataLength'].astype(int)

    EmotibitStartTime = emotibit_dataframe['time'].iloc[0]

    emotibit_dataframe['time'] = emotibit_dataframe['time'].apply(lambda x:  x - EmotibitStartTime)   
    emotibit_dataframe['time'] = emotibit_dataframe['time'].apply(lambda x:  x/1000)
    # Agrupar por el valor de tiempo (columna 3)
    time_groups = emotibit_dataframe.groupby('time')

    final_emotibit_dataframe_data = pd.DataFrame(columns=emotibit_final_columns)

    ### ITERADOR DE GRUPOS
    for tiempo, grupo in time_groups:
        tempDataFrame = pd.DataFrame(columns=emotibit_final_columns)

        contador = 0
        ### ITERADOR DE FILAS
        for index, row in grupo.iterrows():
            
            # Obtener el valor de la fila en la columna actual
            groupTag = grupo['TypeTag'][index]

            numberOfPackets = grupo['DataLength'][index]
            
            # Calcular la media para la primera fila y columnas especificadas
            groupValueDf = grupo.iloc[contador, 6:(6 + numberOfPackets)]
            groupValueDf = groupValueDf.astype(float)
            groupValue = groupValueDf.mean() 

            contador += 1

            # Asignar time
            tempDataFrame.at[0, 'time'] =  grupo['time'][index]

            ### ITERADOR PARA SABER EL TAG
            for column in tempDataFrame.columns:
                if groupTag == column:
                    tempDataFrame.at[0, column] = groupValue

        # pasar tempDataFrame a finalDataFrame
        final_emotibit_dataframe_data = pd.concat([final_emotibit_dataframe_data, tempDataFrame], ignore_index=True)   
        
        
    final_emotibit_dataframe_data_raw = final_emotibit_dataframe_data.sort_values(by='time')

    return final_emotibit_dataframe_data_raw

def get_emotibit_preprocessed(emotibit_dataframe_raw):

    preprocessed_emotibit = emotibit_dataframe_raw.dropna(subset=emotibit_final_columns, how='all')

    # Configuracion del StardardScaler
    scaler = StandardScaler()

    # Escalar el tiempo
    emotibit_endtime = preprocessed_emotibit["time"].iloc[-1]
    preprocessed_emotibit["time"] = preprocessed_emotibit["time"] * chunk_time / emotibit_endtime

    time_values = np.arange(0, chunk_time, 0.004)
    time_df = pd.DataFrame({'time': time_values})

    # Añadir las entradas de tiempo regular
    preprocessed_emotibit = preprocessed_emotibit.merge(time_df, on="time", how="outer")

    preprocessed_emotibit = preprocessed_emotibit.sort_values(by="time")

    # Mantener el valor hasta que encontremos uno nuevo
    preprocessed_emotibit.ffill(inplace=True)

    # Eliminar entradas de tiempo irregular
    preprocessed_emotibit = preprocessed_emotibit[preprocessed_emotibit["time"].isin(time_df['time'])]

    columnas_a_eliminar = ['ER']

    preprocessed_emotibit = preprocessed_emotibit.drop(columns = columnas_a_eliminar)

    # Aplicar StandardScaler a las columnas seleccionadas
    preprocessed_emotibit.iloc[:, 1:] = scaler.fit_transform(preprocessed_emotibit.iloc[:, 1:])

    preprocessed_emotibit.bfill(inplace=True)

    return preprocessed_emotibit

def get_eeg_preprocessed(eeg_realtime_dataframe):
    
    n = len(eeg_realtime_dataframe)

    # Crea una lista o rango de tiempo desde 0.000 hasta el final, con un incremento de 0.004
    time_values = [0.004 * i for i in range(n)]

    # Inserta la columna de tiempo en la posición 0
    eeg_realtime_dataframe.insert(0, 'time', time_values)

    eeg_realtime_dataframe = eeg_realtime_dataframe.drop(columns=['Digital', 'Sample'])

    #Canales del casco
    EEG_CHANNELS = 16
    CHUNK_SIZE = 0.5 # En segundos
    SAMPLE_RATE = 250 # En Hz

    EEG_COLUMNS = eeg_realtime_dataframe.columns[1:17]

    preprocessed_eeg = eeg_realtime_dataframe.dropna(subset=EEG_COLUMNS)

    # Definir un filtro, por ejemplo, un filtro paso bajo Butterworth
    ORDER = 6
    FS = 2000  # Frecuencia de muestreo (Hz)
    FC = 300  # Frecuencia de corte (Hz)
    b, a = signal.butter(ORDER, FC / (FS / 2), 'low')

    # Configuracion de ICA
    ica = FastICA(n_components=EEG_CHANNELS, random_state=42, whiten='arbitrary-variance', tol=0.001)

    # Definir la cantidad de últimas medidas para calcular la media
    N = int(CHUNK_SIZE * SAMPLE_RATE)

    # Aplicar el filtro a cada columna por separado
    for column in EEG_COLUMNS:
        preprocessed_eeg[column] = signal.filtfilt(b, a, preprocessed_eeg[column])

    # Aplicar FastICA a las columnas del EEG
    preprocessed_eeg[EEG_COLUMNS] = ica.fit_transform(preprocessed_eeg[EEG_COLUMNS])

    # Crear la nueva columna "media activación" para cada electrodo
    for electrode in range(EEG_CHANNELS):  
        col_name = f'{EEG_COLUMNS[electrode]}_media_activacion'
        preprocessed_eeg[col_name] = preprocessed_eeg.iloc[:, electrode + 1].rolling(window=N).mean()
        
    preprocessed_eeg.bfill(inplace=True)

    return preprocessed_eeg

def emotibit(emotibit_data_list_):

    # dataframe con los datos recibidos
    emotibit_realtime_dataframe = bytes_to_dataframe(emotibit_data_list_)
    # dataframa en formato raw
    emotibit_dataframe_raw = get_emotibit_raw(emotibit_realtime_dataframe)
    # dataframe con los datos preprocesados
    emotibit_dataframe_preprocessed = get_emotibit_preprocessed(emotibit_dataframe_raw)
    
    emotibit_dataframe_preprocessed = emotibit_dataframe_preprocessed.reset_index(drop=True)

    any_nan_rows = emotibit_dataframe_preprocessed.isnull().any(axis=1).sum()
    assert any_nan_rows == 0, "No se han eliminado todos los NaN" + str(any_nan_rows)

    return emotibit_dataframe_preprocessed

def eeg(dataframe_eeg_):

    eeg_realtime_dataframe = dataframe_eeg_
    # preprocesado de los datos
    eeg_dataframe_preprocessed = get_eeg_preprocessed(eeg_realtime_dataframe)

    # Recortar el final
    eeg_dataframe_preprocessed = eeg_dataframe_preprocessed[eeg_dataframe_preprocessed['time'] < chunk_time]
    eeg_dataframe_preprocessed = eeg_dataframe_preprocessed.reset_index(drop=True)

    any_nan_rows = eeg_dataframe_preprocessed.isnull().any(axis=1).sum()
    assert any_nan_rows == 0, "No se han eliminado todos los NaN" + str(any_nan_rows)

    return eeg_dataframe_preprocessed

def concat_and_transpose_dataframes(emotibit_preprocessed_,eeg_preprocessed_):

    preprocessed_data = pd.concat([eeg_preprocessed_, emotibit_preprocessed_.iloc[:,1:]], axis=1)

    any_nan_rows = preprocessed_data.isnull().any(axis=1).sum()
    assert any_nan_rows == 0, "No se han eliminado todos los NaN" + str(any_nan_rows)

    preprocessed_data = preprocessed_data.drop('time', axis=1)

    #Reestructuracion en filas con un chunk de informacion
    expanded_columns = [f'{col}_t{((i - 1) / SAMPLE_RATE):.3f}' for col in preprocessed_data.columns for i in range(1, chunk_size + 1)]

    rows, cols = preprocessed_data.shape

    compressed_data = np.zeros((rows // chunk_size, cols * chunk_size))

    for row in range(0, rows // chunk_size):
        start_idx = row * chunk_size
        end_idx = start_idx + chunk_size
        
        for idx, col in enumerate(preprocessed_data.columns):
            col_data = preprocessed_data[col].values[start_idx:end_idx]
            compressed_data[row, idx * chunk_size:(idx + 1) * chunk_size] = col_data

    preprocessed_data = pd.DataFrame(compressed_data, columns=expanded_columns)

    return preprocessed_data

def preprocess_predict_and_send(emotibit_data_list_, dataframe_eeg_):
    
    emotibit_preprocessed_ = emotibit(emotibit_data_list_)

    eeg_preprocessed_ = eeg(dataframe_eeg_)

    preprocessed_data = concat_and_transpose_dataframes(emotibit_preprocessed_, eeg_preprocessed_)

    predicted_emotions = predict(preprocessed_data.values)

    send_emotions_to_unreal(predicted_emotions)


def send_emotions_to_unreal(predicted_emotions):

    port = 30000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Enviamos el mensaje a localhost en el puerto especificado
        sock.sendto(predicted_emotions.encode('utf-8'), ('localhost', port))
        print(f"Mensaje enviado a localhost en el puerto {port}: {predicted_emotions}")
    except Exception as e:
        print(f"Error al enviar el mensaje: {e}")
    finally:
        # Cerramos el socket
        sock.close()

def predict(preprocessed_data):

    emotions_models = ['RandomForest_Focus_EEG_Emotibit_chunk_10_seg.joblib', 
                       'RandomForest_Amusement_EEG_Emotibit_chunk_10_seg.joblib', 
                       'RandomForest_Surprise_EEG_Emotibit_chunk_10_seg.joblib', 
                       'RandomForest_Fright_EEG_Emotibit_chunk_10_seg.joblib', 
                       'RandomForest_Inmersion_EEG_Emotibit_chunk_10_seg.joblib']
    predicted_emotions = []

    for emotion_model in emotions_models:
        model = joblib.load(emotion_model)
        predicted_emotions.append(model.predict(preprocessed_data)[0])

    predicted_emotions = ','.join(map(str, predicted_emotions))
    
    return predicted_emotions + '\0'
    
    

if __name__ == '__main__':
    client_thread = threading.Thread(target=listen_client)
    client_thread.start()

    eeg_thread = threading.Thread(target=listen_eeg)
    eeg_thread.start()


