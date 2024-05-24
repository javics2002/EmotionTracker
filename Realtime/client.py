import socket
import time

exit = False
record = False

def client_program():
    # Socket Emotibit
    UDP_IP = "localhost"
    UDP_PORT = 12346 # Port direction found in C:\Program Files\EmotiBit\EmotiBit Oscilloscope\data

    emotibit_socket = socket.socket(socket.AF_INET, 
                        socket.SOCK_DGRAM) 
    emotibit_socket.bind((UDP_IP, UDP_PORT))


    # Socket Client
    client_socket = socket.socket()
    
    # Dirección IP y puerto del servidor
    host = '26.88.39.239'  # Cambia a la IP del servidor si está en otro ordenador
    port = 5000  # Debe coincidir con el puerto que el servidor está escuchando
    
    # Conéctate al servidor
    client_socket.connect((host, port))
    client_socket.setblocking(False)

    start_time = time.time()
    chunk_time = 10
    record = True
    firstTime = True
    emotibit_data_list = []
    data = ""   
    #UDP
    try:
        while not exit:
            while record:
                if firstTime:
                    print("start time emotibit: " + str(time.time()))
                    start_time = time.time()
                    firstTime = False

                if(time.time()-start_time <= chunk_time):
                    data, addr = emotibit_socket.recvfrom(1024)
                
                    # print(data)
                    emotibit_data_list.append(data)

                # Buffer para recibir datos
                buffer = ""
                host_data = any

                try:
                    # Intentar recibir datos de manera no bloqueante
                    while True:
                        # Recibe la respuesta del servidor
                        host_data = client_socket.recv(1024).decode()

                        if not host_data:
                            # No hay más datos disponibles para leer
                            break

                        # Concatenar los datos recibidos al buffer
                        buffer += host_data

                except BlockingIOError:
                    # Esto se produce cuando no hay datos disponibles para leer de inmediato
                    pass

                # print(data)

                if (host_data == 'stop'):
                    client_socket.setblocking(True)

                    print(emotibit_data_list)
                    record = False
                    print(time.time())
                    emotibit_data_list = b';'.join(emotibit_data_list)

                    # Enviar los datos a través del socket
                    client_socket.sendall(emotibit_data_list)

                    host_data = any
                    
                    emotibit_data_list = []
                    emotibit_data_list.clear()
                    buffer = ""
                    data = ""
                    #client_socket.send(emotibit_data_list)
                    break

                # print(f"Respuesta del servidor: {data}")

            host_data = client_socket.recv(1024).decode()
            print(host_data + str(" Waiting for host\n"))
            if (host_data == 'record'):
                record = True
                firstTime = True
                client_socket.setblocking(False)
            print(f"Server response: {data}")

    except Exception as e:
        print("An error occurred:", e)
    finally:
        emotibit_socket.close()
    
    # Cierra la conexión
    client_socket.close()

if __name__ == '__main__':
    client_program()