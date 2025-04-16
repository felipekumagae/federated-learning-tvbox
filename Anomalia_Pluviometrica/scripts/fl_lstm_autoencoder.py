
import multiprocessing
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 🔧 Parâmetros
WINDOW_SIZE = 24
NUM_CLIENTS = 3
EPOCHS = 5
BATCH_SIZE = 32
ROUNDS = 3

# 🔹 Funções utilitárias
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    return df.values

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def create_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation="relu", input_shape=(WINDOW_SIZE, n_features), return_sequences=True),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=False),
        tf.keras.layers.RepeatVector(WINDOW_SIZE),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# 🔹 Classe Cliente FL
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data, self.train_data,
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.train_data, self.train_data, verbose=0)
        return loss, len(self.train_data), {"mse": loss}

# 🔹 Carregar dados e preparar partições
data = load_data("../data/Dataset_Anomalia.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
sequences = create_sequences(scaled_data, WINDOW_SIZE)

# Dividir os dados entre os clientes
clients = []
for i in range(NUM_CLIENTS):
    part = sequences[i::NUM_CLIENTS]
    model = create_model(n_features=sequences.shape[2])
    client = FLClient(model, part)
    clients.append(client)

# 🔹 Funções para servidor e clientes
def start_server():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_eval_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    fl.server.start_server(server_address="127.0.0.1:8080", config={"num_rounds": ROUNDS}, strategy=strategy)

def start_client(client_id):
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=clients[client_id])

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    server = multiprocessing.Process(target=start_server)
    client_procs = [multiprocessing.Process(target=start_client, args=(i,)) for i in range(NUM_CLIENTS)]

    server.start()
    import time; time.sleep(2)

    for p in client_procs:
        p.start()
    for p in client_procs:
        p.join()

    server.join()
