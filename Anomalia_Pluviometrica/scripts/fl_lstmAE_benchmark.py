
import multiprocessing
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
import time
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WINDOW_SIZE = 24
BATCH_SIZE = 32
DATA_PATH = "../data/Dataset_Anomalia.csv"

# Combinações a testar: (EPOCHS, ROUNDS, NUM_CLIENTS)
CONFIGURATIONS = [
    (3, 3, 2),
    (5, 3, 3),
    (5, 5, 4),
    (10, 3, 3),
    (10, 10, 5)
]

results = []

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
        tf.keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=False),
        tf.keras.layers.RepeatVector(WINDOW_SIZE),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, epochs):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs
        self.latest_loss = None

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data, self.train_data,
                       epochs=self.epochs, batch_size=BATCH_SIZE, verbose=0)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.train_data, self.train_data, verbose=0)
        self.latest_loss = loss
        return loss, len(self.train_data), {"mse": loss}

def run_federated_simulation(EPOCHS, ROUNDS, NUM_CLIENTS):
    logging.info(f"🚀 Rodando FL com {NUM_CLIENTS} clientes, {EPOCHS} épocas, {ROUNDS} rounds")
    data = load_data(DATA_PATH)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    sequences = create_sequences(scaled_data, WINDOW_SIZE)

    clients = []
    for i in range(NUM_CLIENTS):
        part = sequences[i::NUM_CLIENTS]
        model = create_model(n_features=sequences.shape[2])
        client = FLClient(model, part, epochs=EPOCHS)
        clients.append(client)

    def start_server():
        strategy = fl.server.strategy.FedAvg()
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=strategy
        )

    def start_client(client_id):
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=clients[client_id].to_client()
        )

    multiprocessing.set_start_method("spawn", force=True)
    server = multiprocessing.Process(target=start_server)
    client_procs = [
        multiprocessing.Process(target=start_client, args=(i,))
        for i in range(NUM_CLIENTS)
    ]

    server.start()
    time.sleep(2)
    for p in client_procs:
        p.start()
    for p in client_procs:
        p.join()
    server.join()

    # Coletar e salvar resultados
    losses = [c.latest_loss for c in clients if c.latest_loss is not None]
    mean_loss = np.mean(losses) if losses else None
    results.append({
        "epochs": EPOCHS,
        "rounds": ROUNDS,
        "num_clients": NUM_CLIENTS,
        "mean_mse": mean_loss
    })
    logging.info(f"✅ Resultado médio MSE: {mean_loss:.6f}")


if __name__ == "__main__":
    for EPOCHS, ROUNDS, NUM_CLIENTS in CONFIGURATIONS:
        run_federated_simulation(EPOCHS, ROUNDS, NUM_CLIENTS)

    # Salvar resultados
    os.makedirs("benchmark_results", exist_ok=True)
    with open("benchmark_results/federated_lstm_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Mostrar melhor configuração
    best = min(results, key=lambda x: x["mean_mse"] if x["mean_mse"] is not None else float("inf"))
    logging.info(f"🏆 Melhor desempenho -> Epochs: {best['epochs']}, Rounds: {best['rounds']}, Clients: {best['num_clients']}, MSE: {best['mean_mse']:.6f}")
