import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import ace_tools as tools


# Crear grafo dirigido
G = nx.DiGraph()

# Agregar nodos (routers)
G.add_nodes_from(["A", "B", "C", "D", "E"])

# Agregar enlaces con peso (latencia en ms, ancho de banda en Mbps)
edges = [
    ("A", "B", {"latency": 10, "bandwidth": 50}),
    ("A", "C", {"latency": 20, "bandwidth": 30}),
    ("B", "D", {"latency": 5, "bandwidth": 40}),
    ("C", "D", {"latency": 15, "bandwidth": 20}),
    ("D", "E", {"latency": 10, "bandwidth": 60}),
]

G.add_edges_from(edges)

# Visualizar la red
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
labels = {(u, v): f"{d['latency']}ms" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()

# Crear dataset de tráfico de red
np.random.seed(42)
num_samples = 1000

# Simular ancho de banda usado (10 - 100 Mbps)
bandwidth_used = np.random.uniform(10, 100, num_samples)

# Simular latencia (5 - 50 ms)
latency = np.random.uniform(5, 50, num_samples)

# Simular pérdida de paquetes (0% - 5%)
packet_loss = np.random.uniform(0, 5, num_samples)

# Simular congestión (1 = congestión, 0 = no congestión)
congestion = (bandwidth_used / 100 + latency / 50 + packet_loss / 5) > 1.2
congestion = congestion.astype(int)  # Convertir a 0 y 1

# Crear DataFrame
df = pd.DataFrame({"Bandwidth_Used": bandwidth_used, "Latency": latency, "Packet_Loss": packet_loss, "Congestion": congestion})

# Mostrar datos

tools.display_dataframe_to_user(name="Datos de tráfico de red", dataframe=df)

# Separar features y labels
X = df.drop(columns=["Congestion"])
y = df["Congestion"]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear modelo de red neuronal
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(3,)),  # Capa oculta 1
    keras.layers.Dense(8, activation="relu"),                     # Capa oculta 2
    keras.layers.Dense(1, activation="sigmoid")                   # Capa de salida
])

# Compilar modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar modelo
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Evaluar modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en datos de prueba: {test_acc:.2f}")

# Predicciones
predictions = (model.predict(X_test) > 0.5).astype(int)

# Matriz de confusión


conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Congestión", "Congestión"], yticklabels=["No Congestión", "Congestión"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Reporte de clasificación
print(classification_report(y_test, predictions))

# Datos de prueba
new_data = np.array([[70, 30, 2]])  # Ejemplo: 70 Mbps usado, 30 ms latencia, 2% pérdida
new_data_scaled = scaler.transform(new_data)

# Predicción
prediction = model.predict(new_data_scaled)
congestion_status = "Congestión" if prediction > 0.5 else "No Congestión"
print(f"Predicción de congestión: {congestion_status}")