from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from flask import Flask, request, jsonify

# Wczytanie i przygotowanie danych
iris_data = load_iris()
features, labels = iris_data.data, iris_data.target

# Podział na zbiory treningowe i testowe
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.4, random_state=42
)

# Inicjalizacja i trenowanie modelu perceptronu
perceptron_model = Perceptron()
perceptron_model.fit(train_features, train_labels)

# Ocena modelu na zbiorze testowym
test_predictions = perceptron_model.predict(test_features)
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Dokładność perceptronu: {accuracy}")

# Zapis
with open('model.pkl', 'wb') as model_file:
    pickle.dump(perceptron_model, model_file)

# Konfiguracja Flask
app = Flask(__name__)

# Ładowanie modelu 
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    received_data = request.get_json()
    input_features = np.array(received_data['features']).reshape(1, -1)
    result = loaded_model.predict(input_features)
    return jsonify({'prediction': int(result[0])})

# Uruchomienie
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
