
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

class MLPGenreClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.4):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer('all-mpnet-base-v2', device=device)
mlb = joblib.load("mlb.pkl")
year_scaler = joblib.load("year_scaler.pkl")
model = MLPGenreClassifier(769, [1280, 768, 384], len(mlb.classes_))
model.load_state_dict(torch.load("best_mlp_sbert_fold_1.pth", map_location=device))
model.eval()

@app.route("/")
def home():
    return "🎬 API de Clasificación de Géneros de Películas Activa"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    plot = data.get("plot")
    year = data.get("year")

    if not plot or year is None:
        return jsonify({"error": "Faltan campos 'plot' y/o 'year'"}), 400

    embedding = embedder.encode([plot])[0]
    year_scaled = year_scaler.transform([[year]])[0]
    x_input = np.hstack([embedding, year_scaled])
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(x_tensor)).cpu().numpy()[0]
        predicted_labels = mlb.classes_[(output > 0.5)]

    return jsonify({
        "genres": predicted_labels.tolist(),
        "probabilidades": {mlb.classes_[i]: float(output[i]) for i in range(len(output))}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
