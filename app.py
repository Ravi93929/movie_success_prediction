
import streamlit as st
import torch, joblib, numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model_pipeline():
    pipeline = joblib.load("movie_success_pipeline_final.pkl")
    input_dim = len(pipeline["feature_names"])

    class FinalMovieNN(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.45),
                torch.nn.Linear(512, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.35),
                torch.nn.Linear(256, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.25),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)

    model = FinalMovieNN(input_dim)
    model.load_state_dict(torch.load("movie_success_nn_model_final.pt", map_location="cpu"))
    model.eval()
    sentence_model = SentenceTransformer(pipeline["plot_model_name"])
    return model, pipeline, sentence_model

model, pipeline, sentence_model = load_model_pipeline()
scaler = pipeline["scaler"]; pca = pipeline["pca"]
mlb = pipeline["mlb"]; top_directors = pipeline["top_directors"]
BEST_THRESHOLD = pipeline["best_threshold"]

def predict_movie(movie):
    num = np.array([[movie["budget"], movie["opening_weekend"],
                     movie["imdb_rating"], movie["metascore"], movie["year"]]])
    num_scaled = scaler.transform(num)
    genre_vec = mlb.transform([movie["genres"]])[0]
    director_vec = [1 if movie["director"] == d else 0 for d in top_directors]
    plot_emb = sentence_model.encode([movie["plot"]])
    plot_pca = pca.transform(plot_emb)
    sentiment_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    sentiment_score = np.array([sentiment_map.get(movie["sentiment"], 0.5)])
    features = np.concatenate([num_scaled[0], genre_vec, director_vec, plot_pca[0], sentiment_score])
    X = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad(): prob = model(X).item()
    label = "Success" if prob >= BEST_THRESHOLD else "Failure"
    return label, prob

st.set_page_config(page_title="🎬 Movie Success Predictor", layout="centered")
st.title("🎬 Movie Success Prediction (Colab Demo)")

col1, col2 = st.columns(2)
with col1:
    budget = st.number_input("💰 Budget", min_value=0)
    imdb = st.number_input("⭐ IMDb Rating", 0.0, 10.0, step=0.1)
    meta = st.number_input("📰 Metascore", 0, 100)
with col2:
    opening = st.number_input("🎟️ Opening Weekend", 0)
    year = st.number_input("📅 Release Year", 2000, 2030)

genres = st.text_input("🎭 Genres (comma separated)", "Action, Adventure")
director = st.text_input("🎬 Director Name", "Christopher Nolan")
plot = st.text_area("🧠 Plot Summary", "A scientist builds a device that can reverse time itself...")
sentiment = st.selectbox("💡 Sentiment", ["positive", "neutral", "negative"])

if st.button("Predict 🎯"):
    movie = {
        "budget": budget,
        "opening_weekend": opening,
        "imdb_rating": imdb,
        "metascore": meta,
        "year": year,
        "genres": [g.strip().capitalize() for g in genres.split(",") if g.strip()],
        "director": director,
        "plot": plot,
        "sentiment": sentiment
    }
    label, prob = predict_movie(movie)
    st.success(f"✅ Prediction: {label}")
    st.info(f"🎯 Probability: {prob*100:.2f}% (Threshold: {BEST_THRESHOLD:.2f})")
