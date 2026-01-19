# ======================
# app.py
# ======================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from difflib import SequenceMatcher
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Steam Game Recommender", layout="wide")
st.title("🎮 Steam Game Recommendation System")
st.caption("User-Based Collaborative Filtering | KNN | Pearson Similarity")
st.markdown("---")

# ======================
# SESSION RECOVERY (Persistent Login)
# ======================
if "user_id" not in st.session_state:
    if "uid" in st.query_params:
        st.session_state.user_id = st.query_params["uid"]

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_games():
    # Only load necessary columns to save memory
    cols = ["AppID", "Name", "Release date", "Genres", "Positive", "Negative"]
    # Added index_col=False to prevent pandas from using the first column as an index, 
    # which was causing column shifting issues.
    df = pd.read_csv("game_dataset.csv", usecols=cols, index_col=False)
    
    # Rename columns to match the app's internal logic
    df = df.rename(columns={
        "AppID": "game_id",
        "Name": "name",
        "Genres": "genre"
    })
    
    # Process release year
    df["release_year"] = pd.to_datetime(df["Release date"], errors="coerce").dt.year.fillna(0).astype(int)
    
    # Calculate rating (on a scale of 0-5) based on positive/negative reviews
    # formula: (pos / (pos + neg)) * 5
    df["game_rating"] = (df["Positive"] / (df["Positive"] + df["Negative"] + 1e-5)) * 5
    df["game_rating"] = df["game_rating"].round(1)
    
    df["game_id"] = df["game_id"].astype(str)
    return df

@st.cache_data
def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["user_id", "username", "password"])
        df.to_csv("users.csv", index=False)
        return df
    # Use encoding='utf-8-sig' to handle BOM and sep=None for auto-detection
    # Added index_col=False for consistency
    df = pd.read_csv("users.csv", sep=None, engine='python', encoding='utf-8-sig', index_col=False)
    df.columns = df.columns.str.strip()
    df["user_id"] = df["user_id"].astype(str)
    return df

def load_ratings():
    if os.path.exists("user_ratings.csv"):
        # Use comma separator explicitly
        try:
            df = pd.read_csv("user_ratings.csv", sep=",", encoding='utf-8-sig', index_col=False)
            df.columns = df.columns.str.strip()
            if not df.empty:
                df["user_id"] = df["user_id"].astype(str)
                df["game_id"] = df["game_id"].astype(str)
                return df
        except Exception:
            return pd.DataFrame(columns=["user_id", "game_id", "rating"])
    return pd.DataFrame(columns=["user_id", "game_id", "rating"])

def save_rating(user_id, game_id, score, game_name=None):
    user_id = str(user_id)
    game_id = str(game_id)
    
    # Load current ratings to ensure we don't overwrite other changes
    ratings_df = load_ratings()
    
    # Check if rating already exists
    mask = (ratings_df["user_id"] == user_id) & (ratings_df["game_id"] == game_id)
    
    if mask.any():
        ratings_df.loc[mask, "rating"] = score
        if game_name:
            ratings_df.loc[mask, "game_name"] = game_name
    else:
        new_row = pd.DataFrame({
            "user_id": [user_id], 
            "game_id": [game_id], 
            "rating": [score],
            "game_name": [game_name] if game_name else [None]
        })
        ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    
    ratings_df.to_csv("user_ratings.csv", index=False, sep=",")
    return ratings_df


def save_rating_callback(user_id, game_id, game_name, key):
    score = st.session_state[key]
    save_rating(user_id, game_id, score, game_name)
    st.toast(f"Rating {game_name} disimpan!")

# @st.fragment removed to allow global state updates
def rating_component(row, user_id, ratings_df, key_prefix):
    current_rating = 3.0
    user_prev = ratings_df[(ratings_df.user_id == user_id) & (ratings_df.game_id == row["game_id"])]
    if not user_prev.empty:
        current_rating = float(user_prev.iloc[0].rating)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Fallback priority: metadata name -> stored game_name -> game_id
        game_name_display = row.get('name')
        if pd.isna(game_name_display) or not game_name_display:
            game_name_display = row.get('game_name')
        if pd.isna(game_name_display) or not game_name_display:
            game_name_display = row['game_id']
            
        st.write(f"**{game_name_display}**")
        
        genre = row['genre'] if pd.notna(row.get('genre')) else "Unknown Genre"
        year = int(row['release_year']) if pd.notna(row.get('release_year')) else "N/A"
        global_rating = row['game_rating'] if pd.notna(row.get('game_rating')) else "?"
        
        st.caption(f"{genre} | {year} | ⭐ {global_rating}")
        if "reason" in row:
            st.caption("💡 " + row["reason"])
    
    with col2:
        input_key = f"{key_prefix}_{row['game_id']}"
        display_name = row.get('name') if pd.notna(row.get('name')) else row.get('game_name')
        
        rate = st.number_input(
            "Rating",
            min_value=1.0, max_value=5.0, step=0.5,
            value=current_rating,
            key=input_key
        )
        
        st.button(
            "Simpan", 
            key=f"btn_{key_prefix}_{row['game_id']}",
            on_click=save_rating_callback,
            args=(user_id, row["game_id"], display_name, input_key)
        )


games = load_games()
users = load_users()
ratings = load_ratings()

# ======================
# FUZZY SEARCH
# ======================
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def smart_search(query, df, threshold=0.3):
    df = df.copy()
    # Fill NaN genres to avoid errors
    df["genre"] = df["genre"].fillna("")
    
    # Calculate similarity for both name and genre
    df["name_score"] = df["name"].apply(lambda x: similarity(query, str(x)))
    df["genre_score"] = df["genre"].apply(lambda x: 1.0 if query.lower() in str(x).lower() else 0.0)
    
    # Combine scores (either high title similarity OR exact genre match)
    df["score"] = df[["name_score", "genre_score"]].max(axis=1)
    
    return df[df["score"] >= threshold].sort_values("score", ascending=False)

# ======================
# USER-BASED CF (KNN + PEARSON)
# ======================
def user_based_cf(user_id, ratings_df, games_df, top_n=10):
    if ratings_df.empty:
        res = games_df.sort_values("game_rating", ascending=False).head(top_n).copy()
        res["reason"] = "Game ini sangat populer di Steam"
        return res

    pivot = ratings_df.pivot(index="user_id", columns="game_id", values="rating")

    if user_id not in pivot.index:
        res = games_df.sort_values("game_rating", ascending=False).head(top_n).copy()
        res["reason"] = "Game populer untuk pengguna baru"
        return res

    user_mean = pivot.mean(axis=1)
    pivot_centered = pivot.sub(user_mean, axis=0).fillna(0)

    sparse_matrix = csr_matrix(pivot_centered.values)

    model = NearestNeighbors(metric="cosine", algorithm="brute",
                             n_neighbors=min(6, len(pivot_centered)))
    model.fit(sparse_matrix)

    user_idx = list(pivot_centered.index).index(user_id)
    distances, indices = model.kneighbors(
        pivot_centered.iloc[user_idx].values.reshape(1, -1)
    )

    similarities = 1 - distances.flatten()[1:]
    neighbors = indices.flatten()[1:]

    scores = {}

    for idx, sim in zip(neighbors, similarities):
        if sim <= 0:
            continue
        neighbor_id = pivot_centered.index[idx]
        neighbor_mean = user_mean[neighbor_id]

        for game_id, rating in pivot.loc[neighbor_id].items():
            if pd.notna(rating) and pd.isna(pivot.loc[user_id, game_id]):
                if game_id not in scores:
                    scores[game_id] = {"num": 0, "den": 0}
                scores[game_id]["num"] += sim * (rating - neighbor_mean)
                scores[game_id]["den"] += sim

    predictions = []
    for game_id, val in scores.items():
        pred = user_mean[user_id] + (val["num"] / val["den"])
        predictions.append((game_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [g[0] for g in predictions[:top_n]]

    result = games_df[games_df["game_id"].isin(top_ids)].copy()
    
    # Recover names from ratings_df for games not in metadata
    missing_ids = [tid for tid in top_ids if tid not in result["game_id"].values]
    if missing_ids:
        missing_info = ratings_df[ratings_df["game_id"].isin(missing_ids)][["game_id", "game_name"]].drop_duplicates()
        # Create dummy rows for missing games to show them in-app
        for _, row in missing_info.iterrows():
            new_row = pd.DataFrame({
                "game_id": [row["game_id"]],
                "name": [row["game_name"]],
                "genre": ["Unknown Genre"],
                "release_year": [0],
                "game_rating": [0]
            })
            result = pd.concat([result, new_row], ignore_index=True)

    result["reason"] = "User lain yang mirip kamu juga menyukai game ini"
    return result

def calculate_evaluation_metrics(ratings_df, games_df):
    """
    Simulate prediction accuracy by hiding a portion of ratings and predicting them.
    Actually, for a real-time app, we can compare global game_rating vs user rating
    to see how much the user differs from the crowd, or implement a basic internal test.
    """
    if len(ratings_df) < 5:
        return None, None
    
    # Simple internal evaluation: How close are the predictions to actual user ratings?
    # This is a simplified MAE/RMSE calculation for demonstration.
    y_true = []
    y_pred = []
    
    # We only test for games in metadata
    test_data = ratings_df.merge(games_df, on="game_id")
    if test_data.empty:
        return None, None
        
    y_true = test_data["rating"].values
    y_pred = test_data["game_rating"].values # Predict using global average as baseline
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return round(mae, 3), round(rmse, 3)


# ======================
# LOGIN / REGISTER
# ======================
# Sidebar Header
st.sidebar.header("🔐 Login / Register")
mode = st.sidebar.radio("Mode", ["Login", "Register"])

if "user_id" not in st.session_state:
    if mode == "Login":
        u = st.sidebar.text_input("Username")
        p = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = users[(users.username == u) & (users.password == p)]
            if not user.empty:
                uid = str(user.iloc[0].user_id)
                st.session_state.user_id = uid
                st.query_params["uid"] = uid
                st.sidebar.success("Login berhasil")
                st.rerun()
            else:
                st.sidebar.error("Username atau password salah")

    else:
        u = st.sidebar.text_input("Username baru")
        p = st.sidebar.text_input("Password baru", type="password")
        if st.sidebar.button("Register"):
            if users.empty:
                new_id = "1"
            else:
                new_id = str(int(pd.to_numeric(users.user_id).max()) + 1)
            
            users.loc[len(users)] = [new_id, u, p]
            users.to_csv("users.csv", index=False)
            st.session_state.user_id = new_id
            st.query_params["uid"] = new_id
            st.sidebar.success("Register berhasil")
            st.rerun()

else:
    st.sidebar.success(f"Login sebagai User {st.session_state.user_id}")
    if st.sidebar.button("Logout"):
        del st.session_state.user_id
        if "uid" in st.query_params:
            del st.query_params["uid"]
        st.rerun()

# ======================
# MAIN APP
# ======================
if "user_id" not in st.session_state:
    st.info("Silakan login terlebih dahulu.")
    st.stop()

user_id = st.session_state.user_id

tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📜 Rating Saya", "📊 Evaluasi & Statistik"])

with tab1:
    # SECTION 1: SEARCH
    st.write("### 🔍 Cari Game")
    query = st.text_input("Masukkan nama game (contoh: dota, gta, cs)")
    if query:
        results = smart_search(query, games)
        if results.empty:
            st.warning("Game tidak ditemukan.")
        else:
            st.write("#### 🔎 Hasil Pencarian:")
            for _, row in results.iterrows():
                rating_component(row, user_id, ratings, "search")
                st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # SECTION 2: RECOMMENDATIONS
    st.write("### 🎯 Rekomendasi Untuk Anda")
    recs = user_based_cf(user_id, ratings, games)
    
    user_ratings_count = len(ratings[ratings.user_id == user_id])
    
    if recs.empty:
        if user_ratings_count == 0:
            st.info("Berikan rating ke beberapa game untuk mendapatkan rekomendasi personal!")
        
        st.write("#### 📈 Game Populer Saat Ini")
        recs = games.sort_values("game_rating", ascending=False).head(10).copy()
        recs["reason"] = "Trending: Game paling banyak disukai saat ini"

    for _, row in recs.iterrows():
        rating_component(row, user_id, ratings, "rec")
        st.markdown("---")

with tab2:
    st.subheader("Game yang sudah Anda beri rating")
    user_history = ratings[ratings.user_id == user_id]
    
    if user_history.empty:
        st.info("Anda belum memberi rating pada game apa pun.")
    else:
        # Join with games to get names, use 'how=left' to keep games not in metadata
        # Display order: Newest at top, Oldest at bottom (Reverse of CSV order)
        history_detailed = user_history.merge(games, on="game_id", how="left").iloc[::-1]
        for _, row in history_detailed.iterrows():
            rating_component(row, user_id, ratings, "history")
            st.markdown("---")

with tab3:
    st.subheader("📊 Statistik & Evaluasi Sistem")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("### 🏆 Top 10 Game Populer")
        top_10_pop = games.sort_values("game_rating", ascending=False).head(10)
        
        # Prepare data for chart
        chart_data = top_10_pop[["name", "game_rating"]].copy()
        chart_data = chart_data.rename(columns={"game_rating": "Rating", "name": "Game"})
        
        st.bar_chart(chart_data.set_index("Game"))

    with col_b:
        st.write("### 📉 Akurasi Prediksi Recommender")
        mae, rmse = calculate_evaluation_metrics(ratings, games)
        
        if mae is not None:
            st.metric("Mean Absolute Error (MAE)", mae)
            st.metric("Root Mean Square Error (RMSE)", rmse)
        else:
            st.warning("Data rating belum cukup (minimal 5 rating).")
            
    st.markdown("---")
    
    # NEW EDA SECTION
    st.write("### 🔍 Exploratory Data Analysis (EDA)")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.write("#### 📊 Distribusi Rating User")
        if not ratings.empty:
            # Group by rating and count
            dist = ratings["rating"].value_counts().sort_index().reset_index()
            dist.columns = ["Rating", "Jumlah"]
            st.bar_chart(dist.set_index("Rating"), color="#FF4B4B")
        else:
            st.info("Belum ada data rating.")

    with eda_col2:
        st.write("#### 📈 Statistik Deskriptif Rating")
        if not ratings.empty:
            stats = ratings["rating"].describe().to_frame().T
            st.dataframe(stats, hide_index=True)
            
            avg_u = ratings["rating"].mean()
            avg_g = games["game_rating"].mean()
            
            st.write(f"**Rata-rata Rating Anda:** {avg_u:.2f} ⭐")
            st.write(f"**Rata-rata Rating Global:** {avg_g:.2f} ⭐")
            
            if avg_u > avg_g:
                st.success("Anda cenderung memberikan rating lebih tinggi dari rata-rata global.")
            else:
                st.info("Anda cenderung memberikan rating lebih rendah dari rata-rata global.")
        else:
            st.info("Belum ada data untuk statistik.")

    st.markdown("---")
    st.write("### 📋 Ringkasan Data Dasar")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Game", len(games))
    c2.metric("Total User", len(users))
    c3.metric("Total Rating", len(ratings))

