# ===============================================
# Paso 1: Exploración de datos
# ===============================================

import os, re
from collections import Counter
import pandas as pd
import numpy as np

def extract_year(title: str):
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"\((\d{4})\)$", title.strip())
    return int(m.group(1)) if m else np.nan

def split_genres(s: str):
    if pd.isna(s) or s == "":
        return ["(no genres listed)"]
    return s.split("|")

def describe_df(name, df: pd.DataFrame, head_n=5):
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    print("columnas:", list(df.columns))
    print(f"\n{head_n} primeras filas:")
    print(df.head(head_n))
    print("\nNulos por columna:")
    print(df.isna().sum())

# ---------- Cargar ----------
TRAIN = "movies_train.csv"
TEST  = "movies_test.csv"
ML_MOVIES = os.path.join("ml-25m", "movies.csv")

if not os.path.exists(TRAIN): raise FileNotFoundError(TRAIN)
if not os.path.exists(TEST):  raise FileNotFoundError(TEST)
if not os.path.exists(ML_MOVIES): raise FileNotFoundError(ML_MOVIES)

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
ml_movies = pd.read_csv(ML_MOVIES)

# ---------- Chequeos ----------
describe_df("TRAIN", train)
describe_df("TEST", test)
describe_df("MOVIELENS movies.csv", ml_movies.sample(min(5, len(ml_movies)), random_state=42))

# ---------- Cruzar IDs ----------
needed_ids = set(train["movieId"].tolist() + test["movieId"].tolist())
movies_sub = ml_movies[ml_movies["movieId"].isin(needed_ids)].copy()
print(f"\nTotal movieId requeridos: {len(needed_ids)}")
print("Encontrados en movies.csv:", movies_sub.shape[0])

# ---------- Explorar géneros ----------
movies_sub["genres"] = movies_sub["genres"].fillna("(no genres listed)")
all_genres = []
for s in movies_sub["genres"]:
    all_genres.extend(split_genres(s))

genre_counts = Counter(all_genres)
print("\nTop 15 géneros (conteo en train+test):")
for g, c in genre_counts.most_common(15):
    print(f"  {g:15s} -> {c}")

no_genres = movies_sub[movies_sub["genres"].eq("(no genres listed)")].shape[0]
print("\nPelículas sin género:", no_genres)

# ---------- Explorar años ----------
movies_sub["year"] = movies_sub["title"].apply(extract_year)
print("\nAño (resumen):")
print(movies_sub["year"].describe())

missing_year = movies_sub["year"].isna().sum()
print("Años faltantes:", missing_year)

# ---------- Duplicados ----------
dups = movies_sub[movies_sub.duplicated("movieId", keep=False)]
print("\nDuplicados por movieId:", dups.shape[0])

# ===============================================
# Paso 2: Limpieza + Construcción de Features
# ===============================================

from sklearn.preprocessing import StandardScaler

def build_features_from_movies(movies_df: pd.DataFrame):
    df = movies_df.copy()
    df["genres"] = df["genres"].fillna("(no genres listed)")
    genre_set = set()
    for s in df["genres"]:
        genre_set.update(s.split("|"))
    genre_list = sorted(list(genre_set))

    for g in genre_list:
        df[g] = df["genres"].apply(lambda s: 1 if g in s.split("|") else 0).astype(int)

    df["year"] = df["title"].apply(extract_year)
    df["year"] = df["year"].fillna(df["year"].median())
    feat_cols = genre_list + ["year"]
    return df, feat_cols

def main():
    needed_ids = set(train["movieId"].tolist() + test["movieId"].tolist())
    movies_sub = ml_movies[ml_movies["movieId"].isin(needed_ids)].copy()
    movies_feats, feat_cols = build_features_from_movies(movies_sub)

    # IDs disponibles
    id_to_row = movies_feats.set_index("movieId")
    available_ids = set(id_to_row.index)

    def get_safe_features(df_ids):
        feats = []
        for mid in df_ids:
            if mid in available_ids:
                feats.append(id_to_row.loc[mid, feat_cols].values)
            else:
                empty = [0] * (len(feat_cols) - 1) + [movies_feats["year"].median()]
                feats.append(empty)
        return pd.DataFrame(feats, columns=feat_cols)

    X_train_raw = get_safe_features(train["movieId"].values)
    X_test_raw  = get_safe_features(test["movieId"].values)

    # Escalado
    scaler = StandardScaler()
    X_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
    X_all_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=feat_cols)

    X_train_scaled = X_all_scaled.iloc[:len(X_train_raw)].reset_index(drop=True)
    X_test_scaled  = X_all_scaled.iloc[len(X_train_raw):].reset_index(drop=True)

    os.makedirs("outputs", exist_ok=True)
    X_train_scaled.to_csv("outputs/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("outputs/X_test_scaled.csv", index=False)

    print("\nFeatures generados correctamente.")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    print(f"Columnas: {feat_cols}")

if __name__ == "__main__":
    main()
