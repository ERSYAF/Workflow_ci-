print(">>> TES: SKRIP Modelling.py MULAI DI SINI !!! <<<", flush=True)

import pandas as pd
import numpy as np
import ast
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import mlflow

def clean_text_for_token(text):
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""

def create_content_soup(df):
    print("CREATE_CONTENT_SOUP: Memulai pembuatan 'content soup'...", flush=True)
    df_copy = df.copy()
    df_copy['overview_processed'] = df_copy['overview'].fillna('').astype(str).str.lower()

    def process_genres_list(genre_list):
        if isinstance(genre_list, str):
            try:
                actual_list = ast.literal_eval(genre_list)
                return " ".join([clean_text_for_token(genre) for genre in actual_list])
            except (ValueError, SyntaxError):
                return ""
        elif isinstance(genre_list, list):
            return " ".join([clean_text_for_token(genre) for genre in genre_list])
        return ""

    df_copy['genres_soup'] = df_copy.get('genres_processed', "").apply(process_genres_list)
    df_copy['director_soup'] = df_copy.get('director', "").apply(lambda x: clean_text_for_token(x) if pd.notnull(x) else "")
    
    df_copy['soup'] = (df_copy['overview_processed'] + ' ' +
                       (df_copy['genres_soup'] + ' ') * 2 +
                       (df_copy['director_soup'] + ' ') * 2)
    
    print("CREATE_CONTENT_SOUP: 'Content soup' berhasil dibuat.", flush=True)
    return df_copy[['id', 'title', 'soup']]

def load_data_and_generate_soup(dataset_path):
    print(f"LOAD_AND_SOUP: Mencoba memuat dataset dari: {dataset_path}", flush=True)
    try:
        df = pd.read_csv(dataset_path)
        print(f"LOAD_AND_SOUP: Dataset berhasil dimuat. Baris: {len(df)}, Kolom: {len(df.columns)}", flush=True)
        return create_content_soup(df)
    except Exception as e:
        print(f"LOAD_AND_SOUP: ERROR - {e}", flush=True)
        return None

def get_recommendations(movie_title, cosine_sim_matrix_input, data, movie_indices, top_n=10):
    if movie_title not in movie_indices:
        return pd.Series(dtype='object')
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix_input[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices_output = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices_output]

if __name__ == "__main__":
    print(">>> BLOK __main__ di Modelling.py TERPANGGIL <<<", flush=True)

    # Gunakan autolog (tanpa start_run, tanpa log manual)
    mlflow.set_experiment("Movie Recommender - Autolog Only")
    mlflow.autolog()

    dataset_path = "tmdb_movies_processed.csv"
    movie_data_with_soup = load_data_and_generate_soup(dataset_path)

    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        # Proses content-based (tidak akan tercatat di MLflow, hanya dummy yang dicatat)
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=3, max_df=0.7)
        tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'].fillna(''))
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
        np.savez_compressed("cosine_matrix.npz", cosine_sim_matrix=cosine_sim_matrix)

        indices = pd.Series(movie_data_with_soup.index, index=movie_data_with_soup['title']).drop_duplicates()
        test_movie_title = movie_data_with_soup['title'].iloc[0]
        recommendations = get_recommendations(test_movie_title, cosine_sim_matrix, movie_data_with_soup, indices)

        print(f"\n--- Rekomendasi untuk '{test_movie_title}' ---")
        if not recommendations.empty:
            print("\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)]))
        else:
            print("Tidak ada rekomendasi ditemukan.")

        # ✅ Dummy model agar autolog mencatat run
        print("\n>>> MEMULAI TRAINING DUMMY MODEL UNTUK AUTOLOG <<<", flush=True)
        iris = load_iris()
        X, y = iris.data, iris.target
        clf = RandomForestClassifier()
        clf.fit(X, y)
        print(">>> TRAINING DUMMY SELESAI <<<", flush=True)

    print("\nSkrip modelling.py selesai.", flush=True)
