print(">>> TES: SKRIP Modelling.py MULAI DI SINI !!! <<<", flush=True)
import pandas as pd
import numpy as np
import ast
import os
import time
import traceback
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn

def clean_text_for_token(text):
    """Menghilangkan spasi dan mengubah ke huruf kecil untuk nama/genre."""
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""

def create_content_soup(df):
    """Mempersiapkan dan menggabungkan fitur konten menjadi satu 'soup' string per film."""
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

    if 'genres_processed' in df_copy.columns:
        df_copy['genres_soup'] = df_copy['genres_processed'].apply(process_genres_list)
    else:
        df_copy['genres_soup'] = ""
        
    if 'director' in df_copy.columns:
        df_copy['director_soup'] = df_copy['director'].apply(lambda x: clean_text_for_token(x) if pd.notnull(x) else "")
    else:
        df_copy['director_soup'] = ""

    df_copy['soup'] = (df_copy['overview_processed'] + ' ' +
                       (df_copy['genres_soup'] + ' ') * 2 +
                       (df_copy['director_soup'] + ' ') * 2)
                      
    print("CREATE_CONTENT_SOUP: 'Content soup' berhasil dibuat.", flush=True)
    return df_copy[['id', 'title', 'soup']]

def load_data_and_generate_soup(dataset_path):
    """Memuat dataset dari path CSV dan memanggil create_content_soup."""
    print(f"LOAD_AND_SOUP: Mencoba memuat dataset dari: {dataset_path}", flush=True)
    try:
        df = pd.read_csv(dataset_path)
        print(f"LOAD_AND_SOUP: Dataset berhasil dimuat. Baris: {len(df)}, Kolom: {len(df.columns)}", flush=True)
        
        required_cols = ['id', 'title', 'overview', 'genres_processed', 'director']
        if not all(col in df.columns for col in required_cols):
            print(f"LOAD_AND_SOUP: ERROR - Kolom penting tidak ada. Dibutuhkan: {required_cols}", flush=True)
            return None
            
        return create_content_soup(df)
    except FileNotFoundError:
        print(f"LOAD_AND_SOUP: ERROR - File dataset tidak ditemukan di: {dataset_path}", flush=True)
        return None
    except Exception as e:
        print(f"LOAD_AND_SOUP: ERROR - Terjadi kesalahan saat memuat atau memproses data: {e}", flush=True)
        traceback.print_exc()
        return None

def get_recommendations(movie_title, cosine_sim_matrix_input, data, movie_indices, top_n=10):
    """Fungsi untuk mendapatkan rekomendasi film berdasarkan judul."""
    print(f"\nRECOMMENDER: Mencari rekomendasi untuk film: '{movie_title}'")
    if movie_title not in movie_indices:
        print(f"RECOMMENDER: Judul film '{movie_title}' tidak ditemukan.", flush=True)
        return pd.Series(dtype='object') 
    
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix_input[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices_output = [i[0] for i in sim_scores]
    recommended_movies = data['title'].iloc[movie_indices_output]
    print("RECOMMENDER: Rekomendasi ditemukan.", flush=True)
    return recommended_movies


if __name__ == "__main__":
    print(">>> BLOK __main__ di Modelling.py TERPANGGIL <<<", flush=True)
    

    experiment_name = "Movie Recommender - Final" 
    mlflow.set_experiment(experiment_name)
    

    path_ke_dataset = "tmdb_movies_processed.csv"
    

    movie_data_with_soup = load_data_and_generate_soup(path_ke_dataset)
    

    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        with mlflow.start_run(run_name="ContentBased_Recommender_Run") as run:
            run_id = run.info.run_id
            print(f"\nMAIN: Memulai MLflow Run ID: {run_id}", flush=True)
            mlflow.set_tag("model_type", "Content-Based Recommender")


            print("MAIN: Memulai TF-IDF Vectorization...", flush=True)
            tfidf_params = {
                "stop_words": 'english',
                "ngram_range": (1,2),
                "min_df": 3,
                "max_df": 0.7
            }
            mlflow.log_params(tfidf_params)
            
            tfidf = TfidfVectorizer(**tfidf_params)
            tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'].fillna(''))
            
            mlflow.log_metric("tfidf_num_features", tfidf_matrix.shape[1])
            

            tfidf_path = "tfidf_vectorizer.pkl"
            joblib.dump(tfidf, tfidf_path)
            mlflow.log_artifact(tfidf_path, artifact_path="model")
            os.remove(tfidf_path)
            print("MAIN: TF-IDF Vectorizer disimpan dan dilog ke MLflow.", flush=True)

            print("MAIN: Menghitung Cosine Similarity...", flush=True)
            cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            

            cosine_sim_path = "cosine_similarity_matrix.npz"
            np.savez_compressed(cosine_sim_path, cosine_sim_matrix=cosine_sim_matrix)
            mlflow.log_artifact(cosine_sim_path, artifact_path="model")
            os.remove(cosine_sim_path)
            print("MAIN: Cosine Similarity Matrix disimpan dan dilog ke MLflow.", flush=True)


            movie_titles_path = "movie_titles.csv"
            movie_data_with_soup[['id', 'title']].to_csv(movie_titles_path, index=False)
            mlflow.log_artifact(movie_titles_path, artifact_path="data")
            os.remove(movie_titles_path)

            indices = pd.Series(movie_data_with_soup.index, index=movie_data_with_soup['title']).drop_duplicates()
            test_movie_title = movie_data_with_soup['title'].iloc[0]
            recommendations = get_recommendations(test_movie_title, cosine_sim_matrix, movie_data_with_soup, indices)
            
            if not recommendations.empty:
                recs_str = "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(recommendations)])
                mlflow.log_text(recs_str, "sample_recommendations.txt")
            
            print(f"MAIN: MLflow Run ID: {run_id} selesai.", flush=True)
    else:
        print("\nMAIN: Gagal memuat data. Proses MLflow tidak dilanjutkan.", flush=True)

    print("\nSkrip modelling.py selesai.", flush=True)