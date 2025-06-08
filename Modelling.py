print(">>> TES: SKRIP Modelling.py MULAI DI SINI !!! <<<", flush=True)
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow 
import mlflow.sklearn 
import joblib 


def clean_text_for_token(text):
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""

def create_content_soup(df):
    print("CREATE_CONTENT_SOUP: Memulai pembuatan 'content soup'...", flush=True)
    df_copy = df.copy()
    df_copy['overview_processed'] = df_copy['overview'].fillna('').astype(str).str.lower()
    def process_genres_list(genre_list):
        if isinstance(genre_list, list):
            return " ".join([clean_text_for_token(genre) for genre in genre_list])
        elif isinstance(genre_list, str):
            try:
                actual_list = ast.literal_eval(genre_list)
                return " ".join([clean_text_for_token(genre) for genre in actual_list])
            except: return ""
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
                       df_copy['genres_soup'] + ' ' + df_copy['genres_soup'] + ' ' +
                       df_copy['director_soup'] + ' ' + df_copy['director_soup'])
    print("CREATE_CONTENT_SOUP: 'Content soup' berhasil dibuat.", flush=True)
    return df_copy[['id', 'title', 'soup']]

def load_data_and_generate_soup(dataset_path):
    print(f"LOAD_AND_SOUP: Mencoba memuat dataset dari: {dataset_path}", flush=True)
    try:
        df = pd.read_csv(dataset_path)
        print(f"LOAD_AND_SOUP: Dataset berhasil dimuat. Baris: {len(df)}, Kolom: {len(df.columns)}", flush=True)
        print(f"LOAD_AND_SOUP: Kolom: {df.columns.tolist()}", flush=True)
        if not all(col in df.columns for col in ['id', 'title', 'overview', 'genres_processed', 'director']):
            print("LOAD_AND_SOUP: ERROR - Kolom penting tidak ada.", flush=True)
            return None
        return create_content_soup(df)
    except FileNotFoundError:
        print(f"LOAD_AND_SOUP: ERROR - File tidak ditemukan: {dataset_path}", flush=True)
        return None
    except Exception as e:
        print(f"LOAD_AND_SOUP: ERROR - Kesalahan lain: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def get_recommendations(movie_title, cosine_sim_matrix_input, data, movie_indices, top_n=10):
    print(f"\nRECOMMENDER: Mencari rekomendasi untuk film: '{movie_title}'")
    if movie_title not in movie_indices:
        print(f"RECOMMENDER: Judul film '{movie_title}' tidak ditemukan.")
        similar_titles = [title for title in movie_indices.index if movie_title.lower() in title.lower()]
        if similar_titles:
            print(f"RECOMMENDER: Mungkin maksud Anda: {similar_titles[:5]}?")
        return pd.Series(dtype='object') 
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix_input[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices_output = [i[0] for i in sim_scores]
    recommended_movies = data['title'].iloc[movie_indices_output]
    print(f"RECOMMENDER: Rekomendasi ditemukan.")
    return recommended_movies

if __name__ == "__main__":
    print(">>> BLOK __main__ di Modelling.py TERPANGGIL <<<", flush=True)


    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("MLflow Tracking URI berhasil diatur ke http://127.0.0.1:5000", flush=True)
    except Exception as e_mlflow_uri:
        print(f"Gagal mengatur MLflow Tracking URI: {e_mlflow_uri}. Pastikan server MLflow UI berjalan.", flush=True)


    experiment_name = "Movie Recommender - Content Based" 
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location="mlflow_artifacts_recommender") 
        print(f"Eksperimen '{experiment_name}' DIBUAT dengan ID: {experiment_id}", flush=True)
    else:
        experiment_id = experiment.experiment_id
        print(f"Eksperimen '{experiment_name}' SUDAH ADA dengan ID: {experiment_id}", flush=True)
    mlflow.set_experiment(experiment_name)


    path_ke_dataset_yang_sudah_diproses = r"D:\Kuliah\Coding Camp 2025\MSML_Era_Syafina\Eksperimen_SML_Era-Syafina\Membangun_Model\tmdb_movies_processed.csv" 
                                      

    print(f"MAIN: Akan memuat data dan membuat soup dari: {path_ke_dataset_yang_sudah_diproses}", flush=True)
    movie_data_with_soup = load_data_and_generate_soup(path_ke_dataset_yang_sudah_diproses)
    
    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        with mlflow.start_run(run_name="ContentBasedRecommender_Run1") as run:
            run_id = run.info.run_id
            print(f"\nMAIN: Memulai MLflow Run ID: {run_id} untuk eksperimen ID: {experiment_id}", flush=True)
            mlflow.set_tag("developer", "Era_Syafina")
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
            movie_data_with_soup['soup'] = movie_data_with_soup['soup'].fillna('')
            tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'])
            
            print(f"MAIN: Matriks TF-IDF: {tfidf_matrix.shape}", flush=True)
            mlflow.log_metric("tfidf_matrix_rows", tfidf_matrix.shape[0])
            mlflow.log_metric("tfidf_matrix_cols", tfidf_matrix.shape[1])


            tfidf_path = "tfidf_vectorizer.pkl"
            joblib.dump(tfidf, tfidf_path)
            mlflow.log_artifact(tfidf_path, artifact_path="tfidf_model") 
            print(f"MAIN: TF-IDF Vectorizer disimpan dan dilog ke MLflow.", flush=True)

            print("MAIN: Menghitung Cosine Similarity...", flush=True)
            cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            print(f"MAIN: Cosine Similarity Matrix: {cosine_sim_matrix.shape}", flush=True)
            mlflow.log_metric("cosine_sim_rows", cosine_sim_matrix.shape[0])
            mlflow.log_metric("cosine_sim_cols", cosine_sim_matrix.shape[1])


            cosine_sim_path = "cosine_similarity_matrix.npz"
            np.savez_compressed(cosine_sim_path, cosine_sim_matrix=cosine_sim_matrix)
            mlflow.log_artifact(cosine_sim_path, artifact_path="similarity_matrix")
            print(f"MAIN: Cosine Similarity Matrix disimpan dan dilog ke MLflow.", flush=True)


            movie_titles_ids_path = "movie_titles_ids.csv"
            movie_data_with_soup[['id', 'title']].to_csv(movie_titles_ids_path, index=False)
            mlflow.log_artifact(movie_titles_ids_path, artifact_path="data_references")
            print(f"MAIN: Movie titles and IDs disimpan dan dilog ke MLflow.", flush=True)

            indices = pd.Series(movie_data_with_soup.index, index=movie_data_with_soup['title']).drop_duplicates()
            test_movie_title_1 = movie_data_with_soup['title'].iloc[0]
            recommendations = get_recommendations(test_movie_title_1, cosine_sim_matrix, movie_data_with_soup, indices)
            
            print(f"\n--- Rekomendasi untuk '{test_movie_title_1}' ---")
            if not recommendations.empty:
                recs_str = "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(recommendations)])
                print(recs_str)
                mlflow.log_text(recs_str, "sample_recommendations_for_avatar.txt")
            else:
                print("Tidak ada rekomendasi ditemukan.")

            print(f"MAIN: MLflow Run ID: {run_id} selesai. Cek UI di http://127.0.0.1:5000", flush=True)
    else:
        print("\nMAIN: Gagal memuat data atau membuat 'soup'. Proses MLflow tidak dilanjutkan.", flush=True)

    print("\nSkrip modelling.py selesai (dengan MLflow).", flush=True)