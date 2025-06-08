print(">>> TES: SKRIP Modelling.py (dengan Tuning) MULAI DI SINI !!! <<<", flush=True)
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import joblib
import time 
import traceback

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
    print(">>> BLOK __main__ di Modelling.py (Tuning) TERPANGGIL <<<", flush=True)

    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("MLflow Tracking URI berhasil diatur ke http://127.0.0.1:5000", flush=True)
    except Exception as e_mlflow_uri:
        print(f"Gagal mengatur MLflow Tracking URI: {e_mlflow_uri}. Pastikan server MLflow UI berjalan.", flush=True)


    experiment_name = "Movie Recommender - TFIDF Tuning" 
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = "mlruns_tfidf_tuning_artifacts" 
        if not os.path.exists(artifact_location):
            os.makedirs(artifact_location)
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        print(f"Eksperimen '{experiment_name}' DIBUAT dengan ID: {experiment_id}", flush=True)
    else:
        experiment_id = experiment.experiment_id
        print(f"Eksperimen '{experiment_name}' SUDAH ADA dengan ID: {experiment_id}", flush=True)
    mlflow.set_experiment(experiment_name)

    path_ke_dataset_yang_sudah_diproses = r"D:\Kuliah\Coding Camp 2025\MSML_Era_Syafina\Eksperimen_SML_Era-Syafina\Membangun_Model\tmdb_movies_processed.csv"
    
    print(f"MAIN: Akan memuat data dan membuat soup dari: {path_ke_dataset_yang_sudah_diproses}", flush=True)
    print(f"  CWD saat ini (seharusnya): {os.getcwd()}", flush=True) 
    print(f"  Mencoba memuat dari (abspath): {os.path.abspath(path_ke_dataset_yang_sudah_diproses)}", flush=True)
        
    movie_data_with_soup_original = load_data_and_generate_soup(path_ke_dataset_yang_sudah_diproses)
    
    
    print(f"MAIN: Akan memuat data dan membuat soup dari: {path_ke_dataset_yang_sudah_diproses}", flush=True)
    movie_data_with_soup_original = load_data_and_generate_soup(path_ke_dataset_yang_sudah_diproses)
    
    if movie_data_with_soup_original is None or movie_data_with_soup_original.empty:
        print("\nMAIN: Gagal memuat data atau membuat 'soup'. Proses tuning dihentikan.", flush=True)
    else:
        print(f"\nMAIN: Data asli dimuat, jumlah film: {len(movie_data_with_soup_original)}", flush=True)

        tfidf_param_sets = [
            {"stop_words": 'english', "ngram_range": (1,2), "min_df": 3, "max_df": 0.7, "sublinear_tf": False, "norm": 'l2'},
            {"stop_words": 'english', "ngram_range": (1,1), "min_df": 5, "max_df": 0.5, "sublinear_tf": False, "norm": 'l2', "max_features": 10000},
            {"stop_words": 'english', "ngram_range": (1,2), "min_df": 2, "max_df": 0.8, "sublinear_tf": True, "norm": 'l2'},
            {"stop_words": 'english', "ngram_range": (1,3), "min_df": 3, "max_df": 0.7, "sublinear_tf": False, "norm": 'l1'},
        ]

        test_movie_titles = [movie_data_with_soup_original['title'].iloc[0]] 
        if len(movie_data_with_soup_original) > 100: 
             title_100 = movie_data_with_soup_original['title'].iloc[100]
             if title_100 not in test_movie_titles: test_movie_titles.append(title_100)


        for i, params in enumerate(tfidf_param_sets):
            run_name = f"TFIDF_Tune_Run_{i+1}"
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                print(f"\n======== Memulai MLflow Run: {run_name} (ID: {run_id}) ========", flush=True)
                mlflow.set_tag("tuning_round", "1")
                mlflow.set_tag("tfidf_params_set", f"set_{i+1}")

                mlflow.log_params(params)
                print(f"PARAMS: {params}", flush=True)

                movie_data_run = movie_data_with_soup_original.copy()

                try:
                    start_time_total = time.time() # Mulai timer


                    print("TF-IDF: Memulai Vectorization...", flush=True)
                    tfidf = TfidfVectorizer(**params)
                    movie_data_run['soup'] = movie_data_run['soup'].fillna('')
                    tfidf_matrix = tfidf.fit_transform(movie_data_run['soup'])
                    
                    tfidf_shape = tfidf_matrix.shape
                    print(f"TF-IDF: Matriks TF-IDF: {tfidf_shape}", flush=True)
                    mlflow.log_metric("tfidf_matrix_rows", tfidf_shape[0])
                    mlflow.log_metric("tfidf_matrix_cols", tfidf_shape[1]) 
                    
                    tfidf_path = f"tfidf_vectorizer_run_{run_id}.pkl"
                    joblib.dump(tfidf, tfidf_path)
                    mlflow.log_artifact(tfidf_path, artifact_path="tfidf_model")
                    os.remove(tfidf_path) 
                    print("TF-IDF: Vectorizer disimpan dan dilog.", flush=True)

    
                    print("SIMILARITY: Menghitung Cosine Similarity...", flush=True)
                    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    cosine_shape = cosine_sim_matrix.shape
                    print(f"SIMILARITY: Cosine Similarity Matrix: {cosine_shape}", flush=True)
                    mlflow.log_metric("cosine_sim_rows", cosine_shape[0])
                    mlflow.log_metric("cosine_sim_cols", cosine_shape[1])

              
                    indices_run = pd.Series(movie_data_run.index, index=movie_data_run['title']).drop_duplicates()
                    
                    all_recs_str = ""
                    for test_title in test_movie_titles:
                        if test_title in indices_run:
                            recommendations = get_recommendations(test_title, cosine_sim_matrix, movie_data_run, indices_run, top_n=5) 
                            all_recs_str += f"--- Rekomendasi untuk '{test_title}' ---\n"
                            if not recommendations.empty:
                                for j, movie in enumerate(recommendations):
                                    all_recs_str += f"{j+1}. {movie}\n"
                            else:
                                all_recs_str += "Tidak ada rekomendasi ditemukan.\n"
                            all_recs_str += "\n"
                        else:
                            all_recs_str += f"Film tes '{test_title}' tidak ditemukan di dataset run ini.\n\n"
                    
                    if all_recs_str:
                        mlflow.log_text(all_recs_str, f"sample_recommendations_run_{run_id}.txt")
                        print(f"RECOMMENDATIONS: Contoh rekomendasi dilog sebagai artefak.", flush=True)
                        print(all_recs_str)

                    end_time_total = time.time()
                    mlflow.log_metric("total_processing_time_seconds", end_time_total - start_time_total)
                    print(f"TIME: Total waktu proses untuk run ini: {end_time_total - start_time_total:.2f} detik", flush=True)

                except Exception as e_run:
                    print(f"ERROR dalam MLflow Run {run_name}: {e_run}", flush=True)
                    traceback.print_exc()
                    mlflow.set_tag("run_status", "failed")
                    mlflow.log_param("error_message", str(e_run)[:250])

                print(f"======== Selesai MLflow Run: {run_name} (ID: {run_id}) ========", flush=True)

    print("\nSkrip modelling.py (Tuning) selesai.", flush=True)