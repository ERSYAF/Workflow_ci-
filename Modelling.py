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



if __name__ == "__main__":

    mlflow.set_experiment("Movie Recommender - CI Run")
    

    path_ke_dataset = "tmdb_movies_processed.csv"
    
    movie_data_with_soup = load_data_and_generate_soup(path_ke_dataset)
    
    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        with mlflow.start_run(run_name="Automated_CI_Run") as run:

            tfidf_params = {"stop_words": 'english', "ngram_range": (1,2)}
            tfidf = TfidfVectorizer(**tfidf_params)
            tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'].fillna(''))
            
            mlflow.log_params(tfidf_params)
            mlflow.log_metric("tfidf_num_features", tfidf_matrix.shape[1])
            
            tfidf_path = "tfidf_vectorizer.pkl"
            joblib.dump(tfidf, tfidf_path)
            mlflow.log_artifact(tfidf_path, artifact_path="model")
            os.remove(tfidf_path) 
            
            print("Proses CI selesai dan log tersimpan di MLflow.")
    else:
        print("Gagal memuat data, proses CI dihentikan.")
