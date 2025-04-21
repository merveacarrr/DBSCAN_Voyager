import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sqlalchemy import create_engine

def get_db_connection():
    """Veritabanı bağlantısını oluşturur"""
    return create_engine("postgresql+psycopg2://postgres:meri3516@localhost/postgres")

def find_optimal_eps(X_scaled, min_samples=3, plot_knee=False):
    """
    DBSCAN için optimal eps değerini bulur
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, min_samples-1])
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]
    
    if plot_knee:
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--', 
                   label=f'Optimal eps: {optimal_eps:.2f}')
        plt.title('Elbow Method for Optimal eps')
        plt.legend()
        plt.show()
    
    return optimal_eps

def plot_clusters(df, x_col, y_col, title="Cluster Visualization"):
    """
    Kümeleme sonuçlarını görselleştirir
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df[x_col], df[y_col], c=df['cluster'], 
                         cmap='viridis', s=60, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.show()