import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN

class DBSCANVisualizer:
    def __init__(self, X, eps=0.5, min_samples=5):
        self.X = X
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None        
        self.clusters = None
        self.point_type = None
        self.fit()

    def fit(self):
        # Run DBSCAN clustering        
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.X)
        self.clusters = self.dbscan.labels_.astype(str)     

          
        

    def plot_clusters(self):
        # Handling '-1' as 'Noise'
        clusters = self.clusters.copy()
        clusters[clusters == '-1'] = 'Noise'

        # Create color map
        unique_clusters = np.unique(clusters)
        colors = ['grey' if cluster == 'Noise' else color 
                  for cluster, color in zip(unique_clusters, px.colors.qualitative.G10)]
        color_map = dict(zip(unique_clusters, colors))

        # Plot
        fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], 
                         color=clusters, color_discrete_map=color_map)        
        fig.update_layout(coloraxis_showscale=False, width=600, height=400, title='Data - DBSCAN Algorithm')
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x",scaleratio=1))
        fig.update_layout(legend_title_text='Cluster')
        fig.update_traces(marker=dict(size=8, opacity=0.7))        
        fig.show()


    def plot_point_types(self):
        # Determine point types
        self.point_type = np.full_like(self.dbscan.labels_, 'Border', dtype=object)
        self.point_type[self.dbscan.core_sample_indices_] = 'Core'
        self.point_type[self.dbscan.labels_ == -1] = 'Noise' 

        # Create symbol map
        symbol_map = {'Noise': 'circle-open-dot', 'Core': 'circle', 'Border': 'circle-open'}

        # Create a color map for clusters
        unique_clusters = np.unique(self.clusters)
        colors = ['grey' if cluster == '-1' else color for cluster, color in zip(unique_clusters, px.colors.qualitative.G10)]
        color_map = dict(zip(unique_clusters, colors))

        # Plot
        fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], 
                         symbol=self.point_type, symbol_map=symbol_map,     
                         color=self.clusters, color_discrete_map=color_map)
        fig.update_layout(coloraxis_showscale=False, width=600, height=400, title='Data - DBSCAN Algorithm')
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x",scaleratio=1))
        fig.update_layout(legend_title_text='Point Type')
        fig.update_traces(marker=dict(size=8, opacity=0.7))        
        fig.show()


