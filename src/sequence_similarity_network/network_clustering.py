from scipy.spatial.distance import euclidean
from itertools import combinations
import pandas as pd
import subprocess
from os import remove
from networkx import Graph
import community as community_louvain

class SimilarityNetwork:
    def __init__(self, dataset, method, parameter, aproximation = "lower"):
        self.dataset = dataset
        self.method = method
        self.parameter = parameter
        self.aproximation = aproximation
    
    def get_distance_matrix(self):
        return None

    def __iqr(self, target, iqr_coeficient, aproximation):
        q3 = target.distance.quantile(.75)
        q1 = target.distance.quantile(.25)
        if aproximation == "upper":
            top = q3 + iqr_coeficient * abs(q3 - q1)
            return target[target.distance >= top]
        if aproximation == "lower":
            bottom = q1 - iqr_coeficient * abs(q3 - q1)
            return target[target.distance <= bottom]

    def __z_score(self, target, z_score_threshold, aproximation):
        """Z score outliers detection method"""
        mean = target.distance.mean()
        std = target.distance.std()
        if aproximation == "upper":
            return target[
                (target.distance - mean) / std >= z_score_threshold]
        if aproximation == "lower":
            return target[
                (target.distance - mean) / std <= -z_score_threshold]

    def __percentile(self, target, quantile, aproximation):
        if aproximation == "upper":
            return target[target.distance >= target.distance.quantile(quantile)]
        if aproximation == "lower":
            return target[target.distance <= 1 - target.distance.quantile(quantile)]
    
    def filter_distances(self):
        if self.method == "iqr":
            self.new_distances = self.__iqr(self.distance_df, self.parameter, self.aproximation)
        if self.method == "z_score":
            self.new_distances = self.__z_score(self.distance_df, self.parameter, self.aproximation)
        if self.method == "percentile":
            self.new_distances = self.__percentile(self.distance_df, self.parameter, self.aproximation)
    
    def create_graph(self):
        self.graph = Graph()
        for _, row in self.new_distances.iterrows():
            self.graph.add_edge(row.id_a, row.id_b, distance=row.distance)
    
    def get_community(self):
        partition = community_louvain.best_partition(self.graph)
        modularity_value = round(
            community_louvain.modularity(partition, self.graph), 3
        )
        for a in partition:
            self.dataset.loc[int(a), "cluster"] = partition[a]
        return modularity_value

    def get_community_clustering(self):
        self.get_distance_matrix()
        self.filter_distances()
        self.create_graph()
        modularity_value = self.get_community()
        return modularity_value, self.dataset



class NSSN(SimilarityNetwork):
    def __init__(self, dataset, method, parameter, aproximation = "lower"):
        super().__init__(dataset, method, parameter, aproximation)

    def get_distance_matrix(self):
        distances_list = []
        for a, b in combinations(self.dataset.iterrows(), 2):
            index_a = a[0]
            index_b = b[0]
            vector_a = a[1][1:]
            vector_b = b[1][1:]
            distances_list.append({
                "id_a": index_a,
                "id_b": index_b,
                "distance": euclidean(vector_a, vector_b)
            })
        self.distance_df = pd.DataFrame(distances_list)

class SSN(SimilarityNetwork):
    def __init__(self, dataset, method, parameter,
                 aproximation = "upper", clustal_path = "clustalo"):
        super().__init__(dataset, method, parameter, aproximation)
        self.clustal_path = clustal_path
        self.temp_file_path = "temp.fasta"
        self.output_dist_file = "distance_matrix.csv"
        self.df2fasta()

    def df2fasta(self):
        fasta_text = ""
        for _, row in self.dataset.iterrows():
            fasta_text += f">{row.id}\n{row.sequence}\n"
        with open(self.temp_file_path, encoding="utf-8", mode="w") as file:
            file.write(fasta_text)

    def get_distance_matrix(self):
        command = [
                self.clustal_path,
                "-i",
                self.temp_file_path,
                f"--distmat-out={self.output_dist_file}",
                "--full",
                "--force",
            ]
        subprocess.check_output(command)
        distance_table = pd.read_csv(
            self.output_dist_file, header=None, delimiter=r"\s+", skiprows=1
        )
        self.x_labels = distance_table[0]
        distance_table.drop([0], axis=1, inplace=True)
        self.z_values = distance_table.values
        distances_list = []
        for a, b in combinations(range(len(self.x_labels)), 2):
            distances_list.append({
                "id_a": a,
                "id_b": b,
                "distance": self.z_values[a, b]
            })
        self.distance_df = pd.DataFrame(distances_list)
        remove(self.temp_file_path)
        remove(self.output_dist_file)