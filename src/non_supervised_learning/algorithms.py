from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch

import numpy as np

class Algorithm(object):

    def apply_kmeans(
        self, 
        dataset=None, 
        n_clusters=8, 
        init="k-means++",
        n_init=10, 
        max_iter=300,
        tol=1e-4,
        algorithm='auto'):
        
        kmeans_instance = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=None,
            algorithm=algorithm)
        
        kmeans_instance.fit(dataset)
        return kmeans_instance
    
    def apply_minibatch(
        self, 
        dataset=None, 
        n_clusters=8,
        init="k-means++",
        max_iter=100, 
        batch_size=1024,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None, 
        n_init=3,
        reassignment_ratio=0.01):
        
        minibatch_instance = MiniBatchKMeans(
            n_clusters=n_clusters, 
            init=init,
            max_iter=max_iter, 
            batch_size=batch_size,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio)

        minibatch_instance.fit(dataset)
        return minibatch_instance
    
    def apply_affinity_propagation(
        self, 
        dataset=None, 
        damping=0.5, 
        max_iter=200, 
        convergence_iter=15,
        preference=None, 
        affinity="euclidean",
        random_state=None):

        affinity_instance = AffinityPropagation(
            damping=damping,
            max_iter=max_iter, 
            convergence_iter=convergence_iter,
            affinity=affinity,
            random_state=random_state,
            preference=preference)

        affinity_instance.fit(dataset)
        return affinity_instance
    
    def apply_mean_shift(
        self, 
        dataset=None,
        bandwidth=None,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        n_jobs=-1,
        max_iter=300):

        meanshift_instance = MeanShift(
            n_jobs=n_jobs,
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            max_iter=max_iter)
        
        meanshift_instance.fit(dataset)
        return meanshift_instance
    
    def apply_spectral_clustering(
        self, 
        dataset=None, 
        n_clusters=8, 
        eigen_solver=None, 
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        eigen_tol=0.0,
        n_neighbors=10,
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        n_jobs=-1):

        spectral_instance = SpectralClustering(
            n_clusters=n_clusters, 
            eigen_solver=eigen_solver, 
            random_state=random_state,
            n_init=n_init, 
            gamma=gamma,
            affinity=affinity,
            eigen_tol=eigen_tol,
            n_neighbors=n_neighbors,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            n_jobs=n_jobs)

        spectral_instance.fit(dataset)
        return spectral_instance
    
    def apply_agglomerative_clustering(
        self, 
        dataset=None, 
        n_clusters=2, 
        memory=None,
        connectivity=None,
        affinity = "euclidean",
        compute_full_tree="auto",
        linkage='ward',
        distance_threshold=None,
        compute_distances=False):

        agglomerative_instance = AgglomerativeClustering(
            n_clusters=n_clusters, 
            affinity=affinity,
            linkage=linkage,
            memory=memory,
            connectivity=connectivity,
            compute_distances=compute_distances,
            compute_full_tree=compute_full_tree,
            distance_threshold=distance_threshold)

        agglomerative_instance.fit(dataset)
        return agglomerative_instance
    
    
    def apply_DBSCAN(
        self, 
        dataset=None, 
        eps=0.5, 
        min_samples=5,
        metric="euclidean", 
        algorithm='auto',
        leaf_size=30,
        p=None,
        n_jobs=-1):

        dbscan_instance = DBSCAN(
            eps=eps, 
            min_samples=min_samples, 
            algorithm=algorithm,
            metric=metric,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs)

        dbscan_instance.fit(dataset)
        return dbscan_instance
    
    def apply_OPTICS(
        self, 
        dataset=None,
        min_samples=5,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        cluster_method="xi",
        eps=None,
        xi=0.05,
        predecessor_correction=True,
        min_cluster_size=None,
        algorithm="auto",
        leaf_size=30,
        n_jobs=-1):

        optics_instance = OPTICS(
            n_jobs=n_jobs,
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            p=p,
            cluster_method=cluster_method,
            eps=eps,
            xi=xi,
            predecessor_correction=predecessor_correction,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            leaf_size=leaf_size)

        optics_instance.fit(dataset)
        return optics_instance
    
    def apply_birch(
        self, 
        dataset=None, 
        threshold=0.5, 
        branching_factor=50, 
        n_clusters=3,
        compute_labels=True):

        birch_instance = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=compute_labels)

        birch_instance.fit(dataset)
        return birch_instance
    