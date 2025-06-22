import numpy as np
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
from logging import Logger
from sklearn.manifold import TSNE

class ClusterizeError(Exception):
    pass

class Clusterize:
    ClusteringLogicType = Literal["DBSCAN", "DBCLASD"]
    POINT_NOISE = -1
    POINT_UNVISITED = 0
    POINT_CORE = 1
    CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                          '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
                          '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']

    def __check_data(self, eps_value: float, min_pts_value: int, histograms: np.ndarray, file_names: List[str]):
        if not isinstance(histograms, np.ndarray) or histograms.ndim != 2:
            raise TypeError("histograms must be a 2D NumPy array.")
        if histograms.shape[0] == 0:
            if len(file_names) != 0:
                raise ValueError("Received file names but no histograms.")
            return

        first_hist_dim = histograms.shape[1]
        if eps_value <= 0:
            raise ValueError("eps_value must be positive number.")
        elif min_pts_value <= 0:
            raise ValueError("min_pts_value must be positive number.")
        elif len(histograms) != len(file_names):  # histograms.shape[0] is the number of samples
            raise ValueError(
                f"Number of histograms ({histograms.shape[0]}) and file_names ({len(file_names)}) must have the same length.")
        elif not isinstance(file_names, list) or not all(isinstance(f, str) for f in file_names):
            raise TypeError("file_names must be a list of strings.")


    # Change the type hint for histograms to np.ndarray
    def __init__(self, eps_value: float, min_pts_value: int, histograms: np.ndarray,
                 file_names: List[str], logger: Logger, clustering_type: ClusteringLogicType = "DBSCAN") -> None:
        self.histograms_matrix = histograms
        self.file_names = np.array(file_names)

        self.__check_data(eps_value, min_pts_value, self.histograms_matrix,
                          list(self.file_names))
        self.logger = logger
        self.eps_value = eps_value
        self.min_pts_value = min_pts_value
        self.clustering_type = clustering_type
        self.cluster_labels_: Optional[np.ndarray] = None

    def _log_initial_info(self) -> None:
        self.logger.info("--- Starting Clustering Process ---")
        self.logger.info(f"Clustering method: {self.clustering_type}")
        self.logger.info(f"Parameters: eps={self.eps_value}, min_samples={self.min_pts_value}")
        if self.histograms_matrix.shape[0] > 0:
            self.logger.info(
                f"Processing {self.histograms_matrix.shape[0]} videos with feature vectors of size {self.histograms_matrix.shape[1]}.")
        else:
            self.logger.info("No video data to process (histograms matrix is empty).")
        self.logger.info("------------------------------------")


    # find all points in the neighborhood of a point
    def __region_query(self, point_idx: int) -> np.ndarray:
        current_point_vector = self.histograms_matrix[point_idx]
        # calculate the distance between the current point and all other points in the matrix
        distances = np.linalg.norm(self.histograms_matrix - current_point_vector, axis=1)
        # find all points indexes that are within the radius of the current point (dist is <= eps)
        neighbor_indices = np.where(distances <= self.eps_value)[0]
        return neighbor_indices

    # expand cluster from core point
    def __expand_cluster_from_core(self, initial_neighbor_indices: np.ndarray, labels: np.ndarray, current_cluster_id: int) -> None:
        seeds = list(initial_neighbor_indices) # list to process
        head_idx = 0  # current point that processes

        # while seeds have points to process
        while head_idx < len(seeds):
            current_seed_idx = seeds[head_idx]  # get next point to process
            head_idx += 1  # move pointer

            # if point was NOISE before - it is now POINT_BORDER point, ignore it for now
            if labels[current_seed_idx] == self.POINT_NOISE:
                labels[current_seed_idx] = current_cluster_id

            # if point was POINT_UNVISITED before
            elif labels[current_seed_idx] == self.POINT_UNVISITED:
                # set label of current claster
                labels[current_seed_idx] = current_cluster_id

                # get current point neighbors
                current_seed_neighbors = self.__region_query(current_seed_idx)

                # if current_seed_idx is POINT_CORE
                if len(current_seed_neighbors) >= self.min_pts_value:
                    # add her neighbors to list to process if they are POINT_UNVISITED or POINT_NOISE
                    for neighbor_of_seed_idx in current_seed_neighbors:
                        if labels[neighbor_of_seed_idx] == self.POINT_UNVISITED or labels[neighbor_of_seed_idx] == self.POINT_NOISE:
                            seeds.append(neighbor_of_seed_idx)
            # if point > 0 (is already in some cluster) ignore it - DBSCAN algorithm is not united clasteers if point is already in some cluster
            else:
                continue

    def __generate_dbscan(self) -> Optional[np.ndarray]:
        print("Generating DBSCAN clusters with histograms and file names.")
        if not hasattr(self, 'histograms_matrix') or self.histograms_matrix.shape[0] == 0:
            print("No histogram data to cluster (matrix is empty or has 0 samples).")
            self.cluster_labels_ = np.array([], dtype=int)
            return self.cluster_labels_

        samples = self.histograms_matrix.shape[0]

        if samples < self.min_pts_value:
            print(
                f"Warning: Number of samples ({samples}) is less than min_pts_value ({self.min_pts_value}). "
                "All points might be classified as noise unless eps_value is very large.")

        labels = np.full(samples, 0, dtype=int)
        current_cluster_id = 0

        for point_idx in range(samples):
            # skip if point was processed before
            if labels[point_idx] != self.POINT_UNVISITED:
                continue

            neighbor_indices = self.__region_query(point_idx)

            # check if point is a POINT_CORE
            if len(neighbor_indices) < self.min_pts_value:
                labels[point_idx] = self.POINT_NOISE
            else:
                current_cluster_id += 1
                labels[point_idx] = current_cluster_id
                self.__expand_cluster_from_core(neighbor_indices, labels, current_cluster_id)

        self.cluster_labels_ = labels
        print(f"DBSCAN processing finished. Found {current_cluster_id} cluster(s) (excluding noise).")
        if samples > 0:
            unique_labels_found, counts = np.unique(self.cluster_labels_, return_counts=True)
            for label_val, count in zip(unique_labels_found, counts):
                if label_val == -1:
                    print(f"  Noise points: {count}")
                else:
                    print(f"  Cluster {label_val}: {count} points")
        else:
            print("  No data points were processed.")


        return self.cluster_labels_


    def _generate_dbclasd(self) -> Optional[np.ndarray]:
        self.logger.warning("DBCLASD is not implemented. Returning None.")
        return None

    def visualize_clusters(self) -> None:
        if self.cluster_labels_ is None or self.histograms_matrix.shape[0] == 0:
            self.logger.error("Cannot create plot. Run generate_clusters() first.")
            return

        self.logger.info("Preparing visualization: reducing dimensionality with t-SNE...")

        n_samples = self.histograms_matrix.shape[0]

        if n_samples <= 1:
            self.logger.warning("Cannot run t-SNE and visualize with only 1 data point. Skipping.")
            return

        perplexity_value = min(30.0, float(n_samples - 1)) if n_samples > 1 else 1.0 # Ensure perplexity is valid

        if n_samples <= 1:
            self.logger.warning("Cannot run t-SNE and visualize with only 1 data point. Skipping.")
            return

        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_value, n_iter=400, random_state=42, init='pca',
                    learning_rate='auto')
        vis_data = tsne.fit_transform(self.histograms_matrix)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(16, 12))

        unique_labels = sorted(list(np.unique(self.cluster_labels_)))

        for i, label in enumerate(unique_labels):
            mask = (self.cluster_labels_ == label)
            count = np.sum(mask)

            if label == -1:
                color = 'gray'
                legend_label = f'Noise ({count} videos)'
                marker = 'x'
                size = 50
                alpha = 0.7
            else:
                color = self.CLUSTER_COLORS[label % len(self.CLUSTER_COLORS)]
                legend_label = f'Cluster {label} ({count} videos)'
                marker = 'o'
                size = 80
                alpha = 0.9

            plt.scatter(vis_data[mask, 0], vis_data[mask, 1],
                        c=color, label=legend_label, s=size,
                        marker=marker, alpha=alpha, edgecolors='black', linewidths=0.5)

        for i in range(n_samples):
            plt.annotate(str(self.file_names[i]),
                         (vis_data[i, 0], vis_data[i, 1]),
                         textcoords="offset points",
                         xytext=(5, 5),
                         ha='left',
                         fontsize=8,
                         alpha=0.8)

        plt.title(f'Video Cluster Map (Method: {self.clustering_type})', fontsize=18, fontweight='bold')
        plt.suptitle('Each point is a video. Points close to each other are visually similar.', fontsize=12, y=0.92)
        plt.xlabel("t-SNE Feature 1", fontsize=12)
        plt.ylabel("t-SNE Feature 2", fontsize=12)
        plt.legend(loc='best', fontsize=10, shadow=True, fancybox=True, title="Video Groups")
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        self.logger.info("Displaying clustering plot.")
        plt.show()

    def log_cluster_results(self) -> None:
        if self.cluster_labels_ is None:
            self.logger.info("No clustering results to log.")
            return

        unique_labels, counts = np.unique(self.cluster_labels_, return_counts=True)
        self.logger.info("\n--- Clustering Results Summary ---")
        total_points = len(self.cluster_labels_)
        if total_points == 0:
            self.logger.info("No points were clustered.")
            return

        for label_val, count in zip(unique_labels, counts):
            percentage = (count / total_points) * 100
            if label_val == -1:
                self.logger.info(f"  Noise points: {count} ({percentage:.2f}%)")
            else:
                self.logger.info(f"  Cluster {label_val}: {count} points ({percentage:.2f}%)")
                # Optionally, log the files in each cluster
                cluster_files = self.file_names[self.cluster_labels_ == label_val]
                self.logger.debug(f"    Files in Cluster {label_val}: {', '.join(cluster_files)}")
        self.logger.info("----------------------------------\n")

    def generate_clusters(self) -> Optional[np.ndarray]:
        self._log_initial_info()
        if self.clustering_type == "DBSCAN":
            self.__generate_dbscan()
        elif self.clustering_type == "DBCLASD":
            self._generate_dbclasd()
        else:
            raise ClusterizeError(f"Unsupported type: {self.clustering_type}")
        self.log_cluster_results()

        return self.cluster_labels_