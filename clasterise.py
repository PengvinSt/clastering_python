import numpy as np
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
from logging import Logger
from sklearn.manifold import TSNE
import math

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
        self.logger.info("Generating DBSCAN clusters (self-implemented)...")
        if self.histograms_matrix.shape[0] == 0:
            self.logger.warning("No histogram data to cluster (matrix is empty).")
            self.cluster_labels_ = np.array([], dtype=int)
            return self.cluster_labels_

        samples = self.histograms_matrix.shape[0]

        if samples < self.min_pts_value:
            self.logger.warning(
                f"Number of samples ({samples}) is less than min_pts_value ({self.min_pts_value}). "
                "All points might be classified as noise unless eps_value is very large.")

        labels = np.full(samples, self.POINT_UNVISITED, dtype=int)
        current_cluster_id = 0

        for point_idx in range(samples):
            if labels[point_idx] != self.POINT_UNVISITED:
                continue

            neighbor_indices = self.__region_query(point_idx)

            if len(neighbor_indices) < self.min_pts_value:
                labels[point_idx] = self.POINT_NOISE
            else:
                current_cluster_id += 1
                labels[point_idx] = current_cluster_id
                self.__expand_cluster_from_core(neighbor_indices, labels, current_cluster_id)

        self.cluster_labels_ = labels
        self.logger.info(f"DBSCAN processing finished. Found {current_cluster_id} cluster(s) (excluding noise).")
        return self.cluster_labels_


    def __calculate_nnds(self, cluster_indices: List[int], point_matrix: np.ndarray) -> List[float]:
        nnds = []
        num_points_in_cluster = len(cluster_indices)

        if num_points_in_cluster < 2:
            return [] # NNDS is not well-defined for clusters with < 2 points.

        current_cluster_points = point_matrix[cluster_indices]

        for i in range(num_points_in_cluster):
            p_vector = current_cluster_points[i]
            min_dist_sq = float('inf')

            for j in range(num_points_in_cluster):
                if i == j:
                    continue

                q_vector = current_cluster_points[j]
                dist_sq = np.sum((p_vector - q_vector) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq

            if min_dist_sq == float('inf'):
                # Hopes this will never run ...
                self.logger.warning(f"Point {cluster_indices[i]} in cluster has no neighbors. NND set to infinity.")
                nnds.append(float('inf'))
            else:
                nnds.append(np.sqrt(min_dist_sq))

        return nnds

    def __get_volume_unit_d_ball(self, d: int) -> float:
        if d < 0: raise ValueError("Dimension d must be non-negative.")
        if d == 0: return 1.0
        return math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    def _chi_squared_critical_value(self, df: int, alpha: float) -> float:
        # For alpha = 0.05
        crit_vals_0_05 = {
            1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
            6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
        }
        if df <= 0: return float('inf')

        if df in crit_vals_0_05:
            return crit_vals_0_05[df]
        elif df > 10:
            return crit_vals_0_05[10] + 1.5 * (df - 10) # Heuristic increase
        else:
            return crit_vals_0_05.get(df, crit_vals_0_05[1]) # Fallback


    def __chi_squared_goodness_of_fit_test(
            self,
            nnds: List[float],
            num_points_in_cluster: int,
            dimensions: int,
            significance_level: float = 0.05
    ) -> bool:
        if not nnds or num_points_in_cluster < max(3, dimensions + 1) or not all(math.isfinite(d) for d in nnds):
            self.logger.debug(f"Chi-squared test skipped: not enough points ({num_points_in_cluster}) or invalid NNDs. Defaulting to valid.")
            return True

        # Binning NNDs
        num_bins = max(2, int(math.sqrt(num_points_in_cluster)))

        min_nnd_val, max_nnd_val = min(nnds), max(nnds)
        if abs(min_nnd_val - max_nnd_val) < 1e-9 and num_points_in_cluster > 1:
            # All NNDs effectively zero for multiple points -> not a spatially distributed cluster.
            if min_nnd_val < 1e-9:
                self.logger.debug("Chi-squared test: All NNDs are zero for multiple points. Likely invalid distribution.")
                return False
            # force 1 bin when all NNDs are identical !== 0
            num_bins = 1 if num_bins > 1 else num_bins # Keep at least 1 bin

        try:
            hist_observed, bin_edges = np.histogram(nnds, bins=num_bins)
        except ValueError as e:
            self.logger.warning(f"Chi-squared test: Error during histogramming: {e}. Assuming valid.")
            return True

        observed_frequencies = hist_observed

        # Estimate lambda
        mean_nnd = np.mean(nnds)
        if mean_nnd < 1e-9:
            self.logger.debug(f"Chi-squared test: Mean NND is ~0 for {num_points_in_cluster} points. Assuming invalid.")
            return num_points_in_cluster <= 1

        vol_unit_ball = self.__get_volume_unit_d_ball(dimensions)
        if vol_unit_ball < 1e-9:
            self.logger.warning("Chi-squared test: Volume of unit ball is ~0. Cannot proceed.")
            return False

        try:
            gamma_val_for_lambda = math.gamma(1.0 / dimensions + 1.0)
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Chi-squared test: Error calculating gamma function: {e}. Cannot estimate lambda.")
            return True

        try:
            lambda_estimate_factor = (gamma_val_for_lambda / mean_nnd) ** dimensions
            lambda_estimate = lambda_estimate_factor / vol_unit_ball
        except OverflowError:
            self.logger.warning("Chi-squared test: Overflow during lambda estimation (mean_nnd too small?). Assuming invalid.")
            return False

        if lambda_estimate < 1e-9 or not math.isfinite(lambda_estimate):
            self.logger.warning(f"Chi-squared test: Unreliable lambda estimate ({lambda_estimate}). Defaulting validity.")
            return True

        # Calc expected frequencies
        expected_frequencies = np.zeros(num_bins, dtype=float)
        cdf = lambda r_val: 1.0 - math.exp(-lambda_estimate * vol_unit_ball * (r_val ** dimensions)) if r_val >= 0 else 0.0

        for i in range(num_bins):
            r_lower = bin_edges[i]
            r_upper = bin_edges[i + 1]

            prob_bin = 0.0
            if r_upper > r_lower:
                try:
                    prob_bin = cdf(r_upper) - cdf(r_lower)
                except OverflowError:
                    self.logger.warning(f"Chi-squared test: Overflow calculating CDF for bin {i}. Assuming zero probability for bin.")
                    prob_bin = 0

            expected_frequencies[i] = num_points_in_cluster * prob_bin

        # Prevent  smth/zero
        expected_frequencies[expected_frequencies < 1e-6] = 1e-6

        # Compute Chi squared stat
        chi_squared_statistic = np.sum(
            ((observed_frequencies - expected_frequencies) ** 2) / expected_frequencies
        )
        if not math.isfinite(chi_squared_statistic):
            self.logger.warning("Chi-squared test: Statistic is not finite. Assuming invalid.")
            return False

        df = num_bins - 2 # (k - 1 - m, where k=num_bins, m=1 estimated parmtr)
        if df <= 0: # Not enough degrees of freedom for a proper test
            self.logger.debug(f"Chi-squared test: Not enough degrees of freedom ({df}). Assuming valid due to small N.")
            return True

        # Compare with critical value
        critical_value = self._chi_squared_critical_value(df, significance_level)
        self.logger.debug(f"Chi-squared test: stat={chi_squared_statistic:.2f}, df={df}, crit_val@{significance_level*100}%={critical_value:.2f}. Decision: {chi_squared_statistic < critical_value}")

        return chi_squared_statistic < critical_value

    def __get_potential_candidates_for_cluster(
            self,
            current_cluster_indices: List[int],
            all_labels: np.ndarray,
            point_matrix: np.ndarray
    ) -> List[int]:
        potential_candidates_with_distances = []
        if not current_cluster_indices:
            return []

        # Get actual point vectors for current cluster members
        current_cluster_vectors = point_matrix[current_cluster_indices]

        for i in range(point_matrix.shape[0]):
            if all_labels[i] == self.POINT_UNVISITED:
                candidate_vector = point_matrix[i]
                # Calc minimum distance from candidate to any point in the cluster
                distances_to_cluster_members = np.linalg.norm(current_cluster_vectors - candidate_vector, axis=1)
                min_dist_to_cluster = np.min(distances_to_cluster_members)
                potential_candidates_with_distances.append((min_dist_to_cluster, i))

        # Sort candidates by their distance to the cluster:ASC
        potential_candidates_with_distances.sort(key=lambda x: x[0])
        sorted_candidate_indices = [idx for dist, idx in potential_candidates_with_distances]
        return sorted_candidate_indices


    def __generate_dbclasd(self) -> Optional[np.ndarray]:
        self.logger.info("Generating DBCLASD clusters (self-implemented)...")
        if self.histograms_matrix.shape[0] == 0:
            self.logger.warning("No histogram data to cluster (matrix is empty or has 0 samples).")
            self.cluster_labels_ = np.array([], dtype=int)
            return self.cluster_labels_

        n_samples = self.histograms_matrix.shape[0]
        n_dimensions = self.histograms_matrix.shape[1]

        if n_dimensions == 0:
            self.logger.error("Error while attempting to cluster: data has 0 dimensions.")
            self.cluster_labels_ = np.array([], dtype=int)
            return self.cluster_labels_

        dbclasd_min_final_cluster_size = max(self.min_pts_value, 2)
        self.logger.info(f"DBCLASD minimum final cluster size set to: {dbclasd_min_final_cluster_size}")

        labels = np.full(n_samples, self.POINT_UNVISITED, dtype=int)
        # Process points in a random order to avoid sensitivity to initial data ordering
        processing_order = np.random.permutation(n_samples)

        current_cluster_id = 0

        for point_idx in processing_order:
            if labels[point_idx] == self.POINT_UNVISITED:
                current_cluster_id += 1
                current_cluster_indices = [point_idx]
                labels[point_idx] = current_cluster_id
                self.logger.debug(f"Attempting to form cluster {current_cluster_id} starting with point {point_idx}")

                # Cluster Bootstrapping Phase: ensure at least 2 points for NNDS
                if len(current_cluster_indices) == 1:
                    seed_vector = self.histograms_matrix[point_idx]
                    unvisited_indices = np.where(labels == self.POINT_UNVISITED)[0]
                    if len(unvisited_indices) > 0:
                        # Find the nearest unassigned neighbor to the seed point
                        dists_to_unvisited = np.linalg.norm(self.histograms_matrix[unvisited_indices] - seed_vector, axis=1)
                        nearest_unvisited_idx_in_list = np.argmin(dists_to_unvisited)
                        best_bootstrap_candidate_idx = unvisited_indices[nearest_unvisited_idx_in_list]

                        tentative_bootstrap_cluster = current_cluster_indices + [best_bootstrap_candidate_idx]
                        nnds_bootstrap = self.__calculate_nnds(tentative_bootstrap_cluster, self.histograms_matrix)

                        if self.__chi_squared_goodness_of_fit_test(nnds_bootstrap, len(tentative_bootstrap_cluster), n_dimensions):
                            current_cluster_indices.append(best_bootstrap_candidate_idx)
                            labels[best_bootstrap_candidate_idx] = current_cluster_id
                            self.logger.debug(f"  Bootstrapped cluster {current_cluster_id} with point {best_bootstrap_candidate_idx}. Size: {len(current_cluster_indices)}")
                        else:
                            self.logger.debug(f"  Bootstrap with point {best_bootstrap_candidate_idx} failed distribution test. Cluster {current_cluster_id} remains size 1 for now.")
                    else:
                        self.logger.debug(f"  No unvisited points left to bootstrap cluster {current_cluster_id}.")


                # Cluster Expansion Phase
                was_expanded_in_iteration = True
                while was_expanded_in_iteration:
                    was_expanded_in_iteration = False

                    if len(current_cluster_indices) < 2:
                        self.logger.debug(f"  Cluster {current_cluster_id} has < 2 points. Stopping expansion.")
                        break

                    # Check current cluster validity BEFORE attempting to add more points
                    current_nnds = self.__calculate_nnds(current_cluster_indices, self.histograms_matrix)
                    if not self.__chi_squared_goodness_of_fit_test(current_nnds, len(current_cluster_indices), n_dimensions):
                        self.logger.debug(f"  Cluster {current_cluster_id} (size {len(current_cluster_indices)}) failed distribution test. Stopping expansion.")
                        break

                    potential_candidates = self.__get_potential_candidates_for_cluster(
                        current_cluster_indices, labels, self.histograms_matrix
                    )

                    candidate_added_this_iteration = False
                    for candidate_idx in potential_candidates:
                        tentative_cluster_indices = current_cluster_indices + [candidate_idx]
                        tentative_nnds = self.__calculate_nnds(tentative_cluster_indices, self.histograms_matrix)

                        if self.__chi_squared_goodness_of_fit_test(tentative_nnds, len(tentative_cluster_indices), n_dimensions):
                            current_cluster_indices.append(candidate_idx)
                            labels[candidate_idx] = current_cluster_id
                            was_expanded_in_iteration = True
                            candidate_added_this_iteration = True
                            self.logger.debug(f"  Added point {candidate_idx} to cluster {current_cluster_id}. New size: {len(current_cluster_indices)}")
                            break # DBCLASD adds one point per iteration

                    if not candidate_added_this_iteration:
                        self.logger.debug(f"  No suitable candidates found for cluster {current_cluster_id}. Stopping expansion.")
                        break # No candidate improved the fit, stop expanding

                # Post-expansion check for cluster validity
                if len(current_cluster_indices) < dbclasd_min_final_cluster_size:
                    self.logger.debug(f"  Cluster {current_cluster_id} (size {len(current_cluster_indices)}) is smaller than min_pts ({dbclasd_min_final_cluster_size}). Marking points as noise.")
                    for pt_idx_in_small_cluster in current_cluster_indices:
                        if labels[pt_idx_in_small_cluster] == current_cluster_id: # Ensure it has not been reassigned (how for DBCLASD???)
                            labels[pt_idx_in_small_cluster] = self.POINT_NOISE


        # Renumber cluster ids to be consecutive (eg 1, 2, 3...)
        unique_final_labels = np.unique(labels)
        cluster_map = {}
        new_cluster_id_counter = 1
        for lbl in sorted(list(unique_final_labels)):
            if lbl > 0: # Only remap actual clusters, not noise or unvisited (if any left)
                cluster_map[lbl] = new_cluster_id_counter
                new_cluster_id_counter += 1

        final_labels_remapped = np.copy(labels)
        for i in range(n_samples):
            old_label = labels[i]
            if old_label in cluster_map:
                final_labels_remapped[i] = cluster_map[old_label]
            elif old_label == self.POINT_UNVISITED:
                # Should not happen if all points are processed, but just in case.
                final_labels_remapped[i] = self.POINT_NOISE

        self.cluster_labels_ = final_labels_remapped
        num_final_clusters = new_cluster_id_counter - 1
        self.logger.info(f"DBCLASD processing finished. Found {num_final_clusters} cluster(s) (excluding noise).")

        return self.cluster_labels_

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

        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_value, max_iter=400, random_state=42, init='pca',
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
                cluster_files = self.file_names[self.cluster_labels_ == label_val]
                self.logger.debug(f"    Files in Cluster {label_val}: {', '.join(cluster_files)}")
        self.logger.info("----------------------------------\n")

    def generate_clusters(self) -> Optional[np.ndarray]:
        self._log_initial_info()
        if self.clustering_type == "DBSCAN":
            self.__generate_dbscan()
        elif self.clustering_type == "DBCLASD":
            self.__generate_dbclasd()
        else:
            raise ClusterizeError(f"Unsupported type: {self.clustering_type}")
        self.log_cluster_results()

        return self.cluster_labels_