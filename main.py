import os
import argparse
import numpy as np
from extract import Extractor
from clasterise import Clusterize
from logger import setup_logger
from sklearn.preprocessing import StandardScaler



def main(args):
    logger = setup_logger()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input")

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        logger.info(f"Created input folder at: {input_folder}")
        logger.info("Please add some .mp4 files to the input folder.")

    logger.info("--- Initializing with parameters ---")
    logger.info(f"Input Folder: {input_folder}")
    logger.info(f"Extractor Parameters:")
    logger.info(f"  Use Optical Flow: {not args.no_flow}")
    logger.info(f"  LBP Points (default): {Extractor.DEFAULT_LBP_POINTS}")
    logger.info(f"  LBP Radius (default): {Extractor.DEFAULT_LBP_RADIUS}")
    logger.info(f"  Histogram Bins (default): {Extractor.DEFAULT_HIST_BINS}")
    logger.info(f"  Sample Rate (default): {Extractor.DEFAULT_SAMPLE_RATE}")
    logger.info(f"  Max Workers (default): {Extractor.DEFAULT_MAX_WORKERS}")

    logger.info(f"Clustering Parameters:")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Epsilon (eps_value): {args.eps}")
    logger.info(f"  Min Points (min_pts_value): {args.min_pts}")
    logger.info("------------------------------------")

    logger.info("--- Starting Histogram Extraction ---")
    extractor_params = {
        'use_optical_flow': not args.no_flow
    }
    extractor = Extractor(input_folder, logger, **extractor_params)
    histograms, file_names = extractor.extract_histograms()

    if histograms:
        logger.info("--- Extraction complete ---")

        logger.info("Scaling feature vectors...")
        histograms_matrix = np.array(histograms)

        scaler = StandardScaler()
        scaled_histograms = scaler.fit_transform(histograms_matrix)

        logger.info("Feature scaling complete.")

        logger.info(f"Successfully extracted histograms for {len(histograms)} videos.")
        if histograms[0] is not None:
             logger.info(f"Histograms vector dimension: {histograms[0].shape[0]}")

        logger.info(f"Start clustering {len(histograms)} videos (method '{args.method}') ...")

        clusteriser = Clusterize(
            file_names=file_names,
            histograms=scaled_histograms,
            logger=logger,
            eps_value=args.eps,
            min_pts_value=args.min_pts,
            clustering_type=args.method
        )

        clusteriser.generate_clusters()
        clusteriser.visualize_clusters()
    else:
        logger.warning("--- Extraction complete: No histograms were extracted. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hurricane detection via video clustering.")

    #Extractor
    parser.add_argument('--no_flow', action='store_true', help='Disable optical flow feature extraction.')

    #Clusterizer
    parser.add_argument('--method', type=str, default='DBSCAN', choices=['DBSCAN', 'DBCLASD'],
                        help='Clustering method to use.')
    parser.add_argument('--eps', type=float, default=25.0, help='Epsilon parameter for clustering. Needs tuning. Default: 25.0')
    parser.add_argument('--min_pts', type=int, default=2, help='Minimum samples parameter for clustering.')

    args = parser.parse_args()
    main(args)
