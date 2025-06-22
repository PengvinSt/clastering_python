import cv2
import numpy as np
import os
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.feature import local_binary_pattern
import logging
from logging import Logger
from logger import ColoredFormatter

def _worker_init(log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Extractor:
    #Histohram
    DEFAULT_HIST_BINS = 32

    #Optical flow
    DEFAULT_OPTICAL_FLOW_BINS = 32
    DEFAULT_USE_OPTICAL_FLOW = False

    #LBP (Local Binary Patterns)
    DEFAULT_LBP_POINTS = 24
    DEFAULT_LBP_RADIUS = 3
    #Video
    DEFAULT_SAMPLE_RATE = 1.0
    DEFAULT_SUPPORTED_FORMATS = ['.mp4']

    #Threads
    DEFAULT_MAX_WORKERS = 16

    def __init__(self,
                 input_folder: str,
                 logger: Logger,
                 hist_bins: int = DEFAULT_HIST_BINS,
                 lbp_points: int = DEFAULT_LBP_POINTS,
                 lbp_radius: int = DEFAULT_LBP_RADIUS,
                 sample_rate: float = DEFAULT_SAMPLE_RATE,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 optical_flow_bins: int = DEFAULT_OPTICAL_FLOW_BINS,
                 use_optical_flow: bool = DEFAULT_USE_OPTICAL_FLOW) -> None:

        self.logger = logger
        self.input_folder = input_folder
        self.hist_bins = hist_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.sample_rate = sample_rate
        self.max_workers = max_workers
        self.optical_flow_bins = optical_flow_bins
        self.use_optical_flow = use_optical_flow

    def __get_video_files(self) -> List[str]:
        video_files: List[str] = []
        if not os.path.exists(self.input_folder):
            self.logger.error(f"Input folder '{self.input_folder}' does not exist")
            return video_files
        for file in os.listdir(self.input_folder):
            if os.path.splitext(file)[1].lower() in self.DEFAULT_SUPPORTED_FORMATS:
                video_files.append(os.path.join(self.input_folder, file))
        return video_files

    # HSV histogram
    def _get_color_histograms(self, frame: np.ndarray) -> np.ndarray:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Get histogram for each HSV channel
        h_hist = cv2.calcHist([hsv_frame], [0], None, [self.hist_bins], [0, 180])
        s_hist = cv2.calcHist([hsv_frame], [1], None, [self.hist_bins], [0, 256])
        v_hist = cv2.calcHist([hsv_frame], [2], None, [self.hist_bins], [0, 256])
        # Normalize HSV data
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        return np.concatenate((h_hist, s_hist, v_hist)).flatten()

    # LBP texture histogram
    def _get_texture_histograms(self, frame: np.ndarray) -> np.ndarray:
        # Process LBP
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_frame, self.lbp_points, self.lbp_radius, method='uniform')
        # Process LBP histogram
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        # Normalize LBP data
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # for smthng/0 cases
        return lbp_hist.flatten()

    # Optical histogram
    def _get_motion_histograms(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> Optional[np.ndarray]:
        if prev_gray is None:
            return None
        # Calc Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Calc speed and angle of motion
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_hist, _ = np.histogram(magnitude.ravel(), bins=self.optical_flow_bins, range=[0, 15])
        ang_hist, _ = np.histogram(angle.ravel(), bins=self.optical_flow_bins, range=[0, 2 * np.pi])

        mag_hist = mag_hist.astype("float")
        ang_hist = ang_hist.astype("float")

        # Normalize Optical data
        cv2.normalize(mag_hist, mag_hist)
        cv2.normalize(ang_hist, ang_hist)
        return np.concatenate((mag_hist, ang_hist)).flatten()

    def _process_single_video(
            self,
            video_path: str,
    ) -> Optional[Tuple[np.ndarray, str]]:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        self.logger.info(f"Start parsing: {video_name}...")

        try:
            video_caption = cv2.VideoCapture(video_path)

            if not video_caption.isOpened():
                self.logger.warning(f"Could not open video file {video_path}")
                return None

            fps = video_caption.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_caption.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0 or total_frames == 0:
                self.logger.warning(f"Video file {video_path} has 0 FPS or 0 frames.")
                return None

            if self.sample_rate <= 0:
                frame_skip = 1
            else:
                frame_skip = max(1, int(fps / self.sample_rate))

            histogram_per_frame = []
            prev_gray_frame = None

            for i in range(total_frames):
                # Skip frames according to sample_rate
                ret, frame = video_caption.read()
                if not ret:
                    break

                if i % frame_skip != 0:
                    continue

                color_histogram = self._get_color_histograms(frame)
                texture_histogram = self._get_texture_histograms(frame)

                if not self.use_optical_flow:
                    histogram_per_frame.append(np.concatenate([color_histogram, texture_histogram]))
                    continue

                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_histogram = self._get_motion_histograms(prev_gray_frame, current_gray)
                prev_gray_frame = current_gray
                if motion_histogram is not None:
                    full_histograms = np.concatenate([color_histogram, texture_histogram, motion_histogram])
                    histogram_per_frame.append(full_histograms)

            if not histogram_per_frame:
                self.logger.warning(f"No histograms were extracted from {video_name}")
                return None

            # Get final data
            final_histogram_vector = np.mean(histogram_per_frame, axis=0)
            self.logger.info(f"Processed {video_name}: extracted vector of size {final_histogram_vector.shape[0]}")
            return final_histogram_vector, video_name

        except Exception as e:
            self.logger.error(f"An exception occurred for {video_path}: {e}", exc_info=True)
            return None
        finally:
            if 'video_caption' in locals() and video_caption.isOpened():
                video_caption.release()

    def extract_histograms(self) -> Tuple[List[np.ndarray], List[str]]:
        video_files = self.__get_video_files()
        histograms_data: List[np.ndarray] = []
        video_names_data: List[str] = []

        if not video_files:
            self.logger.warning("No video files found in input folder")
            return [], []

        log_level = self.logger.level

        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_worker_init, initargs=(log_level,)) as executor:
            future_to_video = {
                executor.submit(
                    self._process_single_video,
                    video_path,
                ): video_path
                for video_path in video_files
            }

            self.logger.info(f"Submitted {len(video_files)} videos for processing to {self.max_workers} workers...")

            for future in as_completed(future_to_video):
                try:
                    result = future.result()
                    if result:
                        histogram_vector, video_name = result
                        histograms_data.append(histogram_vector)
                        video_names_data.append(video_name)
                except Exception as exc:
                    video_path = future_to_video[future]
                    self.logger.error(f"Future for video {video_path} generated an exception: {exc}")

            if not histograms_data:
                self.logger.warning("No valid histograms were extracted from any video.")

            return histograms_data, video_names_data