from __future__ import annotations

import cv2
import numpy as np
import os
import torch
import torch.nn.functional

from path import Path
from typing import List, Optional, Tuple

from dvmvs.config import Config
from dvmvs.dataset_loader import PreprocessImage
from dvmvs.fusionnet.model import CostVolumeDecoder, CostVolumeEncoder, FeatureExtractor, FeatureShrinker, LSTMFusion
from dvmvs.keyframe_buffer import KeyframeBuffer
from dvmvs.utils import cost_volume_fusion
from dvmvs.utils import get_non_differentiable_rectangle_depth_estimation
from dvmvs.utils import get_warp_grid_for_cost_volume_calculation
from dvmvs.utils import visualize_predictions

from smg.utility import DepthImageProcessor, GeometryUtil, ImageUtil, MonocularDepthEstimator


class DVMVSMonocularDepthEstimator(MonocularDepthEstimator):
    """
    A wrapper around the DeepVideoMVS monocular depth estimator.

    .. note::
        This is an encapsulated version of fusionnet\run_testing_online.py from the DeepVideoMVS code,
        to allow me to use DeepVideoMVS for live depth estimation.
    """

    # CONSTRUCTOR

    def __init__(self, dvmvs_root: str = "C:/deep-video-mvs", *, border_to_fill: int = 40,
                 debug: bool = False, max_depth: float = 3.0):
        """
        Construct a DeepVideoMVS-based monocular depth estimator.

        :param dvmvs_root:      The root folder of the DeepVideoMVS repository.
        :param border_to_fill:  The size of the border (in pixels) of the estimated depth image
                                that is to be filled with zeros to help mitigate depth noise.
        :param debug:           Whether or not to enable debugging.
        :param max_depth:       The maximum depth values to keep during post-processing (pixels with depth values
                                greater than this will have their depths set to zero).
        """
        self.__border_to_fill: int = border_to_fill
        self.__debug: bool = debug
        self.__max_postprocessing_depth: float = max_depth

        self.__device: torch.device = torch.device("cuda")

        self.__feature_extractor: FeatureExtractor = FeatureExtractor()
        self.__feature_shrinker: FeatureShrinker = FeatureShrinker()
        self.__cost_volume_encoder: CostVolumeEncoder = CostVolumeEncoder()
        self.__lstm_fusion: LSTMFusion = LSTMFusion()
        self.__cost_volume_decoder: CostVolumeDecoder = CostVolumeDecoder()

        self.__feature_extractor.to(self.__device)
        self.__feature_shrinker.to(self.__device)
        self.__cost_volume_encoder.to(self.__device)
        self.__lstm_fusion.to(self.__device)
        self.__cost_volume_decoder.to(self.__device)

        self.__model: List[torch.nn.Module] = [
            self.__feature_extractor,
            self.__feature_shrinker,
            self.__cost_volume_encoder,
            self.__lstm_fusion,
            self.__cost_volume_decoder
        ]

        for i in range(len(self.__model)):
            try:
                checkpoint = sorted(Path(os.path.join(dvmvs_root, "dvmvs/fusionnet/weights")).files())[i]
                weights = torch.load(checkpoint)
                self.__model[i].load_state_dict(weights)
                self.__model[i].eval()
                print("Loaded weights for", checkpoint)
            except Exception as e:
                print(e)
                print("Could not find the checkpoint for module", i)
                exit(1)

        self.__feature_extractor = self.__model[0]
        self.__feature_shrinker = self.__model[1]
        self.__cost_volume_encoder = self.__model[2]
        self.__lstm_fusion = self.__model[3]
        self.__cost_volume_decoder = self.__model[4]

        self.__warp_grid: torch.Tensor = get_warp_grid_for_cost_volume_calculation(
            width=int(Config.test_image_width / 2),
            height=int(Config.test_image_height / 2),
            device=self.__device
        )

        self.__scale_rgb: float = 255.0
        self.__mean_rgb: List[float] = [0.485, 0.456, 0.406]
        self.__std_rgb: List[float] = [0.229, 0.224, 0.225]

        self.__min_depth: float = 0.25
        self.__max_depth: float = 20.0
        self.__n_depth_levels: int = 64

        self.__keyframe_buffer: KeyframeBuffer = KeyframeBuffer(
            buffer_size=Config.test_keyframe_buffer_size,
            keyframe_pose_distance=Config.test_keyframe_pose_distance,
            optimal_t_score=Config.test_optimal_t_measure,
            optimal_R_score=Config.test_optimal_R_measure,
            store_return_indices=False
        )

        self.__K: Optional[np.ndarray] = None

        self.__lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.__previous_depth: Optional[torch.Tensor] = None
        self.__previous_pose: Optional[torch.Tensor] = None

        # Additional variables needed to perform my own temporal filtering of the depth images. Note that
        # although some of my variables have similar names to the ones used by DeepVideoMVS itself, they're
        # used for a different purpose, and it's important not to get the two confused.
        self.__current_depth_image: Optional[np.ndarray] = None
        self.__current_w_t_c: Optional[np.ndarray] = None
        self.__current_world_points: Optional[np.ndarray] = None
        self.__previous_depth_image: Optional[np.ndarray] = None
        self.__previous_w_t_c: Optional[np.ndarray] = None
        self.__previous_world_points: Optional[np.ndarray] = None

    # PUBLIC METHODS

    # noinspection PyPep8Naming
    def estimate_depth(self, colour_image: np.ndarray, tracker_w_t_c: np.ndarray, *, postprocess: bool = False) \
            -> Optional[np.ndarray]:
        """
        Try to estimate a depth image corresponding to the colour image passed in.

        .. note::
            This will return None if a suitable depth image cannot be estimated for the colour image passed in.
            For more precise details, see KeyframeBuffer.try_new_keyframe in the DeepVideoMVS code.

        :param colour_image:    The colour image.
        :param tracker_w_t_c:   The camera pose corresponding to the colour image (as a camera -> world transform).
        :param postprocess:     Whether or not to apply any optional post-processing to the depth image.
        :return:                The estimated depth image, if possible, or None otherwise.
        """
        # Note: This code is essentially borrowed from the DeepVideoMVS code (with minor tweaks to make it work here).
        #       As such, I haven't done much additional tidying/commenting, as there's not a lot of point.
        with torch.no_grad():
            reference_pose = tracker_w_t_c.copy()
            reference_image = colour_image.copy().astype(np.float32)
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

            # POLL THE KEYFRAME BUFFER
            response = self.__keyframe_buffer.try_new_keyframe(reference_pose, reference_image)
            if response == 0 or response == 2 or response == 4 or response == 5:
                return None
            elif response == 3:
                self.__previous_depth = None
                self.__previous_pose = None
                self.__lstm_state = None
                return None

            preprocessor = PreprocessImage(
                K=self.__K,
                old_width=reference_image.shape[1],
                old_height=reference_image.shape[0],
                new_width=Config.test_image_width,
                new_height=Config.test_image_height,
                distortion_crop=Config.test_distortion_crop,
                perform_crop=Config.test_perform_crop
            )

            reference_image = preprocessor.apply_rgb(
                image=reference_image,
                scale_rgb=self.__scale_rgb,
                mean_rgb=self.__mean_rgb,
                std_rgb=self.__std_rgb
            )

            reference_image_torch = torch.from_numpy(
                np.transpose(reference_image, (2, 0, 1))
            ).float().to(self.__device).unsqueeze(0)
            reference_pose_torch = torch.from_numpy(reference_pose).float().to(self.__device).unsqueeze(0)

            full_K_torch = torch.from_numpy(
                preprocessor.get_updated_intrinsics()
            ).float().to(self.__device).unsqueeze(0)

            half_K_torch = full_K_torch.clone().cuda()
            half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

            lstm_K_bottom = full_K_torch.clone().cuda()
            lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

            measurement_poses_torch = []
            measurement_images_torch = []
            measurement_frames = self.__keyframe_buffer.get_best_measurement_frames(Config.test_n_measurement_frames)
            for (measurement_pose, measurement_image) in measurement_frames:
                measurement_image = preprocessor.apply_rgb(
                    image=measurement_image,
                    scale_rgb=self.__scale_rgb,
                    mean_rgb=self.__mean_rgb,
                    std_rgb=self.__std_rgb
                )
                measurement_image_torch = torch.from_numpy(
                    np.transpose(measurement_image, (2, 0, 1))
                ).float().to(self.__device).unsqueeze(0)
                measurement_pose_torch = torch.from_numpy(measurement_pose).float().to(self.__device).unsqueeze(0)
                measurement_images_torch.append(measurement_image_torch)
                measurement_poses_torch.append(measurement_pose_torch)

            measurement_feature_halfs = []
            for measurement_image_torch in measurement_images_torch:
                measurement_feature_half, _, _, _ = self.__feature_shrinker(
                    *self.__feature_extractor(measurement_image_torch)
                )
                measurement_feature_halfs.append(measurement_feature_half)

            reference_feature_half, reference_feature_quarter, \
                reference_feature_one_eight, reference_feature_one_sixteen = self.__feature_shrinker(
                    *self.__feature_extractor(reference_image_torch)
                )

            cost_volume = cost_volume_fusion(
                image1=reference_feature_half,
                image2s=measurement_feature_halfs,
                pose1=reference_pose_torch,
                pose2s=measurement_poses_torch,
                K=half_K_torch,
                warp_grid=self.__warp_grid,
                min_depth=self.__min_depth,
                max_depth=self.__max_depth,
                n_depth_levels=self.__n_depth_levels,
                device=self.__device,
                dot_product=True
            )

            skip0, skip1, skip2, skip3, bottom = self.__cost_volume_encoder(
                features_half=reference_feature_half,
                features_quarter=reference_feature_quarter,
                features_one_eight=reference_feature_one_eight,
                features_one_sixteen=reference_feature_one_sixteen,
                cost_volume=cost_volume
            )

            if self.__previous_depth is not None:
                depth_estimation = get_non_differentiable_rectangle_depth_estimation(
                    reference_pose_torch=reference_pose_torch,
                    measurement_pose_torch=self.__previous_pose,
                    previous_depth_torch=self.__previous_depth,
                    full_K_torch=full_K_torch,
                    half_K_torch=half_K_torch,
                    original_height=Config.test_image_height,
                    original_width=Config.test_image_width
                )
                depth_estimation = torch.nn.functional.interpolate(
                    input=depth_estimation,
                    scale_factor=(1.0 / 16.0),
                    mode="nearest"
                )
            else:
                depth_estimation = torch.zeros(
                    size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))
                ).to(self.__device)

            self.__lstm_state = self.__lstm_fusion(
                current_encoding=bottom,
                current_state=self.__lstm_state,
                previous_pose=self.__previous_pose,
                current_pose=reference_pose_torch,
                estimated_current_depth=depth_estimation,
                camera_matrix=lstm_K_bottom
            )

            prediction, _, _, _, _ = self.__cost_volume_decoder(
                reference_image_torch, skip0, skip1, skip2, skip3, self.__lstm_state[0]
            )
            self.__previous_depth = prediction.view(1, 1, Config.test_image_height, Config.test_image_width)
            self.__previous_pose = reference_pose_torch

            prediction = prediction.cpu().numpy().squeeze()

            # If debugging is enabled, visualise the reference image, measurement image and predicted depth image.
            if self.__debug:
                # noinspection PyUnboundLocalVariable
                visualize_predictions(
                    numpy_reference_image=reference_image,
                    numpy_measurement_image=measurement_image,
                    numpy_predicted_depth=prediction,
                    normalization_mean=self.__mean_rgb,
                    normalization_std=self.__std_rgb,
                    normalization_scale=self.__scale_rgb,
                    depth_multiplier_for_visualization=5000
                )

            # Make a resized version of the predicted depth image that is the same size as the original input image.
            height, width = colour_image.shape[:2]
            depth_image: np.ndarray = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_NEAREST)

            # Fill the border of the depth image with zeros (depths around the image border are often quite noisy).
            depth_image = ImageUtil.fill_border(depth_image, self.__border_to_fill, 0.0)

            # Return the estimated depth image, post-processing it in the process if requested.
            return self.__postprocess_depth_image(depth_image, tracker_w_t_c) if postprocess else depth_image

    def set_intrinsics(self, intrinsics: np.ndarray) -> MonocularDepthEstimator:
        """
        Set the camera intrinsics.

        :param intrinsics:  The 3x3 camera intrinsics matrix.
        :return:            The current object.
        """
        self.__K = intrinsics
        return self

    # PRIVATE METHODS

    def __postprocess_depth_image(self, depth_image: np.ndarray, tracker_w_t_c: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to post-process the specified depth image to reduce the amount of noise it contains.

        .. note::
            This function will return None if the input depth image does not have depth values for enough pixels.

        :param depth_image:     The input depth image.
        :param tracker_w_t_c:   The camera pose corresponding to the depth image (as a camera -> world transform).
        :return:                The post-processed depth image, if possible, or None otherwise.
        """
        # Update the variables needed to perform temporal filtering.
        self.__previous_depth_image = self.__current_depth_image
        self.__previous_w_t_c = self.__current_w_t_c
        self.__previous_world_points = self.__current_world_points

        self.__current_depth_image = depth_image.copy()
        self.__current_w_t_c = tracker_w_t_c.copy()

        intrinsics: Tuple[float, float, float, float] = GeometryUtil.intrinsics_to_tuple(self.__K)
        self.__current_world_points = GeometryUtil.compute_world_points_image_fast(
            depth_image, tracker_w_t_c, intrinsics
        )

        # If a previous depth image is available with which to perform temporal filtering:
        if self.__previous_depth_image is not None:
            # Remove any pixel in the current depth image that either does not have a corresponding pixel
            # in the previous world-space points image at all, or else does not have one whose world-space
            # point is sufficiently close to the current world-space point.
            postprocessed_depth_image = DepthImageProcessor.remove_temporal_inconsistencies(
                self.__current_depth_image, self.__current_w_t_c, self.__current_world_points,
                self.__previous_depth_image, self.__previous_w_t_c, self.__previous_world_points,
                intrinsics, debug=False, distance_threshold=0.1
            )

            # Further post-process the resulting depth image using the normal approach, and then return it.
            return DepthImageProcessor.postprocess_depth_image(
                postprocessed_depth_image, max_depth=self.__max_postprocessing_depth, max_depth_difference=0.05,
                median_filter_radius=5, min_region_size=5000, min_valid_fraction=0.0
            )

        # Otherwise, skip this depth image and wait till next time.
        else:
            return None
