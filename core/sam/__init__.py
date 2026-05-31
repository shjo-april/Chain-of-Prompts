# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import numpy as np
import sanghyunjo as shjo

from typing import Union, List, Tuple

import hydra
from hydra import initialize_config_module
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_module("core.sam.configs", version_base="1.2")

class ImageSAM:
    """
    Unified wrapper for SAM (baseline/HQ), SAM2, and ZIM.

    - backend: 'sam1' | 'sam2' | 'zim'
    """
    def __init__(self, pt_path: str, backend: str = 'sam1', device: Union[torch.device, str] = 'cuda'):
        self.pt_path = pt_path
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # --- backend auto-detection
        self.backend = backend

        # --- initialize backend-specific predictor
        if self.backend == 'sam1':
            self._build_sam1()
        elif self.backend == 'sam2':
            self._build_sam2()
        elif self.backend == 'sam3':
            self._build_sam3()
        elif self.backend == 'zim':
            self._build_zim()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    # ---------------------------
    # Private Helpers
    # ---------------------------
    def _build_sam1(self):
        from .build_sam import sam_model_registry
        from .build_sam_baseline import sam_model_registry_baseline
        from .predictor import SamPredictor

        tag = shjo.basename(self.pt_path).replace('.pth', '').replace('sam_', '')
        if 'hq' in tag:
            tag = tag.replace('hq_', '')
            sam_fn = sam_model_registry
        else:
            sam_fn = sam_model_registry_baseline

        self._model = sam_fn[tag](self.pt_path).to(self.device)
        self._predictor = SamPredictor(self._model)

    def _build_sam2(self):
        from .build_sam import build_sam2
        from .predictor import SAM2ImagePredictor

        yaml_name = shjo.basename(self.pt_path, replace_ext='.yaml')
        self._model = build_sam2(yaml_name, self.pt_path, self.device)
        self._predictor = SAM2ImagePredictor(self._model)

    def _build_zim(self):
        from .predictor import ZimPredictor
        self._predictor = ZimPredictor(self.pt_path)

    def _build_sam3(self):
        from .predictor import Sam3Processor
        from .build_sam import build_sam3_image_model
        self._model = build_sam3_image_model(checkpoint_path=self.pt_path, enable_inst_interactivity=True)
        self._predictor = Sam3Processor(self._model, confidence_threshold=0.5)
        self._state = None

    def _prepare_input_dict(self, box, point_coords, point_labels, mask_input, text_prompt):
        """
        Normalize and pack user inputs into a dictionary for predictors.
        """
        d = {}
        if box and len(box) == 4:
            d['box'] = np.asarray([box], dtype=np.int32)

        if point_coords:
            d['point_coords'] = np.asarray(point_coords, dtype=np.int32)
            if not point_labels or len(point_labels) != len(point_coords):
                raise ValueError("point_labels must have the same length as point_coords.")
            d['point_labels'] = np.asarray(point_labels, dtype=np.int32)

        if mask_input is not None:
            arr = np.asarray(mask_input)
            if arr.ndim == 2:
                arr = arr[None, ...]
            d['mask_input'] = arr

        if text_prompt is not None and len(text_prompt) > 0:
            d['text_prompt'] = text_prompt

        if hasattr(self, '_state') and self._state is not None:
            d['state'] = self._state
        
        return d

    # ---------------------------
    # Public API
    # ---------------------------
    def _set_image(self, cv_image: np.ndarray):
        """Precompute and cache image embeddings for prediction."""
        if self.backend in ['sam1', 'zim']:
            self._predictor.set_image(cv_image.copy(), 'BGR')
        else:
            rgb_image = shjo.convert(cv_image, 'bgr2rgb')
            if self.backend == 'sam3':
                self._state = self._predictor.set_image(rgb_image.copy())
            else:
                self._predictor.set_image(rgb_image.copy())

    def encode_image(self, cv_image: np.ndarray) -> List[torch.Tensor]:
        """
        Encode image and return multi-scale feature maps.

        Returns list of raw backbone feature tensors:
        - sam1: [torch.Size([1, 256, 64, 64])] (plain ViT neck output only)
        - sam2: [torch.Size([1, 32, 256, 256]), torch.Size([1, 64, 128, 128]), torch.Size([1, 256, 64, 64])] (raw FPN)
        - sam3: [torch.Size([1, 256, H, W]), ...] (raw backbone FPN levels)
        """
        if self.backend in ['sam1', 'zim']:
            # SAM1: plain ViT (no hierarchical FPN), return neck output only [1, 256, 64, 64]
            self._predictor.set_image(cv_image.copy(), 'BGR')
            feature_maps = [self._predictor.features]
        elif self.backend == 'sam3':
            rgb_image = shjo.convert(cv_image, 'bgr2rgb')
            self._state = self._predictor.set_image(rgb_image.copy())
            feature_maps = self._state["backbone_out"]["backbone_fpn"]
            feature_maps = [f.float() for f in feature_maps]
        else:
            # sam2: get raw backbone FPN before conv_s0/conv_s1 projection
            rgb_image = shjo.convert(cv_image, 'bgr2rgb')
            input_image = self._predictor._transforms(rgb_image)
            input_image = input_image[None, ...].to(self._predictor.device)
            feature_maps = self._predictor.model.image_encoder(input_image)['backbone_fpn']

        return feature_maps
    
    def predict(
            self,
            box: List[int] = (),
            point_coords: List[Tuple[int, int]] = (),
            point_labels: List[int] = (),
            mask_input: np.ndarray = None,
            cv_image: np.ndarray = None,
            text_prompt: str = None,
            postprocessing: str = 'union' # None: no postprocessing, 'union': union all masks
        ):
        """
        Run mask prediction.
        """
        if cv_image is not None:
            self._set_image(cv_image)

        input_dict = self._prepare_input_dict(box, point_coords, point_labels, mask_input, text_prompt)
        masks, scores, _ = self._predictor.predict(multimask_output=False, **input_dict)

        if len(masks.shape) == 4:
            masks, scores = masks[0], scores[0] # unpack batch dimension

        # sort by scores
        order = np.argsort(scores)[::-1]
        masks = masks[order].astype(np.float32)
        scores = scores[order].astype(np.float32)

        if postprocessing == 'union':
            merge_fn = lambda data: np.max(data, axis=0, keepdims=True)
            masks, scores = map(merge_fn, [masks, scores])
        
        return masks, scores
    
class VideoSAM:
    def __init__(self, pt_path: str, backend: str = 'sam2', device: Union[torch.device, str] = 'cuda'):
        self.pt_path = pt_path
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # --- backend auto-detection
        self.backend = backend

        # --- initialize backend-specific predictor
        if self.backend == 'sam2':
            self._build_sam2()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ---------------------------
    # Private Helpers
    # ---------------------------
    def _build_sam2(self):
        from .build_sam import build_sam2_video_predictor
        from .predictor import SAM2VideoPredictor

        yaml_name = shjo.basename(self.pt_path, replace_ext='.yaml')
        self._predictor: SAM2VideoPredictor = build_sam2_video_predictor(yaml_name, self.pt_path, self.device)

    def _build_sam3(self):
        from .predictor import Sam3VideoPredictor
        from .build_sam import build_sam3_video_model
        self._model = build_sam3_video_model(checkpoint_path=self.pt_path)
        self._predictor = Sam3VideoPredictor(self._model)
        self.inference_state = None
    
    # ---------------------------
    # Public API
    # ---------------------------
    def extract_features_from_video(self, video_path: str):
        extra_args = {}
        if self.backend == 'sam3': extra_args['session_id'] = self.inference_state
        self.inference_state = self._predictor.init_state(video_path=video_path, **extra_args)

    def clear_prompt(self):
        self.num_object = 0

    def add_prompt(self, box=[], point_coords=[], point_labels=[], mask_input=None, frame_index=0):
        self.num_object += 1

        input_dict = {}
        
        if len(box) > 0: 
            input_dict['box'] = np.asarray([box], dtype=np.float32)
        
        if len(point_coords):
            input_dict['points'] = np.asarray(point_coords, dtype=np.int32)
            input_dict['labels'] = np.asarray(point_labels, dtype=np.int32)
        
        if mask_input is not None:
            self._predictor.add_new_mask(self.inference_state, frame_idx=frame_index, obj_id=self.num_object, mask=mask_input)
        
        if len(box) > 0 or len(point_coords) > 0:
            self._predictor.add_new_points_or_box(self.inference_state, frame_idx=frame_index, obj_id=self.num_object, **input_dict)

    def propagate(self) -> List[np.ndarray]:
        video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(self.inference_state):
            new_mask = np.zeros_like(out_mask_logits[0][0].cpu().numpy()).astype(np.uint8)
            for i, obj_id in enumerate(out_obj_ids):
                new_mask[out_mask_logits[i][0].cpu().numpy() > 0] = obj_id
            video_segments.append(new_mask)
        return video_segments
    
    def get_video_features(self):
        raise NotImplementedError("VideoSAM does not support get_video_features() yet.")

"""
* TODO: Streamline Superpixel and Edge Detectors
"""
class SuperpixelZIM:
    def __init__(self, pt_path, batch=64, details=True, device=torch.device('cuda:0')):
        from .automatic_mask_generator import ZimAutomaticMaskGenerator
        from .build_sam import build_zim_model
        zim = build_zim_model(pt_path)

        # refer to: https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
        params = {
            'points_per_batch': batch
        }
        if details:
            params = {
                'points_per_side': 96,
                'points_per_batch': 128,
                'pred_iou_thresh': 0.65,
                'stability_score_thresh': 0.88,
                'stability_score_offset': 1.0,
                'crop_n_layers': 2,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 30,
            }

        self.predictor = ZimAutomaticMaskGenerator(zim, **params)
    
    @torch.no_grad()
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        masks = self.predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        return masks
    
class SuperpixelSAM2:
    def __init__(self, pt_path, batch=64, details=True, device=torch.device('cuda:0')):
        from .build_sam import build_sam2
        from .automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam = build_sam2(shjo.basename(pt_path.replace('.pt', '.yaml')), pt_path, device, apply_postprocessing=False)
        
        # refer to: https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
        params = {
            'points_per_batch': batch
        }
        if details:
            params = {
                'points_per_side': 96,
                'points_per_batch': 128,
                'pred_iou_thresh': 0.65,
                'stability_score_thresh': 0.88,
                'stability_score_offset': 1.0,
                'crop_n_layers': 2,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 30,
                'use_m2m': True,
                'multimask_output': True,
            }

        self.predictor = SAM2AutomaticMaskGenerator(sam, **params)

    @torch.no_grad()
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        
        masks = self.predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
class SuperpixelSAM3:
    def __init__(self, pt_path, batch=64, details=True, device=torch.device('cuda:0')):
        from .build_sam import build_sam3_image_model
        from .automatic_mask_generator import SAM3AutomaticMaskGenerator
        sam = build_sam3_image_model(checkpoint_path=pt_path)
        
        params = {
            'points_per_batch': batch
        }
        if details:
            params = {
                'points_per_side': 96,
                'points_per_batch': 128,
                'pred_iou_thresh': 0.65,
                'stability_score_thresh': 0.88,
                'stability_score_offset': 1.0,
                'crop_n_layers': 2,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 30,
            }
        self.predictor = SAM3AutomaticMaskGenerator(sam, **params)
        
    @torch.no_grad()
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        
        masks = self.predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks

class EdgeSAM2:
    def __init__(self, pt_path, edge_path='I:/cache/edge.yml.gz', batch=64, details=False, device=torch.device('cuda:0')):
        from .build_sam import build_sam2
        from .automatic_mask_generator import SamEdgeGenerator
        sam = build_sam2(shjo.basename(pt_path.replace('.pt', '.yaml')), pt_path, device, apply_postprocessing=False)

        params = {
            'points_per_batch': batch
        }
        if details:
            params = {
                'points_per_side': 96,
                'points_per_batch': 128,
                'pred_iou_thresh': 0.65,
                'stability_score_thresh': 0.88,
                'stability_score_offset': 1.0,
                'crop_n_layers': 2,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 30,
                'use_m2m': True,
                'multimask_output': True,
            }

        self.edge_predictor = SamEdgeGenerator(
            sam, 
            nms_threshold=0.7, 
            pred_iou_thresh_filtering=True,
            stability_score_thresh_filtering=False,
            **params
        )

        import cv2
        # Note: If you encounter an error with cv2.ximgproc.createStructuredEdgeDetection,
        # - pip uninstall opencv-python opencv-contrib-python -y
        # - pip install opencv-contrib-python --upgrade
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_path)
    
    def predict(self, cv_image):
        cv_image = shjo.convert(cv_image, 'bgr2rgb')
        
        masks = self.edge_predictor.generate(cv_image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        p_max = masks[0]["prob"]
        for mask in masks[1:]:
            p_max = np.maximum(p_max, mask['prob'])
        
        edges = (p_max - p_max.min()) / (p_max.max() - p_max.min())

        # edges = cv2.Canny((p_max * 255).astype(np.uint8), 100, 200) # alternative edge detection
        edges = self.edge_detection.edgesNms(edges, self.edge_detection.computeOrientation(edges))

        return edges