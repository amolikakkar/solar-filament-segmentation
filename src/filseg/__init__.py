"""filseg: solar filament instance segmentation for the Solar Filament Segmentation Challenge 2026."""
from .classical import ClassicalParams, segment_classical
from .io import CocoSet, discover, image_id_from_path, list_images, read_image
from .metrics import score_image, summarize
from .postprocess import PostParams, to_instances
from .preprocess import Disk, find_disk, flatten, preprocess
from .rle import decode_rle, encode_rle

__version__ = "0.1.0"
