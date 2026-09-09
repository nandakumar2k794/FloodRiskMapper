"""
DrainWatch AI Computer Vision package for Chennai Flood Access Risk Mapper.

Supported input types: image, MP4 video, USB webcam, authorized RTSP/HTTP streams.

This system does NOT assume access to Chennai municipal/police CCTV.
All stream-based inference requires authorization from the stream owner.
"""
from cv.models import (
    DrainWatchClass,
    DetectionBox,
    FrameObservation,
    AggregatedObservation,
    BLOCKAGE_CLASSES,
    DRAIN_STRUCTURE_CLASSES,
    WATER_CLASSES,
)
from cv.preprocessing import (
    VideoFrameExtractor,
    ImagePreprocessor,
    ROISelector,
)
from cv.detector import (
    DrainDetector,
    COCO_PROXY_CLASS_MAP,
)
from cv.aggregator import (
    TemporalObservationAggregator,
)
from cv.drainwatch import (
    DrainWatchService,
)

__all__ = [
    # Data models
    "DrainWatchClass",
    "DetectionBox",
    "FrameObservation",
    "AggregatedObservation",
    # Class groupings
    "BLOCKAGE_CLASSES",
    "DRAIN_STRUCTURE_CLASSES",
    "WATER_CLASSES",
    # Preprocessing
    "VideoFrameExtractor",
    "ImagePreprocessor",
    "ROISelector",
    # Detection
    "DrainDetector",
    "COCO_PROXY_CLASS_MAP",
    # Aggregation
    "TemporalObservationAggregator",
    # Service coordinator
    "DrainWatchService",
]
