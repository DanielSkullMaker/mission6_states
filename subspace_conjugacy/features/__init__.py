"""Feature extraction package for image, audio and video vectorization."""

from subspace_conjugacy.features.vectorization import (
    vectorize_image,
    vectorize_batch,
    vectorize_batch_multi,
    load_and_vectorize,
    load_and_vectorize_batch,
    load_and_vectorize_batch_multi,
)
from subspace_conjugacy.features.extraction import (
    extract_class_vectors,
    extract_all_classes,
    extract_training_data,
)
from subspace_conjugacy.features.audio import (
    load_waveform,
    compute_spectrogram,
    compute_mfcc,
    mel_filterbank,
    pad_or_truncate_frames,
    vectorize_audio,
    load_and_vectorize_audio,
    load_and_vectorize_audio_batch,
)
from subspace_conjugacy.features.video import (
    extract_frames,
    extract_frames_from_paths,
    vectorize_frames_concat,
    vectorize_frames_keyframe,
    vectorize_video,
    load_and_vectorize_video,
    load_and_vectorize_video_batch,
)

__all__ = [
    # Vectorization (images)
    "vectorize_image",
    "vectorize_batch",
    "vectorize_batch_multi",
    "load_and_vectorize",
    "load_and_vectorize_batch",
    "load_and_vectorize_batch_multi",
    # Extraction
    "extract_class_vectors",
    "extract_all_classes",
    "extract_training_data",
    # Audio (статья 4)
    "load_waveform",
    "compute_spectrogram",
    "compute_mfcc",
    "mel_filterbank",
    "pad_or_truncate_frames",
    "vectorize_audio",
    "load_and_vectorize_audio",
    "load_and_vectorize_audio_batch",
    # Video (статья 4)
    "extract_frames",
    "extract_frames_from_paths",
    "vectorize_frames_concat",
    "vectorize_frames_keyframe",
    "vectorize_video",
    "load_and_vectorize_video",
    "load_and_vectorize_video_batch",
]
