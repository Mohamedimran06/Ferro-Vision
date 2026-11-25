from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE_DIR = BASE_DIR / "Data" / "Image" / "validation"
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "best_image_classifier.keras"
REFERENCE_EMBEDDINGS_PATH = BASE_DIR / "models" / "casting_reference_embeddings.npz"
IMAGE_SIZE = (128, 128)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
ACCEPTANCE_THRESHOLD = 0.75
EMBEDDING_BATCH_SIZE = 32


def find_reference_images() -> list[Path]:
    if not REFERENCE_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Casting validation image folder not found: {REFERENCE_IMAGE_DIR}"
        )

    class_directories = [
        path for path in REFERENCE_IMAGE_DIR.iterdir() if path.is_dir()
    ]
    class_names = {path.name.strip().casefold() for path in class_directories}
    if not {"normal", "defective"}.issubset(class_names):
        raise FileNotFoundError(
            "Expected Normal and Defective subfolders in "
            f"{REFERENCE_IMAGE_DIR}."
        )

    image_paths = sorted(
        path
        for class_directory in class_directories
        for path in class_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No casting validation images found in {REFERENCE_IMAGE_DIR}.")
    return image_paths


def create_feature_extractor(
    classifier: tf.keras.Model,
) -> Callable[[tf.Tensor], tf.Tensor]:
    pooling_layers = [
        layer
        for layer in classifier.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    ]
    if not pooling_layers:
        raise ValueError(
            "The saved image classifier has no GlobalAveragePooling2D feature layer."
        )
    feature_extractor = tf.keras.Model(
        inputs=classifier.inputs,
        outputs=pooling_layers[-1].input,
        name=f"{classifier.name}_feature_extractor",
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(None, *IMAGE_SIZE, 3),
                dtype=tf.float32,
            )
        ],
        reduce_retracing=True,
    )
    def run_feature_extractor(model_input: tf.Tensor) -> tf.Tensor:
        return feature_extractor(model_input, training=False)

    return run_feature_extractor


def create_classifier_head(classifier: tf.keras.Model) -> tf.keras.Model:
    pooling_layers = [
        layer
        for layer in classifier.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    ]
    if not pooling_layers:
        raise ValueError(
            "The saved image classifier has no GlobalAveragePooling2D feature layer."
        )
    pooling_layer = pooling_layers[-1]
    return tf.keras.Model(
        inputs=pooling_layer.output,
        outputs=classifier.outputs[0],
        name=f"{classifier.name}_classifier_head",
    )


def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    if model_name == "VGG16":
        image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    elif model_name == "MobileNetV2":
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    elif model_name != "CNN":
        raise ValueError(f"Unsupported image classifier: {model_name}")
    return image_array[None, ...]


def extract_feature_map(
    feature_extractor: Callable[[tf.Tensor], tf.Tensor],
    image: Image.Image,
    model_name: str,
) -> np.ndarray:
    model_input = preprocess_image(image, model_name)
    feature_map = feature_extractor(model_input).numpy()[0]
    if feature_map.ndim != 3 or not np.isfinite(feature_map).all():
        raise ValueError("The image produced an invalid feature map.")
    return feature_map


def normalize_embedding(feature_map: np.ndarray) -> np.ndarray:
    if feature_map.ndim != 3 or not np.isfinite(feature_map).all():
        raise ValueError("The image produced an invalid feature map.")
    embedding = np.mean(feature_map, axis=(0, 1))
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(embedding).all() or norm == 0:
        raise ValueError("The image produced an invalid feature embedding.")
    return (embedding / norm).astype(np.float32)


def calculate_similarity(
    query_embedding: np.ndarray, reference_embeddings: np.ndarray
) -> tuple[float, float]:
    if reference_embeddings.ndim != 2 or reference_embeddings.shape[0] == 0:
        raise ValueError("Casting reference embeddings are empty or malformed.")
    if query_embedding.ndim != 1 or query_embedding.shape[0] != reference_embeddings.shape[1]:
        raise ValueError("Query and reference embeddings have incompatible shapes.")

    reference_norms = np.linalg.norm(reference_embeddings, axis=1)
    if not np.isfinite(reference_embeddings).all() or np.any(reference_norms == 0):
        raise ValueError("Casting reference embeddings contain invalid vectors.")
    normalized_references = reference_embeddings / reference_norms[:, None]
    query_norm = float(np.linalg.norm(query_embedding))
    if not np.isfinite(query_embedding).all() or query_norm == 0:
        raise ValueError("The query image produced an invalid feature embedding.")

    similarities = normalized_references @ (query_embedding / query_norm)
    return float(np.max(similarities)), float(np.mean(similarities))


def create_dataset_fingerprint(
    image_paths: list[Path],
    model_name: str,
    model_path: Path = CLASSIFIER_MODEL_PATH,
) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    model_stat = model_path.stat()
    digest.update(f"{model_stat.st_size}:{model_stat.st_mtime_ns}".encode("ascii"))
    for image_path in image_paths:
        stat = image_path.stat()
        relative_path = image_path.relative_to(REFERENCE_IMAGE_DIR)
        digest.update(
            f"{relative_path.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def load_or_create_reference_embeddings(
    feature_extractor: Callable[[tf.Tensor], tf.Tensor],
    model_name: str,
    image_paths: list[Path],
    fingerprint: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    cache_path = REFERENCE_EMBEDDINGS_PATH
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_model = str(cached["model_name"].item())
                cached_fingerprint = str(cached["fingerprint"].item())
                embeddings = cached["embeddings"]
            if (
                cached_model == model_name
                and cached_fingerprint == fingerprint
                and embeddings.ndim == 2
                and embeddings.shape[0] == len(image_paths)
                and embeddings.shape[1] > 0
                and np.isfinite(embeddings).all()
            ):
                return embeddings.astype(np.float32, copy=False)
        except (OSError, ValueError, KeyError):
            pass

    batch_embeddings = []
    total = len(image_paths)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_paths = image_paths[start : start + EMBEDDING_BATCH_SIZE]
        batch_images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                batch_images.append(preprocess_image(image, model_name)[0])
        batch_input = np.stack(batch_images, axis=0)
        feature_maps = feature_extractor(batch_input).numpy()
        embeddings = np.mean(feature_maps, axis=(1, 2))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms == 0):
            raise ValueError("A reference image produced an invalid feature embedding.")
        batch_embeddings.append((embeddings / norms).astype(np.float32))
        if progress_callback is not None:
            progress_callback(min(start + len(batch_paths), total), total)

    reference_embeddings = np.concatenate(batch_embeddings, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_path.parent, suffix=".npz", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(
                temporary_file,
                embeddings=reference_embeddings,
                model_name=np.asarray(model_name),
                fingerprint=np.asarray(fingerprint),
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return reference_embeddings
