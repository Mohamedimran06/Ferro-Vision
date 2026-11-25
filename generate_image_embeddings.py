from pathlib import Path

import tensorflow as tf

from image_validation import (
    CLASSIFIER_MODEL_PATH,
    create_dataset_fingerprint,
    create_feature_extractor,
    find_reference_images,
    load_or_create_reference_embeddings,
)


def main() -> None:
    if not CLASSIFIER_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained image model not found: {CLASSIFIER_MODEL_PATH}")

    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    image_paths = find_reference_images()
    fingerprint = create_dataset_fingerprint(image_paths, classifier.name)
    print(f"Extracting embeddings for {len(image_paths):,} validation images...")

    def report_progress(processed: int, total: int) -> None:
        print(f"\rProcessed {processed:,}/{total:,} images", end="", flush=True)

    embeddings = load_or_create_reference_embeddings(
        create_feature_extractor(classifier),
        classifier.name,
        image_paths,
        fingerprint,
        progress_callback=report_progress,
    )
    print(
        f"\nSaved {embeddings.shape[0]:,} normalized reference vectors "
        f"({embeddings.shape[1]} features each)."
    )
    print(
        "Reference file: "
        f"{Path(__file__).resolve().parent / 'models' / 'casting_reference_embeddings.npz'}"
    )


if __name__ == "__main__":
    main()
