# Ferro Vision Streamlit App

The app has three pages selected with horizontal navigation buttons at the top:

- **Overview** — view a casting inspection dashboard with dataset counts, class distribution, representative validation images, the image-classification workflow, an AI casting inspection flow card, an engine-cycle versus RUL chart, and an interactive turbofan sensor trend.
- **RUL Prediction** — estimate remaining useful life with the trained ANN, using a dataset example, a one-row telemetry CSV, or manual entry of all 25 model features; view a post-prediction timeline of cycles used and sensor-based estimated cycles remaining.
- **Image Classification** — verify that an upload resembles the held-out casting validation images, then classify accepted images as Normal or Defective.

The Image Classification page has a staged image-selection/upload card, a responsive preview and inspection-controls layout, and status cards that show the predicted class and classifier confidence. The Ferro Vision brand header stays sticky while scrolling.

## Run locally

From the project root, install the dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects these project artifacts:

- `Data/Turbofan_Engine_Degradation_Dataset.csv`
- `Data/Image/train/{normal,defective}/` and `Data/Image/validation/{normal,defective}/`
- `models/ann_best.keras`
- `models/ann_preprocessing.joblib`
- `models/best_image_classifier.keras`

The telemetry CSV upload must contain exactly one row with `cycle`, `op_setting_1` through `op_setting_3`, and `sensor_1` through `sensor_21`. A template can be downloaded from the RUL Prediction page. The **Manual Entries** mode provides an input field for each of those 25 ANN features, with dataset medians as editable starting values.

The Overview casting image totals combine the training and validation folders. Defective images are shown as examples only; the bundled labels do not identify individual defect subtypes.

The Overview RUL trend displays recorded RUL over cycle for the first engine in the turbofan telemetry dataset. It uses a responsive Plotly dark-theme area chart with a gradient fill, smoothed neon line, current-cycle marker, and hover values for engine cycles and remaining useful life.

## Casting image validation

The CNN training data uses the prepared split: 2,300 Normal and 3,006 Defective training images; 575 Normal and 752 Defective validation images. The CNN notebook reads these folders directly and verifies the counts.

Before classification, the image page extracts a feature vector from the saved model's `GlobalAveragePooling2D` layer. For the current VGG16 winner, this is the pooled pretrained VGG16 backbone. Reference vectors are generated from the held-out images under `Data/Image/validation/normal/` and `Data/Image/validation/defective/`, L2-normalized, and stored in `models/casting_reference_embeddings.npz`.

At inference, the convolutional backbone runs once. Its feature map is used for similarity validation, then accepted images are passed through the saved classifier head, avoiding a second VGG16 backbone pass.
Feature extraction uses a fixed-size TensorFlow graph to reduce eager-execution overhead.

Generate or refresh the reference vectors manually after changing the validation images or image model:

```powershell
python generate_image_embeddings.py
```

The app also creates or refreshes this cache automatically the first time an image is validated. It reuses the saved vectors on subsequent validations and invalidates the cache when the classifier or validation images change.

For an upload, the app calculates cosine similarity against every reference vector and reports both maximum and average similarity. The configured gate uses **maximum similarity > 0.75**; a value at or below 0.75 is rejected with “Invalid Image. Please upload a valid casting component image.” Only images that pass this gate are sent to the existing Normal/Defective classifier. Average similarity is shown as an additional diagnostic and does not decide acceptance.

The 0.75 threshold is the requested starting policy, not a measured operating point: without non-casting validation images, false-accept performance cannot be estimated. For industrial deployment, validate and tune the threshold on a separate, representative set of casting and non-casting images before relying on the gate for production decisions.
