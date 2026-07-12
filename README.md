# 25CS173 Vet Imaging Data Analysis - Lymphovascular Invasion Detection for Vet Imaging

This repository contains a pathology workflow for detecting lymphovascular invasion (LVI) in NDPI whole-slide images using the CONCH vision-language model.

The main application is a Flask web app in `LVI_detection/app.py`. It accepts a single `.ndpi` slide, runs a two-stage pipeline, and returns heatmaps and CSV/JSON outputs for review.

## What the backend pipeline does

1. Tile the slide and run vascular detection model. (Vascular detection model)
2. Classify vascular candidates with CONCH to judge whther it is LVI slides. (LVI judgement model)
3. Classify vascular candidates with transfer learning model estimate LVI likelihood and generate heatmaps. (LVI detection model)

## Directory structure

```text
.
|-- README.md
|-- requirements.txt  # the library need to be installed
|-- CONCH/  # CONCH git
`-- LVI_detection/  # flask app, detection pipeline, templates, and model scripts
	|-- app.py  # run for detection
	|-- LVI_detection_model/  # LVI judgement and detection model
	|   |-- judgment.py  # execute LVI judgement model
	|   `-- transfer/  # train and execute LVI detectioon model
	|       |-- 3_create_lvi_dataset.py
	|       |-- 4_train_lvi_model.py
	|       `-- infer_lvi.py
	|-- vascular_detection_model/  # train and execute vascular detection model
	|   |-- 1_tile_slides.py
	|   |-- 2_extract_conch_features.py
	|   |-- 3_make_dataset_from_points.py
	|   |-- 4_train_vascular_model.py
	|   |-- main.py  # execute vascular detection model
	|   `-- train_data/  # the data for training the models
	|-- templates/  # webUI templates
	|   |-- preview.html  # show the final result
	|   |-- review.html  # processing page to wait for the result
	|   `-- upload.html
	|-- uploads/  # store uploaded NDPI files
	`-- results/  # store all results
```

The `train_data/` folder is expected to contain local NDPI slides, annotations, tiled coordinates, extracted features, and saved model checkpoints. (Annotation file and trained NDPI slides do not show in github as the file size is large which cannot upload to giithub)

## Requirements

- Python 3.10 is recommended.
- PyTorch, OpenSlide, Flask, NumPy, Pillow, Matplotlib, and the other packages listed in `LVI_detection/requirements.txt`.
- Access to the CONCH token.

## Setup

Create and activate a Python environment, then install the CONCH package and the web app dependencies.

```bash
conda create -n conch python=3.10 -y
conda activate conch
pip install --upgrade pip
pip install -e ./CONCH
pip install -r LVI_detection/templates/requirements.txt
```

If you plan to run the app from a different environment manager, make sure the same dependencies are available and that `conch` can be imported from the local `CONCH/` package.

## Model files (setup)

Make sure the following model file exists before running inference:

- `LVI_detection/vascular_detection_model/train_data/models/vessel_classifier_conch.pth`
- `LVI_detection/vascular_detection_model/train_data/models/lvi_classifier_conch.pth`

The pipeline also loads CONCH models. To use CONCH model/ run the program, make sure the token need to be created through hugging face. You need to generate the token and replace to the `token.txt` through `hf_auth_token="token.txt"` by yourself, as the github security policy, token cannot be push in any format.

## Running the web app

Start the Flask app from the `LVI_detection/` directory:

```bash
cd LVI_detection
python app.py
```

Then open the local server in your browser, upload a `.ndpi` slide, and wait for processing to finish.

## Outputs

For each job, the app writes files under `LVI_detection/results/<job_id>/` and keeps uploaded slides under `LVI_detection/uploads/`.

Typical outputs include:

- `status.txt` - current pipeline state
- `log.txt` - processing log
- `lvi_result.json` - slide-level LVI decision
- `lvi_candidates.csv` - vascular candidate tiles
- `lvi_candidates_with_conch_LVI.csv` - tile-level LVI predictions through the CONCH-based LVI judgment model
- `*_vascular_thresholded_heatmap.png` - vascular heatmap
- `*_lvi_thresholded_heatmap.png` - LVI heatmap

A ZIP archive of the result files (.png and .csv) can be downloaded from the web app after processing completes.

## Notes

- The app currently processes one NDPI file at a time per job.
- Large slides can take significant time and disk space to process.
- The scripts assume the model and data paths in the repository layout above.

## Training the models

The repository includes separate training pipelines for the vascular model and the LVI model. The training data, annotation files `.geojson` and slides `.ndpi`, is not committed to GitHub because the NDPI slides and annotation files are too large. Keep those files locally in your own workspace, then generate the datasets and train the models from there.

### 1. Prepare the local training data

Place your training files under `LVI_detection/vascular_detection_model/train_data/` so the scripts can find them. The training workflow expects folders such as:

- `annotations/` for the `.geojson` annotation files
- `ndpiTrain/` for the source training slides, if you keep them in the repo layout

If you have your own NDPI slides and annotations, copy them into your local training directory before running the dataset-generation scripts.

### 2. Train the vascular model

Run the dataset conversion script first, then train the classifier:

```bash
cd LVI_detection/vascular_detection_model
python 1_tile_slides.py
python 2_extract_conch_features.py
python 3_make_dataset_from_points.py
python 4_train_vascular_model.py
```

This creates tile images for each trained slides, `train_data/features/conch_features.npz`, `train_data/vessel_dataset.pkl` and saves the model to `train_data/models/vessel_classifier_conch.pth`.

### 3. Train the LVI model

After the vascular training data is prepared, generate the LVI dataset and train the LVI classifier:

```bash
cd LVI_detection/LVI_detection_model/transfer
python 3_create_lvi_dataset.py
python 4_train_lvi_model.py
```

This creates `../vascular_detection_model/train_data/lvi_dataset.pkl` and saves the model checkpoint to `../vascular_detection_model/train_data/models/lvi_classifier_conch.pth`.

If you skip step2 Train the vascular model, make sure tile images for each trained slides and `train_data/features/conch_features.npz` (output of `1_tile_slides.py` and `2_extract_conch_features.py`) have already in the related folder.

### 4. Run the web app after training

Once both model checkpoints are available, start the Flask app again and upload an NDPI slide to run inference:

```bash
cd LVI_detection
python app.py
```

## Related project

The `CONCH/` folder is a standalone copy of the CONCH model repository and includes usage examples and downstream utilities. If you only want the foundation model documentation, see `CONCH/README.md`.
