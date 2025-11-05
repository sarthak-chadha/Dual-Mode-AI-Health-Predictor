# Dual-Mode AI Disease Predictor (Skin Lesion & Symptom Analysis)

This project is a Streamlit web application that functions as a dual-mode health predictor. It leverages two distinct machine learning models to provide preliminary insights into potential health conditions.


## 🚀 Features

* **Mode 1: Skin Lesion Analysis:**
    * Utilizes a high-performance **hybrid EfficientNet + LightGBM (LGBM) model**.
    * Classifies **8 different types of skin lesions** (e.g., Melanoma, Basal Cell Carcinoma, Benign Keratosis) based on an uploaded image and patient metadata (age, sex, anatomical site).
    * Achieves high accuracy by using a Keras/TensorFlow CNN for feature extraction and an LGBM model for classification.

* **Mode 2: Symptom-Based Analysis:**
    * Employs a **RandomForest Classifier**.
    * Predicts one of **40+ general diseases** (e.g., Fungal Infection, Common Cold, Pneumonia) based on a checklist of symptoms.

* **Integrated AI Health Assistant:**
    * Includes a **Gemini-powered chatbot** in the sidebar to answer general health questions and provide information about the app's functionality.

## 🛠️ Model Architecture

This project uses two separate models, which are loaded by the main Streamlit application.

### 1. Skin Lesion (Hybrid) Model
This is a two-stage model trained in the `LGBM_Model.ipynb` notebook on the ISIC 2019 dataset.

1.  **Stage 1: Feature Extractor (Keras/TensorFlow)**
    * An `EfficientNetB3` model, pre-trained on ImageNet, is fine-tuned on the skin lesion images *and* patient metadata.
    * This model (`skin_hybrid_stage_1.keras`) is not used for final classification but as a powerful feature extractor.

2.  **Stage 2: Classifier (LightGBM)**
    * The features from the Keras model are extracted and used to train a `LGBMClassifier`.
    * This final classifier (`skin_hybrid_lgbm_model.joblib`) is extremely fast and accurate at predicting the lesion type from the extracted features.

### 2. Symptom-Based Model
This is a `RandomForestClassifier` (from Scikit-learn) trained in the `Ai_Project.ipynb` notebook. It uses a CSV dataset of symptoms to predict one of 41 diseases.

## 📂 Project Structure

To run this application, you **must** place the pre-trained models and preprocessing objects in the correct folders. The app (`app2LGBM.py`) expects the following structure:

```
your-project-folder/
│
├── app2LGBM.py                 # The main Streamlit app to run
├── Ai_Project.ipynb            # Notebook to train the symptom model
├── LGBM_Model.ipynb            # Notebook to train the hybrid skin model
├── DeepModel.ipynb             # (Older version notebook, good for reference)
├── app_Deep.py                 # (Older version app, good for reference)
├── requirements.txt            # Python dependencies
├── .gitignore                  # To ignore large files and secrets
└── .streamlit/
    └── secrets.toml            # To store your Gemini API Key
│
├── models/                     # Folder for all trained model files
│   ├── skin_hybrid_stage_1.keras   (Output of LGBM_Model.ipynb)
│   ├── skin_hybrid_lgbm_model.joblib (Output of LGBM_Model.ipynb)
│   ├── symptom_predictor_model.pkl   (Output of Ai_Project.ipynb)
│   └── symptom_label_encoder.pkl   (Output of Ai_Project.ipynb)
│
└── preprocessing_objects/      # Folder for skin model preprocessors
    ├── age_scaler2.pkl         (Output of LGBM_Model.ipynb)
    ├── metadata_encoder2.pkl   (Output of LGBM_Model.ipynb)
    └── label_encoder2.pkl        (Output of LGBM_Model.ipynb)
```
**Note:** The training notebooks save preprocessors like `age_scaler.pkl`. You may need to rename them (e.g., to `age_scaler2.pkl`) to match the paths in `app2LGBM.py`.

## ⚙️ How to Run

### 1. Prerequisites
* Python 3.9+
* Access to the datasets:
    * **ISIC 2019 Training Data** (for the skin model, as `archive.zip`)
    * **Symptom Training/Testing CSVs** (for the symptom model)

### 2. Generate the Models

1.  **Symptom Model:**
    * Place `Training.csv` and `Testing.csv` in your Google Drive (as shown in the notebook).
    * Run the `Ai_Project.ipynb` notebook.
    * Download the output files (`symptom_predictor_model.pkl` and `symptom_label_encoder.pkl`) and place them in the `models/` folder.

2.  **Skin Lesion (Hybrid) Model:**
    * Place the `archive.zip` (ISIC 2019 dataset) in your Google Drive.
    * Run the `LGBM_Model.ipynb` notebook (all cells).
    * This will save `skin_hybrid_stage_1.keras`, `skin_hybrid_lgbm_model.joblib`, and the preprocessing files.
    * Download and place these files into the correct `models/` and `preprocessing_objects/` folders as shown in the structure above.

### 3. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Set Up API Key
The Gemini chatbot requires an API key.

1.  Create a folder named `.streamlit` in the root of your project.
2.  Inside it, create a file named `secrets.toml`.
3.  Add your API key to this file:
    ```toml
    GEMINI_API_KEY = "YOUR_API_KEY_HERE"
    ```
4.  This file is included in the `.gitignore` and will **not** be uploaded to GitHub.

### 5. Run the Streamlit App
Once all models are in place and dependencies are installed, run the app from your terminal:
```bash
streamlit run app2LGBM.py
```

## ⚠️ Disclaimer
This tool is for informational and educational purposes only. The predictions are **not a substitute for a professional medical diagnosis**, advice, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
