# --- 1. Import Libraries FIRST ---
import streamlit as st
import pandas as pd
import joblib
import os
import warnings
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
# NEW GEMINI IMPORTS
from google import genai
from google.genai.errors import APIError  # type: ignore

# --- 2. Basic Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Define Disease and Symptom Content ---

# 1. Skin Disease Descriptions (The 8 CNN targets)
SKIN_DISEASE_DESCRIPTIONS = {
    'AK': 'Actinic Keratosis (AK): A rough, scaly patch of skin resulting from sun exposure. It is considered a **precancerous lesion** that may develop into squamous cell carcinoma.',
    'BCC': 'Basal Cell Carcinoma (BCC): The most common form of skin cancer. It usually appears as a pearly, translucent bump, often in sun-exposed areas.',
    'BKL': 'Benign Keratosis (BKL): A non-cancerous skin growth, such as seborrheic keratosis, which often looks waxy, brown, or "pasted on" to the skin.',
    'DF': 'Dermatofibroma (DF): A common, harmless skin lesion that feels like a hard, firm lump or nodule, often appearing on the lower legs.',
    'MEL': 'Melanoma (MEL): The most serious type of skin cancer, which develops in the pigment-producing cells (melanocytes). **Requires immediate medical attention.**',
    'NV': 'Melanocytic Nevus (NV): A common mole. These are usually benign (non-cancerous) growths of pigment-forming cells. Monitoring is advised.',
    'SCC': 'Squamous Cell Carcinoma (SCC): The second most common form of skin cancer, often appearing as a firm, scaly red patch or a wart-like growth.',
    'VASC': 'Vascular Lesion (VASC): A common, benign lesion made up of blood vessels (e.g., cherry angioma or hemangioma).',
}

# 2. General Symptom Descriptions (EXPANDED FOR 40+ DISEASES)
GENERAL_DISEASE_DESCRIPTIONS = {
    'Fungal infection': 'A localized infection typically causing itching, burning, and red/scaly skin. Often treated with topical antifungals.',
    'Common Cold': 'A viral infection causing sneezing, sore throat, and congestion. Typically resolves within 7-10 days.',
    'Pneumonia': 'An infection that inflames the air sacs in one or both lungs. Symptoms include cough with phlegm, fever, and difficulty breathing. Requires medical confirmation.',
    'Jaundice': 'A condition marked by yellow skin/eyes due to excess bilirubin, often indicating liver or gallbladder issues. Requires further testing.',
    'Typhoid': 'A bacterial infection causing high fever, abdominal pain, and headache.',
    'Malaria': 'A disease caused by a parasite, transmitted by mosquitoes, causing fever, chills, and flu-like illness.',
    'Chicken pox': 'A viral infection causing an itchy rash with small, fluid-filled blisters.',
    'Dengue': 'A mosquito-borne viral disease causing fever, rash, and muscle and joint pain.',
    'Migraine': 'A severe headache often accompanied by nausea and light sensitivity.',
    'Cervical spondylosis': 'Age-related wear and tear affecting the neck bones and discs.',
    'Hypothyroidism': 'Underactive thyroid gland, leading to fatigue, weight gain, and depression.',
    'Hyperthyroidism': 'Overactive thyroid gland, causing weight loss, rapid heartbeat, and irritability.',
    'Hepatitis A': 'A viral liver disease causing fatigue, nausea, and jaundice.',
    'Hepatitis B': 'A viral liver disease that can cause chronic infection, leading to liver failure.',
    'Hepatitis C': 'A viral liver disease often leading to chronic infection and liver damage.',
    'Hepatitis D': 'A liver disease caused by the Hepatitis D virus, requires Hepatitis B for transmission.',
    'Hepatitis E': 'A viral liver disease, usually causing acute infection.',
    'Alcoholic hepatitis': 'Liver inflammation due to heavy alcohol consumption.',
    'Tuberculosis': 'A bacterial infection, usually affecting the lungs, causing persistent cough and fever.',
    'Allergy': 'An immune response to foreign substances, causing sneezing, itching, or rash.',
    'AIDS': 'A chronic, potentially life-threatening condition caused by the Human Immunodeficiency Virus (HIV).',
    'Diabetes': 'A chronic condition affecting how the body turns food into energy (high blood sugar).',
    'Gastroenteritis': 'Stomach flu, causing diarrhea and vomiting.',
    'Bronchial Asthma': 'A condition in which airways narrow and swell, producing extra mucus.',
    'Hypertension': 'High blood pressure.',
    'Psoriasis': 'A skin disease causing red, itchy, scaly patches.',
    'Impetigo': 'A highly contagious bacterial skin infection.',
    'Drug Reaction': 'An adverse reaction to medication, often presenting as a rash.',
    'Peptic ulcer disease': 'Sores that develop in the lining of the stomach or small intestine.',
    'Gastroesophageal reflux disease': 'Chronic acid reflux (GERD).',
    'Chronic cholestasis': 'Impaired bile flow.',
    'Urinary tract infection': 'Infection in any part of the urinary system (UTI).',
    'Acne': 'A skin condition causing pimples and blackheads.',
    'Arthritis': 'Joint inflammation.',
    'Varicose veins': 'Swollen, twisted veins.',
    'Dimorphic hemmorhoids(piles)': 'Swollen veins in the anus and rectum.',
    'Heart attack': 'Myocardial infarction. Requires emergency medical attention.',
    'Paroxysmal Positional Vertigo': 'Inner ear problem causing brief spinning sensations.',
    'Abdominal aortic aneurysm': 'Swelling of the aorta in the abdomen.',
    'Chronic kidney disease': 'Progressive loss of kidney function.',
    'Liver cirrhosis': 'Late stage scarring of the liver.',
    'Spinal cord injury': 'Damage to the spinal cord causing loss of function.',
    'Stroke': 'Brain attack due to loss of blood flow.',
    'Systemic lupus erythematosus': 'Chronic autoimmune disease.',
    'Osteoarthritis': 'Common form of arthritis.',
}

# Combine all descriptions for the results page lookup
ALL_DISEASE_DESCRIPTIONS = {**SKIN_DISEASE_DESCRIPTIONS, **GENERAL_DISEASE_DESCRIPTIONS}

# 3. Symptom Groups (EXPANDED FOR 100+ SYMPTOMS)
symptom_groups = {
    "🤧 General / Systemic / Fever": ['fatigue', 'malaise', 'chills', 'shivering', 'high_fever', 'mild_fever', 'sweating', 'weight_loss', 'weight_gain', 'restlessness', 'unsteadiness', 'lethargy', 'dehydration', 'altered_sensorium', 'dizziness', 'anxiety', 'mood_swings'],
    "🧠 Head / Neurological": ['headache', 'slurred_speech', 'loss_of_balance', 'neck_pain', 'stiff_neck', 'loss_of_smell', 'visual_disturbances', 'spinning_movements', 'dizziness', 'unsteadiness'],
    "👁️ Skin / Lesions / Nails": ['itching', 'skin_rash', 'nodal_skin_eruptions', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'blister', 'yellowish_skin', 'red_spots_over_body', 'dischromic_patches', 'internal_itching', 'bruising'],
    "👃 ENT / Respiratory": ['cough', 'phlegm', 'continuous_sneezing', 'runny_nose', 'congestion', 'sinus_pressure', 'throat_irritation', 'patches_in_throat', 'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum', 'breathlessness'],
    "❤️ Chest / Heart": ['chest_pain', 'fast_heart_rate', 'palpitations', 'abnormal_menstruation'],
    "🤢 Gastrointestinal": ['stomach_pain', 'acidity', 'vomiting', 'indigestion', 'nausea', 'abdominal_pain', 'diarrhoea', 'constipation', 'pain_during_bowel_movements', 'bloody_stool', 'irritation_in_anus', 'belly_pain', 'stomach_bleeding', 'ulcers_on_tongue', 'excessive_hunger'],
    "🦴 Joints / Muscles": ['joint_pain', 'muscle_wasting', 'muscle_weakness', 'painful_walking', 'hip_joint_pain', 'knee_pain', 'swelling_joints', 'movement_stiffness', 'back_pain', 'weakness_in_limbs', 'cramps', 'muscle_pain'],
    "💧 Urinary / Fluid": ['burning_micturition', 'spotting_urination', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'polyuria', 'yellow_urine', 'puffy_face_and_eyes', 'fluid_overload', 'dark_urine'],
    "💉 Other Vitals": ['yellowing_of_eyes', 'swelled_lymph_nodes', 'swollen_blood_vessels', 'prominent_veins_on_calf', 'cold_hands_and_feets', 'enlarged_thyroid'],
}
# A flat list of the required symptoms for the checkbox state
ALL_SYMPTOM_NAMES = list(set([sym for group in symptom_groups.values() for sym in group]))

# 4. Symptom Descriptions
symptom_descriptions = {
    'fatigue': 'A general feeling of tiredness or exhaustion.', 'malaise': 'A general feeling of discomfort, illness, or uneasiness.',
    'chills': 'A sensation of coldness, often accompanied by shivering.', 'shivering': 'Involuntary muscle contractions causing trembling.',
    'high_fever': 'Body temperature significantly above normal, typically > 103°F (39.4°C).', 'mild_fever': 'Slightly elevated body temperature (99.5°F - 102°F).',
    'sweating': 'Excessive perspiration.', 'weight_loss': 'Decrease in body weight over a period of time.',
    'weight_gain': 'Increase in body weight.', 'restlessness': 'Inability to rest or relax.',
    'unsteadiness': 'Difficulty maintaining balance.', 'lethargy': 'Lack of energy and enthusiasm.',
    'dehydration': 'Loss of body fluids.', 'altered_sensorium': 'A state of mental confusion or change in awareness.',
    'dizziness': 'Feeling lightheaded or faint.', 'anxiety': 'A feeling of worry, nervousness, or unease.',
    'mood_swings': 'Rapidly changing emotional states.', 'headache': 'Pain in the head.',
    'slurred_speech': 'Difficulty articulating words clearly.', 'loss_of_balance': 'Inability to maintain equilibrium.',
    'neck_pain': 'Ache or discomfort in the neck.', 'stiff_neck': 'Difficulty moving the neck.',
    'loss_of_smell': 'Inability to perceive odors (Anosmia).', 'visual_disturbances': 'Problems with sight, such as blurred vision or double vision.',
    'spinning_movements': 'Sensation that the room is spinning (Vertigo).', 'yellowish_skin': 'Skin turning yellow (Jaundice).',
    'yellowing_of_eyes': 'Yellowing of the whites of the eyes (Scleral icterus).', 'redness_of_eyes': 'Inflammation or bloodshot eyes.',
    'blurred_and_distorted_vision': 'Lack of visual sharpness.', 'puffy_face_and_eyes': 'Swelling around the eyes and face.',
    'cough': 'A sudden, forceful expulsion of air from the lungs.', 'phlegm': 'Thick mucus in the throat or lungs.',
    'continuous_sneezing': 'Frequent, repeated sneezes.', 'runny_nose': 'Excess mucus draining from the nose.',
    'congestion': 'Nasal blockage or stuffiness.', 'sinus_pressure': 'Pain or fullness in the facial sinuses.',
    'throat_irritation': 'Soreness or scratchiness in the throat.', 'patches_in_throat': 'Visible lesions or white spots in the throat.',
    'mucoid_sputum': 'Clear, thick mucus expelled when coughing.', 'rusty_sputum': 'Brownish-red sputum, indicating old blood.',
    'blood_in_sputum': 'Expelling blood with cough (Hemoptysis).', 'breathlessness': 'Difficulty breathing or shortness of breath.',
    'chest_pain': 'Discomfort or tightness in the chest.', 'fast_heart_rate': 'Abnormally rapid heartbeat (Tachycardia).',
    'palpitations': 'Feeling of a rapid, fluttering, or pounding heart.', 'itching': 'An uncomfortable, irritating sensation on the skin.',
    'skin_rash': 'Visible eruption on the skin.', 'nodal_skin_eruptions': 'Small, firm, raised bumps (nodules) under the skin.',
    'pus_filled_pimples': 'Pimples containing pus (Pustules).', 'blackheads': 'Dark spots on the skin (Comedones).',
    'scurring': 'Skin scarring.', 'skin_peeling': 'Shedding of the outer layer of skin.',
    'silver_like_dusting': 'Silvery scales on red patches (often Psoriasis).', 'small_dents_in_nails': 'Pitting of the nails.',
    'blister': 'A small bubble on the skin filled with serum.', 'red_spots_over_body': 'Small red spots (Petechiae or Purpura).',
    'dischromic_patches': 'Areas of skin with abnormal pigmentation.', 'internal_itching': 'Itching sensation deep within the body.',
    'bruising': 'Discoloration of the skin resulting from trauma.', 'stomach_pain': 'Ache or discomfort in the abdominal area.',
    'acidity': 'Heartburn or acid reflux.', 'vomiting': 'Forcible expulsion of stomach contents.',
    'indigestion': 'Difficulty breaking down food.', 'nausea': 'Feeling sick to the stomach.',
    'abdominal_pain': 'Pain in the abdomen.', 'diarrhoea': 'Frequent passage of loose, watery stools.',
    'constipation': 'Infrequent or difficult passage of stool.', 'pain_during_bowel_movements': 'Discomfort while defecating.',
    'bloody_stool': 'Presence of blood in feces.', 'irritation_in_anus': 'Itching or burning around the anus.',
    'belly_pain': 'General abdominal discomfort.', 'stomach_bleeding': 'Gastrointestinal bleeding.',
    'ulcers_on_tongue': 'Open sores on the tongue.', 'excessive_hunger': 'Abnormal increase in appetite (Polyphagia).',
    'joint_pain': 'Discomfort in joints.', 'muscle_wasting': 'Loss of muscle tissue.',
    'muscle_weakness': 'Lack of strength in muscles.', 'painful_walking': 'Pain experienced during walking.',
    'hip_joint_pain': 'Pain originating in the hip joint.', 'knee_pain': 'Pain or discomfort in the knee.',
    'swelling_joints': 'Enlargement of joints.', 'movement_stiffness': 'Difficulty moving a joint freely.',
    'back_pain': 'Ache felt in the back.', 'weakness_in_limbs': 'Reduced strength in arms or legs.',
    'cramps': 'Sudden, involuntary muscle contractions.', 'muscle_pain': 'Ache or soreness in muscles.',
    'burning_micturition': 'Burning sensation while urinating.', 'spotting_urination': 'Blood present in the urine.',
    'bladder_discomfort': 'Pain or pressure in the bladder area.', 'foul_smell_of_urine': 'Urine with an unpleasant odor.',
    'continuous_feel_of_urine': 'Frequent or persistent urge to urinate.', 'polyuria': 'Excessive urination.',
    'yellow_urine': 'Abnormally dark yellow urine.', 'puffy_face_and_eyes': 'Swelling around the eyes and face.',
    'fluid_overload': 'Excess fluid retention in the body.',
    'dark_urine': 'Urine that is unusually dark or brown.',
    'swelled_lymph_nodes': 'Enlargement of lymph glands.',
    'swollen_blood_vessels': 'Visible swelling of veins or arteries.',
    'prominent_veins_on_calf': 'Enlarged veins in the lower leg (Varicose veins).',
    'cold_hands_and_feets': 'Abnormally cold extremities.',
    'enlarged_thyroid': 'Swelling of the thyroid gland (Goiter).',
    'abnormal_menstruation': 'Irregular or unusually heavy/light periods.',
}


# --- Load Resources (Using placeholder paths) ---
CNN_MODEL_PATH = 'models/skin_disease_model_B0.keras' 
CNN_ENCODER_PATH = 'preprocessing_objects/label_encoder.pkl'
CNN_META_COLS_PATH = 'preprocessing_objects/metadata_columns.pkl'
CNN_META_ENCODER_PATH = 'preprocessing_objects/metadata_encoder.pkl'
CNN_AGE_SCALER_PATH = 'preprocessing_objects/age_scaler.pkl'

# --- Simple Symptom Model Files (Using placeholder paths) ---
SYMPTOM_MODEL_PATH = 'models/symptom_predictor_model.pkl'
SYMPTOM_ENCODER_PATH = 'models/symptom_label_encoder.pkl'


@st.cache_resource
def load_all_resources():
    resources = {}
    
    # --- 1. Load CNN Resources (Required) ---
    try:
        resources['cnn_model'] = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False) 
        resources['cnn_label_encoder'] = joblib.load(CNN_ENCODER_PATH)
        resources['cnn_meta_cols'] = joblib.load(CNN_META_COLS_PATH)
        resources['cnn_meta_encoder'] = joblib.load(CNN_META_ENCODER_PATH) 
        resources['cnn_age_scaler'] = joblib.load(CNN_AGE_SCALER_PATH)

        resources['cnn_sex'] = list(resources['cnn_meta_encoder'].categories_[0])
        resources['cnn_sites'] = list(resources['cnn_meta_encoder'].categories_[1])
        resources['cnn_disease_names'] = list(resources['cnn_label_encoder'].classes_)

    except FileNotFoundError as fnf:
        # NOTE: Using mock data if files are missing to allow UI functionality
        st.error(f"⚠️ Missing file: {fnf.filename}. Using mock data for CNN prediction.")
        
        # Mock Encoder and Categories
        mock_classes = ['MEL', 'NV', 'BCC', 'AK', 'SCC', 'BKL', 'DF', 'VASC']
        mock_encoder = OneHotEncoder() 
        mock_encoder.classes_ = np.array(mock_classes)
        resources['cnn_label_encoder'] = mock_encoder
        resources['cnn_disease_names'] = mock_classes
        resources['cnn_sex'] = ['male', 'female', 'unknown']
        resources['cnn_sites'] = ['torso', 'head/neck', 'lower extremity', 'unknown']
        
        # Mock Prediction function
        def mock_cnn_predict(img_data, metadata):
            mock_probs = np.zeros(len(mock_classes))
            if metadata['sex'] == 'male':
                mock_probs[mock_classes.index('BCC')] = 0.5
            else:
                mock_probs[mock_classes.index('NV')] = 0.7
            return {'probs': mock_probs, 'encoder': resources['cnn_label_encoder'], 'mode': 'CNN'}
        resources['cnn_model'] = mock_cnn_predict
        resources['cnn_age_scaler'] = StandardScaler()
        resources['cnn_meta_encoder'] = OneHotEncoder()

    
    # --- 2. Load Symptom Model Resources (Fixed Names) ---
    try:
        resources['symptom_model'] = joblib.load(SYMPTOM_MODEL_PATH)
        resources['symptom_encoder'] = joblib.load(SYMPTOM_ENCODER_PATH)
        resources['symptom_names'] = ALL_SYMPTOM_NAMES 
    except FileNotFoundError as fnf:
        st.warning(f"Symptom-Only Model files not found. (Missing: {fnf.filename}). Using mock data.")
        resources['symptom_model'] = None
        resources['symptom_encoder'] = None
        resources['symptom_names'] = ALL_SYMPTOM_NAMES

    return resources

RESOURCES = load_all_resources()


# --- Define Prediction Functions ---
IMG_WIDTH = 224
IMG_HEIGHT = 224

@st.cache_data(show_spinner=False)
def predict_cnn(img_data, metadata):
    # Check if we are using the mock function due to missing model file
    if not isinstance(RESOURCES['cnn_model'], tf.keras.Model):
        return RESOURCES['cnn_model'](img_data, metadata)
        
    img = Image.open(img_data).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = preprocess_input(np.asarray(img).astype(np.float32))
    img_tensor = tf.expand_dims(img_array, axis=0) 
    
    # Scaling and encoding logic requires the actual objects to be loaded/mocked
    age_scaled = RESOURCES['cnn_age_scaler'].transform(np.array([[metadata['age_approx']]]).reshape(-1, 1))[0, 0]
    cat_raw_values = np.array([[metadata['sex'], metadata['anatom_site_general']]])
    cat_encoded = RESOURCES['cnn_meta_encoder'].transform(cat_raw_values) 
    meta_array = np.concatenate([[age_scaled], cat_encoded[0]])
    meta_tensor = tf.expand_dims(meta_array, axis=0)
    
    pred_probs = RESOURCES['cnn_model'].predict([img_tensor, meta_tensor], verbose=0)[0]
    return {'probs': pred_probs, 'encoder': RESOURCES['cnn_label_encoder'], 'mode': 'CNN'}

@st.cache_data(show_spinner=False)
def predict_symptom_only(user_symptoms_state):
    # Mock data generation if model is missing
    if RESOURCES['symptom_model'] is None:
        if sum(user_symptoms_state.values()) < 3: mock_disease = 'Common Cold'
        elif 'high_fever' in user_symptoms_state and user_symptoms_state['high_fever'] == 1: mock_disease = 'Pneumonia'
        elif 'itching' in user_symptoms_state and user_symptoms_state['itching'] == 1: mock_disease = 'Fungal infection'
        else: mock_disease = 'Hypothyroidism'
        
        all_symptom_diseases = list(GENERAL_DISEASE_DESCRIPTIONS.keys())
        # Use a mock encoder
        mock_encoder = OneHotEncoder() 
        mock_encoder.classes_ = np.array(all_symptom_diseases)
        
        mock_probs = np.zeros(len(all_symptom_diseases))
        if mock_disease in all_symptom_diseases:
            mock_probs[all_symptom_diseases.index(mock_disease)] = 0.8
        else: 
            mock_probs[0] = 0.8 # Fallback to first disease

        return {'disease': mock_disease, 'probs': mock_probs, 'encoder': mock_encoder, 'mode': 'Symptom'}

    # Real prediction logic (if symptom model is present)
    try:
        feature_names = RESOURCES['symptom_model'].feature_names_in_
    except AttributeError:
        # Fallback: Assume the feature names are the full list of ALL_SYMPTOM_NAMES
        feature_names = RESOURCES['symptom_names'] 
        
    # Create input DataFrame, ensuring columns are in the correct order
    input_df = pd.DataFrame([user_symptoms_state], columns=feature_names) 

    # Predict
    # Note: Symptom model prediction output might be different (e.g., returns disease name directly)
    pred_encoded = RESOURCES['symptom_model'].predict(input_df)[0]
    pred_probs = RESOURCES['symptom_model'].predict_proba(input_df)[0]
    
    return {'disease': RESOURCES['symptom_encoder'].inverse_transform([pred_encoded])[0],
            'probs': pred_probs, 'encoder': RESOURCES['symptom_encoder'], 'mode': 'Symptom'}


# --- Initialize Session State ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'prediction_mode' not in st.session_state: st.session_state.prediction_mode = None
if 'prediction_result' not in st.session_state: st.session_state.prediction_result = None

if 'user_symptoms_state' not in st.session_state: 
    st.session_state.user_symptoms_state = {sym:0 for sym in RESOURCES['symptom_names']}
if 'cnn_metadata_state' not in st.session_state: 
    st.session_state.cnn_metadata_state = {'age_approx': 45.0, 'sex': 'unknown', 'anatom_site_general': 'unknown', 'uploaded_image': None}
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# --- Navigation Function ---
def nav(page, reset_inputs=False):
    st.session_state.page = page
    if reset_inputs:
        st.session_state.cnn_metadata_state = {'age_approx': 45.0, 'sex': 'unknown', 'anatom_site_general': 'unknown', 'uploaded_image': None} 
        st.session_state.user_symptoms_state = {sym: 0 for sym in RESOURCES['symptom_names']}
        st.session_state.prediction_result = None
        st.session_state.prediction_mode = None
    st.rerun()
    
# --- GLOBAL GEMINI CLIENT INITIALIZATION ---
SYSTEM_PROMPT = (
    "You are a helpful, professional, and empathetic AI Health Assistant "
    "for a disease prediction application. Your purpose is to answer user questions "
    "about the app's features (Skin Image Analysis or Symptom Only mode) "
    "or provide general, non-diagnostic information about the diseases and symptoms "
    "listed in the app. ALWAYS include a DISCLAIMER that you are not a medical doctor "
    "and the user must consult a healthcare professional for diagnosis."
)

# Initialize the Gemini client safely using the Streamlit secret key
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    # Use a placeholder client if initialization fails
    client = None

# --- NEW: Persistent Help Bot Function ---
def persistent_help_bot():
    """
    Renders a fixed-position help button and a chat interface in the sidebar,
    connected to the Gemini API via the globally initialized client.
    """
    # Use HTML/CSS to provide a visual cue for the bot icon
    st.markdown("""
    <style>
    .help-bot-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    .help-bot-icon {
        cursor: pointer;
        font-size: 30px;
        padding: 10px;
        border-radius: 50%;
        background-color: #4CAF50; /* A friendly green color */
        color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        text-align: center;
    }
    .help-bot-icon:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    </style>
    <div class="help-bot-container">
        <div class="help-bot-icon">🤖</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🤖 Gemini Health Assistant")
        st.caption("Powered by Google Gemini. Please ask brief, general questions.")
        
        # Display history
        for message in st.session_state.chat_history:
            st.chat_message(message["role"]).write(message["content"])

        # Input field
        user_query = st.chat_input("Ask a question...")
        
        if user_query:
            # 1. Store user message in history and display it immediately
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                 st.write(user_query)

            # --- API Logic Start ---
            if not client:
                gemini_response = "The Gemini API service failed to initialize. Please ensure you have set your API key in the `.streamlit/secrets.toml` file correctly."
            else:
                with st.spinner("Gemini is thinking..."):
                    try:
                        # Convert Streamlit chat history to the structure required by the API
                        chat_messages = [
                            {"role": m["role"], "parts": [{"text": m["content"]}]}
                            for m in st.session_state.chat_history
                        ]
                        
                        # Call the API
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", 
                            contents=chat_messages,
                            config={"system_instruction": SYSTEM_PROMPT} 
                        )
                        
                        gemini_response = response.text
                    
                    except APIError as e:
                        gemini_response = f"**API Error (Gemini):** Could not complete the request. This may be due to an invalid key or content policy. Details: {e}"
                    except Exception as e:
                        gemini_response = f"**An unexpected Python error occurred:** {e}"

            # 3. Store and display AI response
            st.session_state.chat_history.append({"role": "assistant", "content": gemini_response})
            st.rerun()

# --- Execute Persistent Bot Function ---
persistent_help_bot()
# ------------------------------------


# --- Page Rendering ---

if st.session_state.page == 'home':
    st.title("🩺 Dual-Mode Health Predictor")
    
    st.info("Your AI Health Companion — assisting early awareness and guidance. **(See the 🤖 icon/sidebar to chat with the Gemini Health Assistant!)**")
    
    st.markdown("Predict potential health conditions based on symptoms or by analyzing a skin image.")
    st.warning("Remember: This tool provides **preliminary insights** and is **not a substitute for professional medical advice**.", icon="⚠️")
    st.divider()
    if st.button("Start Diagnosis", type="primary", use_container_width=True):
        nav('options')

elif st.session_state.page == 'options':
    st.header("Choose Diagnosis Method")
    st.markdown("Select how you'd like to provide information.")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Symptoms Only 📝", type="secondary", use_container_width=True):
            st.session_state.prediction_mode = "symptoms_only"
            nav("input")
    with col2:
        if st.button("Skin Image + Metadata 🖼️", type="secondary", use_container_width=True):
            st.session_state.prediction_mode = "image+meta"
            nav("input")
    st.divider()
    if st.button("⬅️ Back to Home"):
        nav("home", reset_inputs=True)

elif st.session_state.page == 'input':
    # --- Dynamic Header ---
    if st.session_state.prediction_mode == "symptoms_only":
        st.title("Enter Your Symptoms 📝")
        st.markdown("Select all general symptoms that apply.")
        
    else: # image+meta mode
        st.title("Skin Image Analysis Input 🖼️")
        st.markdown(f"**This mode checks for the 8 major skin conditions:** {', '.join(RESOURCES['cnn_disease_names'])}.")
    
    st.divider()

    # --- INPUT LOGIC SWITCH ---
    if st.session_state.prediction_mode == "image+meta":
        col_img, col_meta = st.columns(2)
        
        with col_img:
            with st.container(border=True):
                st.subheader("1. Upload Lesion Image")
                uploaded_file = st.file_uploader("Choose a JPEG/PNG file:", type=["jpg", "jpeg", "png"], key="uploaded_file_key", help="Upload a clear image of the lesion.")
                if uploaded_file:
                    st.session_state.cnn_metadata_state['uploaded_image'] = uploaded_file
                    st.image(uploaded_file, caption='Lesion Preview', use_container_width=True)
                else:
                    st.session_state.cnn_metadata_state['uploaded_image'] = None

        with col_meta:
            with st.container(border=True):
                st.subheader("2. Enter Patient Metadata")
                
                age_value = int(st.session_state.cnn_metadata_state['age_approx'])
                age = st.number_input("Approximate Age:", min_value=0, max_value=100, value=age_value, step=1, key='age_input')
                st.session_state.cnn_metadata_state['age_approx'] = float(age)

                options_sex = sorted([s for s in RESOURCES['cnn_sex'] if s != 'unknown']) + ['unknown']
                sex = st.radio("Sex:", options=options_sex, index=options_sex.index(st.session_state.cnn_metadata_state['sex']), horizontal=True, key='sex_input')
                st.session_state.cnn_metadata_state['sex'] = sex
                
                options_site = sorted([s for s in RESOURCES['cnn_sites'] if s != 'unknown']) + ['unknown']
                site = st.selectbox("Anatomical Site:", options=options_site, index=options_site.index(st.session_state.cnn_metadata_state['anatom_site_general']), key='site_input', help="Where is the lesion located on the body? Required for prediction.")
                st.session_state.cnn_metadata_state['anatom_site_general'] = site
        
        validation_check = not st.session_state.cnn_metadata_state['uploaded_image']
        prediction_func = lambda: predict_cnn(st.session_state.cnn_metadata_state['uploaded_image'], st.session_state.cnn_metadata_state)

    else: # symptoms_only mode
        # Simple Symptom Checkbox Interface
        with st.container(border=True):
            st.subheader("Select Symptoms")
            st.markdown("Filter symptoms by category or view all. Select all that apply.")
            
            groups_list = ["Show All Symptoms"] + list(symptom_groups.keys())
            selected_group_radio = st.radio("Filter Symptoms:", groups_list, index=0, horizontal=True, label_visibility="collapsed", key='group_selector')

            items_to_display = RESOURCES['symptom_names'] if selected_group_radio == "Show All Symptoms" else symptom_groups.get(selected_group_radio, [])
            
            st.caption("Hover over a symptom name for a description.")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            checkbox_count = 0
            
            for s in items_to_display:
                if s in st.session_state.user_symptoms_state:
                    with cols[checkbox_count % 3]:
                        display_name = s.replace("_", " ").strip().capitalize()
                        is_checked = st.checkbox(
                            display_name, 
                            value=bool(st.session_state.user_symptoms_state[s]),
                            key=f"cb_{s}",
                            help=symptom_descriptions.get(s, f"Select if you are experiencing: {display_name}")
                        )
                        st.session_state.user_symptoms_state[s] = 1 if is_checked else 0
                    checkbox_count += 1
        
        validation_check = sum(st.session_state.user_symptoms_state.values()) == 0
        prediction_func = lambda: predict_symptom_only(st.session_state.user_symptoms_state)

    # --- Prediction Button Logic (Shared) ---
    st.divider()
    col_pred, col_back, _ = st.columns([1, 1, 4])
    with col_pred:
        if st.button("▶️ Predict", type="primary", use_container_width=True):
            if validation_check:
                st.error("⚠️ Please provide the necessary input (Image/Metadata or selected Symptoms).")
            elif st.session_state.prediction_mode == "symptoms_only" and RESOURCES['symptom_model'] is None and sum(st.session_state.user_symptoms_state.values()) == 0:
                   st.error("Symptom-Only Model is not available and no symptoms were selected for mock testing.")
            else:
                with st.spinner("Analyzing data..."):
                    try:
                        result_dict = prediction_func()
                        
                        # Process results to a standard format for display
                        predicted_index = np.argmax(result_dict['probs'])
                        predicted_disease = result_dict['encoder'].classes_[predicted_index]
                        confidence = result_dict['probs'][predicted_index] * 100
                        
                        top_indices = result_dict['probs'].argsort()[-3:][::-1]
                        top_conditions = [(result_dict['encoder'].classes_[i], result_dict['probs'][i] * 100) for i in top_indices]
                        
                        st.session_state.prediction_result = {
                            'disease': predicted_disease, 
                            'confidence': confidence, 
                            'top_conditions': top_conditions,
                            'mode': result_dict['mode']
                        }
                        nav("results")
                    except Exception as e:
                        st.exception(e)
                        st.error(f"Prediction error: {e}")
                        print(f"Prediction function failed: {e}")

    with col_back:
        if st.button("⬅️ Back to Options", use_container_width=True, type="secondary"):
            nav("options")

elif st.session_state.page == "results":
    st.title("Prediction Results 📊")
    result = st.session_state.prediction_result

    if result:
        main_disease = result['disease']
        mode_text = "Skin Image + Metadata Analysis" if result['mode'] == 'CNN' else "General Symptom Analysis"
        
        # --- Main Result Container (Minimal UI replacement) ---
        with st.container(border=True):
            is_critical = main_disease in ['MEL', 'BCC', 'SCC', 'Heart attack', 'Stroke']
            
            if is_critical:
                st.error(f"### 🛑 CRITICAL WARNING: {main_disease}")
            else:
                st.success(f"### ✅ Predicted Condition: {main_disease}")
            
            st.markdown(f"**Primary Diagnosis:** **{main_disease}**")

            st.divider()
            st.markdown(f"**Confidence Score:** **{result['confidence']:.2f}%**")
            st.progress(int(result['confidence'])) # Confidence Bar
            st.caption(f"Result from: **{mode_text}**")
        
        # --- Disease Description ---
        description = ALL_DISEASE_DESCRIPTIONS.get(main_disease, "No specific description is available for this condition. Please consult a doctor for details.") 
        st.subheader(f"About {main_disease}")
        st.info(description, icon="💡") 

        # --- Other Possible Conditions (Top 3) - REMOVED PER USER REQUEST ---
        
        st.warning("⚠️ **Disclaimer:** This AI prediction is **not a diagnosis**. Consult a healthcare professional for accurate medical advice.", icon="ℹ️")
    else:
        st.warning("No prediction result found. Please start a new diagnosis.")

    st.divider()
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 New Diagnosis", use_container_width=True, type="secondary"):
            nav("options", reset_inputs=True)
    with col2:
        if st.button("🏠 Home", use_container_width=True, type="secondary"):
            nav("home", reset_inputs=True)

else:
    st.error("Invalid page state. Returning home.")
    st.session_state.page = 'home'
    st.rerun()