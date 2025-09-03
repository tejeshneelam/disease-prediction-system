# =============================
# prediction/views.py
# =============================

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
import joblib
import numpy as np
import os
import requests
import json

# Cache of trained models (loaded once per process)
MODELS = {}

CHATBOT_RESPONSES = {
    "bmi": "BMI stands for Body Mass Index. It's a measure of body fat based on your height and weight. You can calculate it using your weight in kilograms divided by the square of your height in meters.",
    "hba1c": "HbA1c stands for Glycated Hemoglobin. It's a measure of your average blood glucose (sugar) levels over the past 2 to 3 months. It's used to diagnose and monitor diabetes.",
    "blood glucose": "Blood glucose level is the amount of glucose in your blood. It is a key indicator of your body's ability to process sugar and is a critical metric for diabetes.",
    "cp": "CP stands for Chest Pain. It is a key indicator of heart disease. The numbers typically represent different types of chest pain, such as typical angina, atypical angina, non-anginal pain, and asymptomatic.",
    "trestbps": "Trestbps stands for Resting Blood Pressure. It is the blood pressure measured when a person is at rest.",
    "chol": "Chol stands for Cholesterol. It's a type of fat found in your blood. High cholesterol can increase the risk of heart disease.",
    "smoking": "Smoking can be a significant risk factor for various diseases. The input usually asks whether you smoke (1) or not (0).",
    "gender": "For the Liver dataset, Gender is typically represented as a number (0 for Female, 1 for Male). For other datasets like Diabetes, it is a category like 'Male' or 'Female'.",
    "age": "Age is an important factor in many disease predictions. Please enter your age in years.",
    # Add more rules as needed
}

# views.py
from transformers import pipeline

# Load once
qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert-base-uncased-distilled-squad", 
    framework="pt"   # force PyTorch
)

KNOWLEDGE_BASE = """
Diabetes is a chronic condition that affects how the body turns food into energy. Symptoms include frequent urination, thirst, and fatigue. Treatment includes lifestyle changes and medication.
Heart disease refers to conditions affecting the heart, such as blocked vessels, chest pain, and heart attacks. Treatment includes lifestyle changes, medications, or surgery.
Hospitals provide healthcare services. You can find nearby hospitals using our hospital locator feature.
BMI (Body Mass Index) = weight(kg) / height(m^2). Normal BMI is 18.5–24.9.
...
"""

def chatbot(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if not user_message:
            return JsonResponse({'message': "Please type a question."})

        try:
            answer = qa_pipeline({
                'question': user_message,
                'context': KNOWLEDGE_BASE
            })
            response = answer['answer']
        except Exception:
            response = "Sorry, I couldn't find an answer. Please consult a doctor."

        return JsonResponse({'message': response})
    return JsonResponse({'message': 'Hello, how can I help you today?'})

def hospitals(request):
    """Find nearby hospitals using Google Places API."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lat = data.get('lat')
            lon = data.get('lon')
            api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
            
            if not all([lat, lon, api_key]):
                print("❌ API Key or coordinates missing!")
                return JsonResponse({'error': 'Missing data or API key'}, status=400)
            
            # Use a slightly larger radius for better results (e.g., 5000 meters = 5 km)
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=5000&type=hospital&key={api_key}"
            
            print(f"🌍 Requesting URL: {url}") # <-- FOR DEBUGGING
            
            response = requests.get(url)
            
            print(f"✅ Google API Status: {response.status_code}") # <-- FOR DEBUGGING
            print(f"✅ Google API Response: {response.text}") # <-- FOR DEBUGGING
            
            results = response.json()
            
            hospitals_list = []
            for result in results.get('results', []):
                hospitals_list.append({
                    'name': result.get('name'),
                    'address': result.get('vicinity')
                })
            
            return JsonResponse({'hospitals': hospitals_list})
            
        except (json.JSONDecodeError, KeyError) as e:
            return JsonResponse({'error': f'Invalid JSON or key: {e}'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def _model_path(*parts: str) -> str:
    return os.path.join(settings.BASE_DIR, *parts)


def load_models():
    """Load all machine-learning models into memory if present."""
    global MODELS
    try:
        heart_model_path = _model_path('prediction', 'models', 'heart_model.pkl')
        diabetes_model_path = _model_path('prediction', 'models', 'diabetes_model.pkl')
        liver_model_path = _model_path('prediction', 'models', 'liver_model.pkl')

        if os.path.exists(heart_model_path):
            MODELS['heart'] = joblib.load(heart_model_path)
        if os.path.exists(diabetes_model_path):
            MODELS['diabetes'] = joblib.load(diabetes_model_path)
        if os.path.exists(liver_model_path):
            MODELS['liver'] = joblib.load(liver_model_path)

        print(f"✅ Models loaded: {list(MODELS.keys())}")
    except Exception as e:
        # Don't crash the server if a model fails to load
        print(f"❌ Error loading models: {e}")


# Load when the module is imported (Django dev server reload-safe)
load_models()


# ---------- Helpers ----------

def safe_float(value, default=0.0):
    """Convert input to float, empty string -> default"""
    try:
        return float(value) if value not in ("", None) else default
    except ValueError:
        return default


# ---------- Views ----------

def home(request):
    return render(request, 'prediction/home.html')


def contact(request):
    return render(request, 'prediction/contact.html')


@require_POST
@csrf_protect
def predict(request):
    """Handle prediction requests. Returns JSON."""
    disease_type = request.POST.get('disease')

    # Validate model availability
    model = MODELS.get(disease_type)
    if model is None:
        return JsonResponse({'error': f"Model '{disease_type}' not available"}, status=400)

    # Collect raw inputs (strings) except csrf + disease
    payload = {k: v for k, v in request.POST.items() if k not in ('csrfmiddlewaretoken', 'disease')}

    try:
        if disease_type == 'heart':
            # Columns from heart.csv (excluding 'target')
            feature_order = [
                'age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal'
            ]
            row = [safe_float(payload.get(col)) for col in feature_order]
            X = np.array(row, dtype=float).reshape(1, -1)

        elif disease_type == 'liver':
            # Columns from Liver.csv (excluding 'Dataset/Diagnosis')
            feature_order = [
                'Age','Gender','BMI','AlcoholConsumption','Smoking',
                'GeneticRisk','PhysicalActivity','Diabetes',
                'Hypertension','LiverFunctionTest'
            ]
            row = [safe_float(payload.get(col)) for col in feature_order]
            X = np.array(row, dtype=float).reshape(1, -1)

        elif disease_type == 'diabetes':
            expected_cols = getattr(model, 'feature_names_in_', None)

            # Raw form values
            gender = payload.get('gender', 'Other')            # 'Male'|'Female'|'Other'
            smoke = payload.get('smoking_history', 'never')    # categories

            numerics = {
                'age': safe_float(payload.get('age')),
                'hypertension': safe_float(payload.get('hypertension')),
                'heart_disease': safe_float(payload.get('heart_disease')),
                'bmi': safe_float(payload.get('bmi')),
                'HbA1c_level': safe_float(payload.get('HbA1c_level')),
                'blood_glucose_level': safe_float(payload.get('blood_glucose_level')),
            }

            default_dummy_cols = [
                'gender_Female', 'gender_Male', 'gender_Other',
                'smoking_history_No Info', 'smoking_history_current', 'smoking_history_ever',
                'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
            ]

            def build_with_expected(expected):
                vec = {col: 0.0 for col in expected}
                for k, v in numerics.items():
                    if k in vec:
                        vec[k] = v
                g_col = f'gender_{gender}'
                s_col = f'smoking_history_{smoke}'
                if g_col in vec:
                    vec[g_col] = 1.0
                if s_col in vec:
                    vec[s_col] = 1.0
                return np.array([[vec[c] for c in expected]], dtype=float)

            if expected_cols is not None:
                X = build_with_expected(list(expected_cols))
            else:
                row = [
                    numerics['age'], numerics['hypertension'], numerics['heart_disease'],
                    numerics['bmi'], numerics['HbA1c_level'], numerics['blood_glucose_level']
                ]
                dummies = []
                for col in default_dummy_cols:
                    if col == f'gender_{gender}' or col == f'smoking_history_{smoke}':
                        dummies.append(1.0)
                    else:
                        dummies.append(0.0)
                X = np.array([row + dummies], dtype=float)

        else:
            return JsonResponse({'error': 'Unknown disease type'}, status=400)

        # Prediction
        y = model.predict(X)
        pred = int(round(float(y[0])))
        return JsonResponse({'prediction': pred})

    except Exception as e:
        return JsonResponse({'error': f'Invalid input or model error: {e}'}, status=400)
