# from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from flask_bcrypt import Bcrypt
# import numpy as np
# import pandas as pd
# import pickle
# from pymongo import MongoClient
# from bson import ObjectId

# # Flask app setup
# app = Flask(__name__)
# app.secret_key = 'healthcaresystem@4566'  # Change this to a secure random key

# # MongoDB setup
# client = MongoClient('mongodb://localhost:27017/')  # Adjust connection string if needed
# db = client['health_app']
# users_collection = db['users']

# # Flask-Login setup
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'
# bcrypt = Bcrypt(app)

# # Load datasets
# sym_des = pd.read_csv("./symtoms_df.csv")
# precautions = pd.read_csv("./precautions_df.csv")
# workout = pd.read_csv("./workout_df.csv")
# description = pd.read_csv("./description.csv")
# medications = pd.read_csv('./medications.csv')
# diets = pd.read_csv("./diets.csv")

# # Load model
# svc = pickle.load(open('./svc.pkl', 'rb'))

# # Disease mapping (assuming svc.predict returns an index)
# disease_list = description['Disease'].tolist()  # List of disease names from description.csv

# # User class for Flask-Login
# class User(UserMixin):
#     def __init__(self, user_id, email, name):
#         self.id = str(user_id)
#         self.email = email
#         self.name = name

# @login_manager.user_loader
# def load_user(user_id):
#     user = users_collection.find_one({'_id': ObjectId(user_id)})
#     if user:
#         return User(user['_id'], user['email'], user['name'])
#     return None

# # Helper function
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])
#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]
#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]
#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]
#     wrkout = workout[workout['disease'] == dis]['workout']
#     return desc, pre, med, die, wrkout

# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

# # Routes
# @app.route("/")
# def index():
#     return render_template("index.html", user=current_user)

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         password = request.form['password']
        
#         if users_collection.find_one({'email': email}):
#             flash('Email already exists!')
#             return redirect(url_for('register'))
        
#         hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#         user_data = {
#             'name': name,
#             'email': email,
#             'password': hashed_password,
#             'health_data': []
#         }
#         result = users_collection.insert_one(user_data)
#         user = User(result.inserted_id, email, name)
#         login_user(user)
#         return redirect(url_for('index'))
    
#     return render_template('register.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
        
#         user = users_collection.find_one({'email': email})
#         if user and bcrypt.check_password_hash(user['password'], password):
#             user_obj = User(user['_id'], user['email'], user['name'])
#             login_user(user_obj)
#             return redirect(url_for('index'))
#         else:
#             flash('Invalid email or password')
    
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))

# @app.route('/predict', methods=['POST'])
# @login_required
# def predict():
#     # Get symptoms from form and split by comma
#     symptoms_input = request.form.get('symptoms', '')  # Default to empty string if not provided
#     symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]  # Split and clean
    
#     if not symptoms:
#         return jsonify({'error': 'No symptoms provided'}), 400

#     # Create input vector
#     input_vector = np.zeros(len(symptoms_dict))
#     for symptom in symptoms:
#         if symptom in symptoms_dict:
#             input_vector[symptoms_dict[symptom]] = 1
    
#     # Predict and convert to disease name
#     pred_index = int(svc.predict([input_vector])[0])  # Convert numpy.int32 to Python int
#     predicted_disease = disease_list[pred_index]  # Map index to disease name
#     desc, pre, med, die, wrkout = helper(predicted_disease)
    
#     # Save prediction to user's health data
#     health_entry = {
#         'symptoms': symptoms,
#         'prediction': predicted_disease,  # Store disease name instead of index
#         'date': pd.Timestamp.now().isoformat()
#     }
#     users_collection.update_one(
#         {'_id': ObjectId(current_user.id)},
#         {'$push': {'health_data': health_entry}}
#     )
    
#     return jsonify({
#         'prediction': predicted_disease,
#         'description': desc,
#         'precautions': [list(p) for p in pre],  # Convert numpy arrays to lists
#         'medications': list(med),
#         'diet': list(die),
#         'workout': list(wrkout)
#     })

# @app.route('/blog')
# def blog():
#     return render_template("blog.html")

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Flask app setup
app = Flask(__name__)
app.secret_key = 'healthcaresystem@4566'  # Change this to a secure random key

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Adjust connection string if needed
db = client['health_app']
users_collection = db['users']

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
bcrypt = Bcrypt(app)

# Load datasets
training = pd.read_csv("./Training.csv")
precautions = pd.read_csv("./precautions_df.csv")
workout = pd.read_csv("./workout_df.csv")
description = pd.read_csv("./description.csv")
medications = pd.read_csv('./medications.csv')
diets = pd.read_csv("./diets.csv")

# sym_des = pd.read_csv("./symtoms_df.csv")
# precautions = pd.read_csv("./precautions_df.csv")
# workout = pd.read_csv("./workout_df.csv")
# description = pd.read_csv("./description.csv")
# medications = pd.read_csv('./medications.csv')
# diets = pd.read_csv("./diets.csv")

# Prepare training data
data_no_duplicates = training.drop_duplicates()
data_no_duplicates = data_no_duplicates.loc[:, (data_no_duplicates != 0).any(axis=0)]
X = data_no_duplicates.iloc[:, :-1]  # Features (symptoms)
y = data_no_duplicates.iloc[:, -1]   # Target (disease)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Define and train multiple models
models = {
    "Logistic Regression": LogisticRegression(C=0.5),
    "SVM": SVC(C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
}

# Train models and evaluate accuracies
model_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy

# Select the best model based on accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, email, name):
        self.id = str(user_id)
        self.email = email
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    if user:
        return User(user['_id'], user['email'], user['name'])
    return None

# Helper function to get disease details
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout

# Routes
@app.route("/")
def index():
    return render_template("index.html", user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        if users_collection.find_one({'email': email}):
            flash('Email already exists!')
            return redirect(url_for('register'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'health_data': []
        }
        result = users_collection.insert_one(user_data)
        user = User(result.inserted_id, email, name)
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = users_collection.find_one({'email': email})
        if user and bcrypt.check_password_hash(user['password'], password):
            user_obj = User(user['_id'], user['email'], user['name'])
            login_user(user_obj)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Get symptoms from form and split by comma
    symptoms_input = request.form.get('symptoms', '')  # Default to empty string if not provided
    symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]  # Split and clean
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Create input vector
    input_vector = np.zeros(len(X.columns))
    for symptom in symptoms:
        if symptom in X.columns:
            input_vector[X.columns.tolist().index(symptom)] = 1
    
    # Predict using the best model
    prediction_encoded = best_model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
    desc, pre, med, die, wrkout = helper(predicted_disease)
    
    # Save prediction to user's health data
    health_entry = {
        'symptoms': symptoms,
        'prediction': predicted_disease,
        'date': pd.Timestamp.now().isoformat()
    }
    users_collection.update_one(
        {'_id': ObjectId(current_user.id)},
        {'$push': {'health_data': health_entry}}
    )
    
    return jsonify({
        'prediction': predicted_disease,
        'description': desc,
        'precautions': [list(p) for p in pre],  # Convert numpy arrays to lists
        'medications': list(med),
        'diet': list(die),
        'workout': list(wrkout),
        'best_model': best_model_name,
        'model_accuracy': model_accuracies[best_model_name]
    })

# @app.route('/blog')
# def blog():
#     return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)