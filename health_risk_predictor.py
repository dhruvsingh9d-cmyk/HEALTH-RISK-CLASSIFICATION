import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic training data for the model
np.random.seed(42)

def generate_training_data(n_samples=5000):
    """Generate synthetic health data for training"""
    data = {
        'age': np.random.randint(18, 85, n_samples),
        'height': np.random.normal(170, 10, n_samples),  # cm
        'weight': np.random.normal(75, 15, n_samples),   # kg
        'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_samples),
        'heart_rate': np.random.normal(75, 12, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),  # mg/dL
        'blood_sugar': np.random.normal(100, 25, n_samples),  # mg/dL
        'exercise_hours_week': np.random.randint(0, 15, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 0=No, 1=Yes
        'alcohol_drinks_week': np.random.randint(0, 20, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),  # 1-10 scale
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 0=No, 1=Yes
    }
    
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Create risk labels based on health factors
    risk_score = (
        (df['age'] > 60) * 1.5 +
        (df['bmi'] > 30) * 2 +
        (df['bmi'] > 25) * 1 +
        (df['blood_pressure_systolic'] > 140) * 2 +
        (df['blood_pressure_systolic'] > 130) * 1 +
        (df['cholesterol'] > 240) * 2 +
        (df['cholesterol'] > 200) * 1 +
        (df['blood_sugar'] > 125) * 2 +
        (df['blood_sugar'] > 100) * 1 +
        (df['smoking'] == 1) * 2.5 +
        (df['exercise_hours_week'] < 3) * 1.5 +
        (df['alcohol_drinks_week'] > 14) * 1 +
        (df['sleep_hours'] < 6) * 1 +
        (df['stress_level'] > 7) * 1 +
        (df['family_history'] == 1) * 1.5 +
        (df['heart_rate'] > 100) * 1 +
        np.random.normal(0, 0.5, n_samples)  # Add some randomness
    )
    
    # Categorize into risk levels
    df['risk_level'] = pd.cut(risk_score, 
                               bins=[-np.inf, 4, 8, np.inf], 
                               labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    return df

# Train the model
print("Training the AI model...")
train_data = generate_training_data(5000)

# Prepare features and target
feature_columns = ['age', 'height', 'weight', 'blood_pressure_systolic', 
                   'blood_pressure_diastolic', 'heart_rate', 'cholesterol', 
                   'blood_sugar', 'exercise_hours_week', 'smoking', 
                   'alcohol_drinks_week', 'sleep_hours', 'stress_level', 
                   'family_history', 'bmi']

X = train_data[feature_columns]
y = train_data['risk_level']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_scaled, y)

print("Model training complete!")

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_health_recommendations(input_data, risk_level, risk_probability):
    """Generate personalized health recommendations"""
    recommendations = []
    
    age, height, weight, bp_sys, bp_dia, hr, chol, bs, exercise, smoking, alcohol, sleep, stress, family = input_data
    bmi = calculate_bmi(height, weight)
    
    # BMI recommendations
    if bmi > 30:
        recommendations.append("⚠️ **Obesity Alert**: Your BMI is {:.1f}. Consider consulting a nutritionist for a weight management plan.".format(bmi))
    elif bmi > 25:
        recommendations.append("⚠️ **Overweight**: Your BMI is {:.1f}. Aim for gradual weight loss through balanced diet and exercise.".format(bmi))
    elif bmi < 18.5:
        recommendations.append("⚠️ **Underweight**: Your BMI is {:.1f}. Consider increasing caloric intake with nutritious foods.".format(bmi))
    else:
        recommendations.append("✅ **Healthy BMI**: Your BMI is {:.1f} - Keep maintaining your healthy weight!".format(bmi))
    
    # Blood Pressure
    if bp_sys > 140 or bp_dia > 90:
        recommendations.append("⚠️ **High Blood Pressure**: Your BP is {}/{}. Reduce salt intake, exercise regularly, and consult a doctor.".format(bp_sys, bp_dia))
    elif bp_sys > 130 or bp_dia > 80:
        recommendations.append("⚠️ **Elevated Blood Pressure**: Monitor your BP regularly and maintain a healthy lifestyle.")
    else:
        recommendations.append("✅ **Normal Blood Pressure**: Keep up the good work!")
    
    # Heart Rate
    if hr > 100:
        recommendations.append("⚠️ **Elevated Heart Rate**: Resting heart rate is high. Consider stress management and cardiovascular exercise.")
    elif hr < 60 and exercise < 5:
        recommendations.append("⚠️ **Low Heart Rate**: Unless you're an athlete, consult with a doctor about your low heart rate.")
    
    # Cholesterol
    if chol > 240:
        recommendations.append("⚠️ **High Cholesterol**: Limit saturated fats, increase fiber intake, and consider medical consultation.")
    elif chol > 200:
        recommendations.append("⚠️ **Borderline High Cholesterol**: Focus on heart-healthy diet with more vegetables and omega-3 fatty acids.")
    else:
        recommendations.append("✅ **Healthy Cholesterol Levels**: Maintain your current diet!")
    
    # Blood Sugar
    if bs > 125:
        recommendations.append("⚠️ **High Blood Sugar**: Risk of diabetes. Reduce sugar intake and consult an endocrinologist.")
    elif bs > 100:
        recommendations.append("⚠️ **Elevated Blood Sugar**: Monitor your sugar intake and increase physical activity.")
    
    # Exercise
    if exercise < 3:
        recommendations.append("⚠️ **Low Physical Activity**: Aim for at least 150 minutes of moderate exercise per week (brisk walking, cycling).")
    else:
        recommendations.append("✅ **Good Exercise Habits**: Keep staying active!")
    
    # Smoking
    if smoking == 1:
        recommendations.append("🚨 **Smoking Alert**: Smoking significantly increases disease risk. Consider smoking cessation programs immediately.")
    
    # Alcohol
    if alcohol > 14:
        recommendations.append("⚠️ **Excessive Alcohol**: Reduce alcohol consumption to lower health risks. Consider professional support if needed.")
    
    # Sleep
    if sleep < 6:
        recommendations.append("⚠️ **Insufficient Sleep**: Aim for 7-9 hours of quality sleep. Poor sleep increases disease risk.")
    elif sleep > 9:
        recommendations.append("⚠️ **Excessive Sleep**: Too much sleep may indicate underlying health issues. Consult a doctor.")
    else:
        recommendations.append("✅ **Healthy Sleep Pattern**: Good sleep habits!")
    
    # Stress
    if stress > 7:
        recommendations.append("⚠️ **High Stress**: Practice stress-reduction techniques like meditation, yoga, or counseling.")
    
    # Family History
    if family == 1:
        recommendations.append("⚠️ **Family History**: Regular health screenings are crucial due to genetic predisposition.")
    
    # Age-based recommendations
    if age > 60:
        recommendations.append("📋 **Age Factor**: Regular comprehensive health check-ups recommended for your age group.")
    
    return recommendations

def predict_health_risk(age, height, weight, bp_systolic, bp_diastolic, 
                        heart_rate, cholesterol, blood_sugar, exercise_hours, 
                        smoking, alcohol_drinks, sleep_hours, stress_level, 
                        family_history):
    """Main prediction function"""
    
    # Calculate BMI
    bmi = calculate_bmi(height, weight)
    
    # Convert inputs to model format
    smoking_binary = 1 if smoking == "Yes" else 0
    family_binary = 1 if family_history == "Yes" else 0
    
    # Prepare input data
    input_data = np.array([[age, height, weight, bp_systolic, bp_diastolic, 
                           heart_rate, cholesterol, blood_sugar, exercise_hours,
                           smoking_binary, alcohol_drinks, sleep_hours, 
                           stress_level, family_binary, bmi]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    risk_prediction = model.predict(input_scaled)[0]
    risk_probabilities = model.predict_proba(input_scaled)[0]
    
    # Get probability for predicted class
    risk_dict = {label: prob for label, prob in zip(model.classes_, risk_probabilities)}
    
    # Create result message
    result_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
    
    result_message = f"""
    ## {result_color[risk_prediction]} HEALTH RISK ASSESSMENT RESULT
    
    **Overall Risk Level: {risk_prediction}**
    
    **Risk Probabilities:**
    - Low Risk: {risk_dict.get('Low Risk', 0):.1%}
    - Medium Risk: {risk_dict.get('Medium Risk', 0):.1%}
    - High Risk: {risk_dict.get('High Risk', 0):.1%}
    
    **Your Health Metrics:**
    - BMI: {bmi:.1f}
    - Blood Pressure: {bp_systolic}/{bp_diastolic} mmHg
    - Heart Rate: {heart_rate} bpm
    - Cholesterol: {cholesterol} mg/dL
    - Blood Sugar: {blood_sugar} mg/dL
    """
    
    # Get recommendations
    input_list = [age, height, weight, bp_systolic, bp_diastolic, heart_rate, 
                  cholesterol, blood_sugar, exercise_hours, smoking_binary, 
                  alcohol_drinks, sleep_hours, stress_level, family_binary]
    
    recommendations = get_health_recommendations(input_list, risk_prediction, 
                                                 risk_dict[risk_prediction])
    
    recommendations_text = "\n\n## 📋 PERSONALIZED RECOMMENDATIONS:\n\n" + "\n\n".join(recommendations)
    
    disclaimer = "\n\n---\n**⚠️ DISCLAIMER:** This is an AI-based prediction tool for educational purposes. Always consult with healthcare professionals for medical advice and diagnosis."
    
    return result_message + recommendations_text + disclaimer

# Create Gradio Interface
with gr.Blocks(title="Health Disease Risk Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 AI Health Disease Risk Predictor
    
    Enter your health information below to get an AI-powered assessment of your disease risk and personalized health recommendations.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Basic Information")
            age = gr.Slider(minimum=18, maximum=100, value=30, label="Age (years)", step=1)
            height = gr.Slider(minimum=140, maximum=220, value=170, label="Height (cm)", step=1)
            weight = gr.Slider(minimum=40, maximum=200, value=70, label="Weight (kg)", step=1)
            
            gr.Markdown("### 💓 Vital Signs")
            bp_systolic = gr.Slider(minimum=80, maximum=200, value=120, label="Blood Pressure - Systolic (mmHg)", step=1)
            bp_diastolic = gr.Slider(minimum=50, maximum=130, value=80, label="Blood Pressure - Diastolic (mmHg)", step=1)
            heart_rate = gr.Slider(minimum=40, maximum=150, value=72, label="Resting Heart Rate (bpm)", step=1)
            
            gr.Markdown("### 🔬 Lab Results")
            cholesterol = gr.Slider(minimum=100, maximum=350, value=200, label="Total Cholesterol (mg/dL)", step=1)
            blood_sugar = gr.Slider(minimum=60, maximum=250, value=100, label="Fasting Blood Sugar (mg/dL)", step=1)
        
        with gr.Column():
            gr.Markdown("### 🏃 Lifestyle Factors")
            exercise_hours = gr.Slider(minimum=0, maximum=20, value=3, label="Exercise Hours per Week", step=0.5)
            smoking = gr.Radio(choices=["No", "Yes"], value="No", label="Do you smoke?")
            alcohol_drinks = gr.Slider(minimum=0, maximum=30, value=0, label="Alcoholic Drinks per Week", step=1)
            sleep_hours = gr.Slider(minimum=3, maximum=12, value=7, label="Average Sleep Hours per Night", step=0.5)
            stress_level = gr.Slider(minimum=1, maximum=10, value=5, label="Stress Level (1-10)", step=1)
            
            gr.Markdown("### 🧬 Medical History")
            family_history = gr.Radio(choices=["No", "Yes"], value="No", 
                                     label="Family History of Heart Disease/Diabetes?")
    
    predict_button = gr.Button("🔍 Analyze Health Risk", variant="primary", size="lg")
    
    output = gr.Markdown(label="Results")
    
    predict_button.click(
        fn=predict_health_risk,
        inputs=[age, height, weight, bp_systolic, bp_diastolic, heart_rate, 
                cholesterol, blood_sugar, exercise_hours, smoking, alcohol_drinks, 
                sleep_hours, stress_level, family_history],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 📊 About This Tool
    This AI model uses machine learning to analyze multiple health parameters and predict disease risk levels. 
    The model considers factors like BMI, blood pressure, cholesterol, lifestyle habits, and family history.
    
    **Note:** This tool is for educational and informational purposes only. It is NOT a substitute for professional medical advice.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
