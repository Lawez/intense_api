
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open("kitale_model.pkl", "rb") as file:
    model = pickle.load(file)

def preprocess_input(data):
    df = pd.DataFrame([data])
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    if 'weather_condition' in df.columns:
        df = pd.get_dummies(df, columns=['weather_condition'])
    return df

def generate_recommendation(user_data):
    try:
        input_df = preprocess_input(user_data)
        prediction = model.predict(input_df)[0]
        levels = {
            0: "☁️ Easy does it! Try a light walk, yoga, or just stretching.",
            1: "⛅ You're in the zone! Jog, cycle, or dance your way through today.",
            2: "☀️ Let's go hard! HIIT, running, or lifting is your best bet."
        }
        return levels.get(prediction, "🏋️ Custom level – adjust inputs or ask your coach!")
    except Exception as e:
        return f"⚠️ Something went wrong: {str(e)}"

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json
    if not user_data:
        return jsonify({"error": "Please send JSON data with user fields"}), 400

    recommendation = generate_recommendation(user_data)
    return jsonify({"success": True, "recommendation": recommendation})

if __name__ == '__main__':
    app.run(debug=True)
