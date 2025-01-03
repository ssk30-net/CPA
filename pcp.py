import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model

# Load the trained model
model = load_model('pitch_prediction_model.h5')

# Encoders and scalers
encoder = LabelEncoder()
scaler = StandardScaler()

# Example venue list for user selection
venues = {
    'Lords': 'London',
    'Melbourne cricket stadium': 'Melbourne',
    'Wankhede mumbai': 'Mumbai',
    'Chepauk': 'Chennai'
}

# OpenWeatherMap API setup
API_KEY = 'YOUR_OPENWEATHERMAP_API_KEY'  # Replace with your OpenWeatherMap API key
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'


# Function to fetch real-time weather data from OpenWeatherMap API
def fetch_weather_data(city):
    url = f"{WEATHER_API_URL}?q={city}&appid={API_KEY}&units=metric"  # units=metric for Celsius temperature
    response = requests.get(url)

    if response.status_code == 200:
        weather_info = response.json()

        # Extract relevant weather information
        temperature = weather_info['main']['temp']
        humidity = weather_info['main']['humidity']
        wind_speed = weather_info['wind']['speed']

        return temperature, humidity, wind_speed
    else:
        print(f"Error fetching weather data. Status code: {response.status_code}")
        return None, None, None


# Function for user input simulation and auto-fetch weather data based on venue
def get_user_input():
    # User selects venue and other features like team, day, etc.
    print("Select the venue from the following options:")
    for idx, venue in enumerate(venues.keys()):
        print(f"{idx}: {venue}")
    venue_index = int(input("Enter the venue index: "))
    venue_selected = list(venues.keys())[venue_index]
    city_selected = venues[venue_selected]  # City corresponding to venue for weather data

    team1 = input("Enter Team 1: ")
    team2 = input("Enter Team 2: ")
    pitch_report = input("Enter Pitch Report (good/bad): ")
    day_of_match = int(input("Enter Day of the Match (1-5): "))

    # Fetch real-time weather data using the selected venue's city
    temperature, humidity, wind_speed = fetch_weather_data(city_selected)

    if temperature is None:
        print("Error in fetching weather data. Please try again.")
        return None

    # Simulating run rate input
    run_rate = float(input("Enter Run Rate: "))

    return {
        "venue": venue_selected,
        "team1": team1,
        "team2": team2,
        "pitch_report": pitch_report,
        "day_of_match": day_of_match,
        "temperature": temperature,
        "humidity": humidity,
        "wind": wind_speed,
        "run_rate": run_rate
    }


# Preprocessing user input
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Encode categorical features (use the same encoder that was used during training)
    df['team1'] = encoder.fit_transform(df['team1'])
    df['team2'] = encoder.fit_transform(df['team2'])
    df['pitch_report'] = encoder.fit_transform(df['pitch_report'])
    df['day_of_match'] = df['day_of_match'].astype(int)

    # Scale numeric features (use the same scaler as training)
    df[['temperature', 'humidity', 'wind', 'run_rate']] = scaler.fit_transform(
        df[['temperature', 'humidity', 'wind', 'run_rate']]
    )

    return df


# Real-time prediction
def predict_pitch_condition():
    # Get user input
    user_input = get_user_input()

    if user_input is None:
        print("Failed to get valid user input. Exiting.")
        return

    # Preprocess input
    preprocessed_input = preprocess_input(user_input)

    # Reshape input to match model's input shape (batch_size, timesteps, features)
    input_reshaped = np.array(preprocessed_input).reshape(1, preprocessed_input.shape[1], 1)

    # Make prediction
    prediction = model.predict(input_reshaped)

    # Decode and display the prediction (assuming binary classification: 'good' or 'bad')
    if prediction > 0.5:
        print("Predicted Pitch Condition: Good")
    else:
        print("Predicted Pitch Condition: Bad")


# Run real-time pitch prediction with weather API integration
predict_pitch_condition()
