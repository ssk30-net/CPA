import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model

# Load the trained model
model = load_model('pitch_prediction_model.h5')

# Encoders and scalers
encoder = LabelEncoder()
scaler = StandardScaler()

# AccuWeather API setup
API_KEY = 'YOUR_ACCUWEATHER_API_KEY'  # Replace with your actual AccuWeather API key
WEATHER_API_URL = 'http://dataservice.accuweather.com/currentconditions/v1/'
LOCATIONS_API_URL = 'http://dataservice.accuweather.com/locations/v1/cities/search'


# Function to fetch AccuWeather location key using stadium name
def get_location_key(stadium_name):
    params = {
        'apikey': API_KEY,
        'q': stadium_name
    }
    response = requests.get(LOCATIONS_API_URL, params=params)
    if response.status_code == 200 and len(response.json()) > 0:
        return response.json()[0]['Key']  # Return the first location key match
    else:
        print(f"Error fetching location key for {stadium_name}. Status code: {response.status_code}")
        return None


# Function to fetch real-time weather data from AccuWeather API
def fetch_weather_data(location_key):
    url = f"{WEATHER_API_URL}{location_key}?apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        weather_info = response.json()[0]

        # Extract relevant weather information
        temperature = weather_info['Temperature']['Metric']['Value']
        humidity = weather_info['RelativeHumidity']
        wind_speed = weather_info['Wind']['Speed']['Metric']['Value']

        return temperature, humidity, wind_speed
    else:
        print(f"Error fetching weather data. Status code: {response.status_code}")
        return None, None, None


# Function to web scrape match details from ESPNcricinfo
def fetch_match_data(stadium_name):
    # ESPNcricinfo URL (this is an example, you will need to modify based on the real URL)
    url = f"https://www.espncricinfo.com/search/results?q={stadium_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Web scraping logic to find teams, pitch report, and run rate
    # You may need to adjust the logic based on the ESPNcricinfo page structure
    team1 = "Team 1 Name"  # Example: Extracted from the page
    team2 = "Team 2 Name"  # Example: Extracted from the page
    pitch_report = "good"  # Example: Extracted from the page
    run_rate = 5.6  # Example: Extracted from the page

    return team1, team2, pitch_report, run_rate


# Function for user input and fetching data
def get_user_input():
    # User inputs stadium
    stadium_name = input("Enter the stadium name: ")

    # Fetch match data from ESPNcricinfo
    team1, team2, pitch_report, run_rate = fetch_match_data(stadium_name)

    # Fetch location key for weather data
    location_key = get_location_key(stadium_name)
    if location_key is None:
        print("Error fetching location key for the stadium. Please try again.")
        return None

    # Fetch real-time weather data
    temperature, humidity, wind_speed = fetch_weather_data(location_key)
    if temperature is None:
        print("Error fetching weather data. Please try again.")
        return None

    # Simulating day of match input
    day_of_match = int(input("Enter Day of the Match (1-5): "))

    return {
        "venue": stadium_name,
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
    df['day_of_match'] = df['day_of
