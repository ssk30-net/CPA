from flask import Flask, render_template, request
import numpy as np
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained Keras model
model = load_model('model/pitch_condition_model.h5')

# Define feature names
FEATURE_NAMES = ['moisture', 'hardness', 'grass_cover']

# Scrape pitch data from a hypothetical webpage
def scrape_pitch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example of extracting pitch data from the page
        moisture = float(soup.find(id='moisture').text)
        hardness = float(soup.find(id='hardness').text)
        grass_cover = float(soup.find(id='grass_cover').text)

        return [moisture, hardness, grass_cover]
    except Exception as e:
        raise ValueError(f"Error scraping data: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if data source is URL or manual input
        if 'url' in request.form and request.form['url']:
            # Scrape data from the provided URL
            url = request.form['url']
            inputs = scrape_pitch_data(url)
        else:
            # Use manual input data
            inputs = [float(request.form[feature]) for feature in FEATURE_NAMES]

        inputs_array = np.array([inputs])

        # Predict pitch condition
        prediction = model.predict(inputs_array)
        condition_classes = ['Good', 'Average', 'Poor']
        predicted_condition = condition_classes[np.argmax(prediction)]

        return render_template('index.html', prediction=f"The pitch condition is predicted as: {predicted_condition}")
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
