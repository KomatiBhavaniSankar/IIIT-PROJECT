import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import requests

# Streamlit header
st.header('Image Classification Model')

# Load the model
model = load_model('C://Users//komat//Desktop/IIIT//Image_classify.keras')

# Categories for classification
data_cat = ['Floods', 'Hurricane', 'Volcanic Eruptions', 'Wildfire']

# Image dimensions
img_height = 180
img_width = 180

# File uploader for image input
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and preprocess the image
    image = tf.keras.utils.load_img(uploaded_image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image) / 255.0  # Normalize the image
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display the uploaded image and prediction results
    st.image(uploaded_image, width=200)
    st.write('The image is classified as: ' + data_cat[np.argmax(score)])
    st.write('Confidence: {:.2f}%'.format(np.max(score) * 100))





# OpenWeather API key
API_KEY = "http://api.openweathermap.org/data/2.5/weather?q=' + city + '&units=metric&appid=dfeadd6fc205a9878413ed61ace9e1cf"

# Function to get weather data
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }
    else:
        st.error("City not found or API limit reached.")
        return None

# Streamlit app layout
st.title("Weather App")
city = st.text_input("Enter a city name", "New York")

if st.button("Get Weather"):
    weather = get_weather(city)
    if weather:
        st.write(f"**City:** {weather['city']}")
        st.write(f"**Temperature:** {weather['temperature']} °C")
        st.write(f"**Description:** {weather['description']}")
        st.write(f"**Humidity:** {weather['humidity']} %")
        st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
