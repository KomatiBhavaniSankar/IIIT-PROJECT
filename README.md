

---

# 🌍 Natural Disasters Prediction Web App

An interactive web application for predicting and monitoring natural disasters using machine learning and real-time weather data. Built with Streamlit as the front end, the app uses a deep learning model to classify satellite images of disasters and integrates real-time weather information from the OpenWeatherMap API. Users can upload satellite images of disasters for classification and access live weather data.

## 🔥 Features

- **Satellite Image Classification**: Classifies natural disasters from satellite images, including floods, hurricanes, volcanic eruptions, and wildfires, using a TensorFlow/Keras deep learning model.
- **Real-Time Weather Data**: Retrieves live weather information such as temperature, humidity, wind speed, and general conditions based on user-provided city names via the OpenWeatherMap API.
- **User Session Management**: Utilizes Streamlit's Session State to manage user sessions within the app.

## 🛠️ Tech Stack

### Front-End

- **Streamlit**: Provides an interactive, responsive web-based user interface for displaying predictions and weather information.

### Back-End

- **TensorFlow & Keras**: Used to build and train a deep learning image classification model to identify disaster types from satellite images.
- **OpenWeatherMap API**: Supplies real-time weather data based on user-input city names.

### Libraries and Tools

- **NumPy**: For efficient data handling and preprocessing, especially for manipulating image data.
- **Requests**: Facilitates API requests to OpenWeatherMap for weather details.
- **TensorFlow Image Utilities**: Manages image processing tasks like loading, resizing, and normalization, essential for accurate model predictions.
- **Streamlit's Session State**: Manages user session data.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- [Python 3.7+](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)

You’ll also need an API key for [OpenWeatherMap](https://openweathermap.org/).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/natural-disasters-prediction.git
   cd natural-disasters-prediction
   ```

2. **Set Up a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a `.env` file in the project root and add the following:

   ```plaintext
   OPENWEATHER_API_KEY=your_openweather_api_key
   ```

### Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

Open a browser and go to `http://localhost:8501` to interact with the application.

## 🖥️ Project Structure

- **`app.py`**: Main application file that handles routing and integrates all functionalities.
- **`model/`**: Contains the pre-trained TensorFlow/Keras model for satellite image classification.
- **`utils/`**: Helper functions for API requests, image processing, and model prediction.
- **`requirements.txt`**: Lists all dependencies required to run the application.

## 📄 Key Functionalities

### 1. Disaster Classification

Users can upload a satellite image of a natural disaster, and the model will classify it into one of four categories:
   - Flood
   - Hurricane
   - Volcanic Eruption
   - Wildfire

   The classification is handled by a Convolutional Neural Network (CNN) trained with TensorFlow and Keras, using satellite images to ensure accuracy.

### 2. Real-Time Weather Data

Using the OpenWeatherMap API, users can:
   - View current weather conditions based on their city input.
   - Access additional data like humidity, temperature, and wind speed to help monitor potential risks.

### 3. User Session Management

Streamlit's Session State is utilized to:
   - Manage user sessions, preserving information between interactions.
   - Maintain app state during user interaction for a seamless experience.

## 🤝 Contributing

Contributions are welcome! If you want to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## 📜 License

This project is licensed under the MIT License. 

## ✨ Acknowledgments

- **OpenWeatherMap** for providing real-time weather data.
- **TensorFlow & Keras** for powering the image classification model.

---
