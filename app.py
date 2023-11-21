# Import necessary packages
from flask import Flask, jsonify, request
import os
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from flask_cors import CORS

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initialize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initialize Google Cloud Storage client
client = storage.Client()

# API route path is "/api/forecast"
# This API will accept only POST requests
@app.route('/api/forecast', methods=['POST'])
def forecast():
    # Extract data from the request
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]

    # Create a DataFrame from the GitHub data
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    # Rename the columns as per Prophet's requirement
    df = df.rename(columns={'ds': 'ds', 'y': 'y'})

    # Initialize Prophet model
    model = Prophet()

    # Fit the model with the GitHub data
    model.fit(df)

    # Make a DataFrame with future dates for forecasting
    future = model.make_future_dataframe(periods=365)  # Forecasting for 1 year (365 days)

    # Generate the forecast
    forecast = model.predict(future)

    # Plot the forecast components
    fig = model.plot_components(forecast)

    # Save the forecast plot as an image
    forecast_plot_path = f"static/images/forecast_{type}_{repo_name}.png"
    plt.savefig(forecast_plot_path)

    # Add your unique Bucket Name if you want to run it locally
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    # Upload the forecast plot image to Google Cloud Storage
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(forecast_plot_path)
    new_blob.upload_from_filename(filename=forecast_plot_path)

    # Construct the response with the image URL
    json_response = {
        "forecast_plot_image_url": f"{BUCKET_NAME}/{forecast_plot_path}"
    }

    # Return the response to the Flask microservice
    return jsonify(json_response)

# Run the app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
