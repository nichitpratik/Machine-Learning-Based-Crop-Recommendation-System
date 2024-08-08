from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get input values from the form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Create feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    # Reshape for prediction
    single_pred = np.array(feature_list).reshape(1, -1)

    # Predict probabilities for all crops
    probabilities = model.predict_proba(single_pred)[0]

    # Create a DataFrame for easier manipulation
    crop_df = pd.DataFrame({
        'Crop': range(1, 23),  # Assuming 22 crops
        'Probability': probabilities
    })

    # Sort crops by probability in descending order
    crop_df = crop_df.sort_values(by='Probability', ascending=False)

    # Map crop numbers to crop names
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Get top 5 crops with highest probabilities
    top_crops = crop_df.head(5)
    # Get All crops with highest probabilities
    # top_crops = crop_df

    # Get crop names with sequence numbers
    crop_names_with_numbers = [(i+1, crop_dict[crop]) for i, crop in enumerate(top_crops['Crop'])]

    # Construct result string
    result = "Top 5 Crop :"
    for seq_num, crop_name in crop_names_with_numbers:
        result += f"{seq_num}. {crop_name} "

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
