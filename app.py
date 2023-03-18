from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model and scaler
model = pickle.load(open('Studen_add_LR_model.pkl', 'rb'))
scaler = pickle.load(open('student_admission_scaler.pkl', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction form
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    gre = float(request.form['gre'])
    toefl = float(request.form['toefl'])
    univ_rating = float(request.form['univ_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = int(request.form['research'])

    # Preprocess the data using the scaler
    data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, research]])
    print(data)
    data_scaled = scaler.transform(data)
    print(data_scaled)

    # Use the model to make a prediction
    prediction = model.predict(data_scaled) 
    prediction=prediction[0]* 100
    prediction=round(prediction,2)

    # Render the results template with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
