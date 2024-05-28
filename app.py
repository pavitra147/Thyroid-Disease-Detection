from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'model/Thyroid_model.pkl'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        features = [float(request.form.get(key)) for key in [
            'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 
            'on_antithyroid_medication', 'thyroid_surgery', 
            'query_hypothyroid', 'query_hyperthyroid', 'pregnant', 
            'sick', 'TSH'
        ]]
        
        # Convert inputs to numpy array and reshape for the model
        input_data = np.array([features])
        
        # Predict using the loaded model
        prediction = model.predict(input_data)
        
        # Get the prediction result
        result = 'Positive for Thyroid Disease' if prediction[0] == 1 else 'Negative for Thyroid Disease'
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
