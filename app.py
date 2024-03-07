from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as joblib

app = Flask(__name__)

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.save')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    
    features = [np.array(float_features)]
    normal = scaler.transform(features)
    
    prediction = model.predict(normal)
    
    prediction_text = "No Risk" if prediction[0] == 0 else "High Risk"
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)