import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder = 'template')
model = pickle.load(open('taxi.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(abs(prediction[0]))
    return render_template('index.html',prediction_text="Number of Weekly Rides should be {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)