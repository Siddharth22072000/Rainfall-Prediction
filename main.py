
from flask import Flask, jsonify, request
import  pickle
import numpy as np

model = pickle.load(open('best_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():

    state = request.form.get('state')
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))

    input_query= np.array([[year, month]])
    result= model.predict(input_query)[0]

    return jsonify(str(result))
if __name__ == '__main__':
    app.run(debug=True)
