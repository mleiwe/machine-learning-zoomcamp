#Libraries and external functions
from flask import Flask, request, jsonify
import pickle

# Load dictionary vectoriser and model file
dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

# Predict function
def pred_Credit(Client):
    X = dv.transform(Client) 
    y_pred = model.predict_proba(X)[0,1]
    Credit = y_pred >= 0.5

    #Now output into ditionary
    result = {'Credit_prob': float(y_pred),
              'Give_Credit': bool(Credit)}
    return result

#Web service 
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json() #gets the value of the request as a JSON and turns it into a JSON
    result = pred_Credit(client)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)