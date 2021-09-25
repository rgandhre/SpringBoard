import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('diamond_rings_pricing_model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    print(model.X_columns)
    data = request.get_json(force=True)
    print(data)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    df_test = pd.DataFrame.from_dict(data)
    print(df_test)
    # predictions
    X_test = df_test[model.X_columns]

    print('before predict')
    result = model.predict(X_test)
    print('after predict')
    # send back to browser
    output = {'results': result[0]}
    print('after output')
    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
