from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.iolib.smpickle import load_pickle
import json

import functions

app = FlaskAPI(__name__)


def run_model(df: pd.DataFrame):
    cleaned = functions.clean_data(df)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_scaler = StandardScaler()
    imputed_std = functions.prep_dataset(cleaned, imputer, std_scaler)

    result = load_pickle("model.pickle")

    with open('variables.json', 'r') as f:
        variables = json.load(f)
    variables.sort()

    outcomes = functions.get_outcomes(imputed_std, result, variables)

    predictions = functions.make_prediction(outcomes)
    sorted_df = df.reindex(sorted(df.columns), axis=1)
    result = pd.concat([predictions, sorted_df], axis=1)

    return result, functions.cutoff


@app.route('/predict', methods=['POST'])
def post_route():
    if request.method == 'POST':

        functions.print_versions()

        print('reading data')
        data = request.get_json(force=True)

        if not isinstance(data, list):
            print('converting to list')
            data = [data]

        df = pd.DataFrame.from_dict(data, orient='columns')

        print('running model')
        predictions, cutoff = run_model(df)
        input_cols = [col for col in df.columns if col not in
                      ['business_outcome', 'cutoff', 'phat']]

        results_dict = []
        for index, row in predictions.iterrows():
            input_val = dict(row)
            output_val = dict()
            output_val['business_outcome'] = input_val['business_outcome']
            output_val['cutoff'] = round(cutoff, 4)
            output_val['phat'] = round(input_val['phat'], 4)
            output_val['variables'] = {col: input_val[col]
                                       for col in input_cols}
            results_dict.append(output_val)

        return jsonify(results_dict)


if __name__ == "__main__":
    app.run(debug=True, port=1313)
