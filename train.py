import argparse
import json
import os
import csv
from pathlib import Path
from typing import Union
from model import Model


def read_data(path: Union[str, Path]) -> dict:
    with open(path) as file:
        reader = csv.reader(file)
        columns = next(reader)
        data = dict(zip(columns, zip(*list(reader))))
        for key in ['target', 'standard_error']:
            if key in data:
                data[key] = list(map(float, data[key]))
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train a model')
    parser.add_argument("--datadir", '-d', help="Path to the CommonLit Readability Prize dataset")
    args = parser.parse_args()

    logdir = Path('logs/')
    datadir = Path(args.datadir)
    os.makedirs(logdir, exist_ok = True)

    X_train = read_data(datadir / 'train.csv')
    X_test = read_data(datadir / 'test.csv')

    model = Model(regularizer = 100).fit(X_train['excerpt'], X_train['target'])
    metric = model.evaluate(X_train['excerpt'], X_train['target'])
    y_pred = model.predict(X_test['excerpt'] + ('test text',))


    model.save(logdir / 'model.pkl')

    with open(logdir / 'metrics.json', 'w') as file:
        json.dump({"rmse": metric}, file)

    with open(logdir / 'predictions.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(zip(X_test['id'] + ('000000000',), y_pred))
