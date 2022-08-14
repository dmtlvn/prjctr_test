# prjctr_test

To *"train"* a model, run:

    ```python train.py --datadir DATADIR```

The weights, predictions and metrics are saved to `logs/`.

To start a server, run:

    ```python app.py --model MODEL_FILE```

The app is deployed to the Heroku, and can be accessed at https://prjctr-test.herokuapp.com/. 

Example request:
```
curl -X POST https://prjctr-test.herokuapp.com/ \
    -H 'Content-Type: application/json' \
    -d '{"text":"test text"}'
>>> {"score": -0.01125}
```
