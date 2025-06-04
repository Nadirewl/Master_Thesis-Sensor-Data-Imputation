# MLFlow setup
Inspired by https://gitlab.mya-dev.io/Analytics/scavenge-port-inspection/anomaly-detection

See usage examples here: https://gitlab.mya-dev.io/search?search=mlflow&nav_source=navbar&project_id=824&group_id=1267&search_code=true&repository_ref=main 

Or a full (complex) example here: https://gitlab.mya-dev.io/Analytics/scavenge-port-inspection/anomaly-detection/-/blame/main/src/anomaly_detection/model_builder.py?ref_type=heads#L402

## Requirements
### Setup
- VSCode (Or manual port-forwarding)
- `poetry add mlflow@latest`

### Run MLflow server 
###### in this shell
```sh
poetry run mlflow server --backend-store-uri sqlite:///experiments.db
```
###### in background (requires manual port-forwarding)
```sh
nohup poetry run mlflow server --backend-store-uri sqlite:///experiments.db &> mlflow.out &
```
### Kill MLflow server (to restart)
```sh
lsof -i :5000 -Fp | cut -c 2- | xargs kill -9
```



[comment]: <> (
    # Allow the notebooks to connect to the mlflow server
    ssh -o ExitOnForwardFailure=yes -R 5000:127.0.0.1:5000 christoffer-sommerlund-thesis
    ssh -o ExitOnForwardFailure=yes -R 5000:127.0.0.1:5000 christoffer-sommerlund-patchcore
)




## Basics
When performing your execution, wrap your main training loop in a mlflow run

```python
def training():
    ...

if __name__ == '__main__':
    for hyperparams in ...:
        with mlflow.start_run():
            mlflow.log_param("n_estimators", hyperparams['n_estimators'])
            model = ...
            model.fit(...)
            score = ...
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(model, "model")
```
