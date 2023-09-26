SRC is setup in a way that we first preprocess the data and then extract meaningful features. 
Preprocessing -> feature_extraction -> dataset -> model -> training -> evaluate

All the hyperparameters will be setup on main.yaml using a hydra config file.

We will use wandb's hyperparameter sweep with hydra for conducting hyperparameter tuning.

Use reqs.txt to install all the dependencies.
```


