# How to test a new method?
Please follow these three steps to test a new method in our benchmark.

1. Add its code in `external`. Please make sure the new method have an independent train and inference stage.
2. Provide a configuration file in `methods` to set hyperparameters and access the codebase. Two classes should be defined in this file: one contains all hyperparameters, the other provides three functions - `apply()`, `inference()` and `load_pred()` - to access the codebase using the hyperparameters specified in the first class.
3. Add its name in `main.py` and `utils.py`.
