# three-body-nn

TO run the training of the single step model do:
```
uv run src/single_step/train_single_step.py
```
For the one shot model do:
```
uv run src/one_shot/train_single_shot.py
```


Models will get saves into outputs/

Then to test the model against our Verlet Solver ground truth:

```
uv run src/compare_models.py
```