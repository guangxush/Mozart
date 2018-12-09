## Adaptive Learning Model Based on Small Number of Samples

### Environment
- keras
- python3
- sklearn
- pandas
- numpy
- h5py


### DataSet

- news

### Use

```
    python adaptive_learning.py
```

### Result

- save result in the 'result' folder


#### dataset

- dataset A (40000/10): train model 1 and using 0.2 part validate the model
- dataset B (1000): using model 1 generate model2 data and train model 2 and using 0.2 part validate the model
- dataset C (1000): test the model1&model2

