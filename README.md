## Adaptive Learning Model Based on Small Number of Samples

### Environment
- keras
- python3
- sklearn
- pandas
- numpy
- h5py


### DataSet

- IMDB data set

### Use

```
    python adaptive_learning.py
```

### Result

- save result in the 'result' folder


#### dataset

- dataset A (3000/10): train model 1 and using 0.2 part validate the model
- dataset B (100): using model 1 generate model2 data and train model 2 and using 0.2 part validate the model
- dataset C (100): test the model1&model2
- the positive data count is equals the count of the negative data

