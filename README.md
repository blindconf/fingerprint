### Computing Fingerprints and running the Open-world setting
To compute the fingerprints run the script as follows:
#### Low-pass-filter
```
python run_modelattribution.py --real-data-path ... --fake-data-path ... --corpus ljspeech --filter-type low_pass_filter --filter-param 1 --scorefunction mahalanobis --transformation Avg_Spec --nfft 128 --hop-len 2 --batchsize 100 --seed 1
```
#### EncoDec
```
python run_modelattribution.py --real-data-path ... --fake-data-path ... --corpus ljspeech --filter-type EncodecFilter --filter-param 24 --scorefunction correlation --transformation Avg_Spec --nfft 2048 --hop-len 128 --batchsize 100 --seed 1
```

### Running the Closed-World setting
To compute in a closed-world setting, select one model from x-vector, vfd-resnet, se-resnet, resnet, lcnn, and fingerprints to train the classifier.

#### Multiclass classifier
```
python src/training/train_model.py --model ... --classification_type multiclass-10 --seed 80
```
#### Binary classifier
```
python src/training/train_model.py --model ... --classification_type binary-10 --seed 80
```
