# [Paper Under Revision] Lightweight Model Attribution and Detection of Synthetic Speech via Audio Residual Fingerprints.

We propose a simple, training-free method for detecting AI-generated speech and attributing it to its source model by leveraging standardized average residuals as distinctive fingerprints. 
Our approach effectively addresses single-model attribution, multi-model attribution, and synthetic versus real speech detection, achieving high accuracy and robustness across diverse speech synthesis systems.

![logo](resources/single_model.png) 

This paper [Lightweight Model Attribution and Detection of Synthetic Speech via Audio Residual Fingerprints](https://github.com/blindconf/fingerprint) is currently under revision to SaTML 2026. A demo with a selection of fake audio samples from different AI-Generated models employed in our experiments is available online: [Fingerprint Demo](https://blindconf.github.io/fingerprint_demo/).
> As speech generation technologies continue to advance in quality and accessibility, the risk of malicious use cases, including impersonation, misinformation, and spoofing, increases rapidly.
> As speech generation technologies advance, so do risks of impersonation, misinformation, and spoofing.
> We present a lightweight, training-free approach for detecting synthetic speech and attributing it to its source model.
> Our method addresses three tasks: (1) single-model attribution in an open-world setting, (2) multi-model attribution in a closed-world setting, and (3) real vs. synthetic speech classification.
> The core idea is simple: we compute standardized average residuals—the difference between an audio signal and its filtered version—to extract model-agnostic fingerprints that capture synthesis artifacts.
> Experiments across multiple synthesis systems and languages show AUROC scores above 99\%, with strong reliability even when only a subset of model outputs is available.
> The method maintains high performance under common audio distortions, including echo and moderate background noise, while data augmentation can improve results in more challenging conditions.
> Out-of-domain detection is performed using Mahalanobis distances to in-domain residual fingerprints, achieving an F1 score of 0.91 on unseen models.
> These results demonstrate that our technique is efficient, generalizable, and practical for digital forensics and security applications.

### Computing Fingerprints and running the Open-world setting
To compute the fingerprints run the script as follows:
#### Low-pass-filter
```
python run_modelattribution.py \
  --corpus ljspeech \
  --data_path /data/DATASETS/WaveFake/ \
  --real_data_path /data/DATASETS/LJSpeech-1.1/wavs/ \
  --window_size 8 \
  --hop_size 0.125 \
  --seed 40 \
  --batchsize 100
```
#### EncoDec
```
python run_modelattribution.py \
  --corpus ljspeech \
  --data_path /data/DATASETS/WaveFake/ \
  --real_data_path /data/DATASETS/LJSpeech-1.1/wavs/ \
  --window_size 8 \
  --hop_size 0.125 \
  --seed 40 \
  --batchsize 100
```

### Running the Closed-World setting
To compute in a closed-world setting, select one model from x-vector, vfd-resnet, se-resnet, resnet, lcnn, and fingerprints to train the classifier.

#### Multiclass classifier
```
python train_model.py \
  --corpus asvspoof \
  --window_size 25 \
  --hop_size 10 \
  --seed 40 \
  --model se-resnet \
  --classification_type multiclass \
  --batchsize 128
  ```
#### Binary classifier
```
python train_model.py \
  --corpus asvspoof \
  --window_size 25 \
  --hop_size 10 \
  --seed 40 \
  --model se-resnet \
  --classification_type binary \
  --batchsize 128
  ```
