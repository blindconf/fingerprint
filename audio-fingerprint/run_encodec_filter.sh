#!/bin/bash 


qr=24
transformation="Avg_Spec"
batchsize=128
DEVICE=3

# LJSpeech and JSUT: 
# Standard Experiments, correlation 
CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus ljspeech --filter-type EncodecFilter --filter-param $qr --scorefunction correlation --transformation $transformation --batchsize $batchsize
CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus jsut --filter-type EncodecFilter --filter-param $qr --scorefunction correlation --transformation $transformation --batchsize $batchsize

# Background Noise Experiments: 
#CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus ljspeech --filter-type EncodecFilter --filter-param $qr --scorefunction correlation --transformation $transformation  --perturbation noise --batchsize $batchsize
#CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus ljspeech --filter-type EncodecFilter --filter-param $qr --scorefunction mahalanobis --transformation $transformation  --perturbation noise --batchsize $batchsize

# Encodec Perturbation Experiments:
#for qr_noise in 1_5 3 6 12 24
#do 
#    CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus ljspeech --filter-type EncodecFilter --filter-param $qr --scorefunction correlation --transformation $transformation  --perturbation encodec --encodec-qr $qr_noise --batchsize $batchsize
#    CUDA_VISIBLE_DEVICES=$DEVICE python run_modelattribution.py --corpus ljspeech --filter-type EncodecFilter --filter-param $qr --scorefunction mahalanobis --transformation $transformation  --perturbation encodec --encodec-qr $qr_noise --batchsize $batchsize
#done 