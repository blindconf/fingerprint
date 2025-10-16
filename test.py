import sys, os
# Absolute path to THIS repo (fingerprint_bachelor/DetectingVocoderFingerprints)
# repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Wipe out any other conflicting paths containing "src"
# sys.path = [repo_root] + [p for p in sys.path if "github_fingerprint" not in p]

# If "src" is already loaded from the wrong repo, drop it
# if "src" in sys.modules:
#     del sys.modules["src"]

import random
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
import pandas as pd
import click
from datetime import datetime
from torch import no_grad, argmax, save
from torchmetrics import F1Score, Precision, Recall, Accuracy, ConfusionMatrix, AUROC
from torch.nn import DataParallel
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch import Generator
from src.datasets.utility import collate_fn, get_datasets, StratifiedSampler, fingerprints_collate_fn
from src.training.utility import get_model, get_optimizer_scheduler_loss_function, get_metric, save_confusion_matrix_to_excel, save_heatmap, set_seed
from src.training.invariables import DEV, DEVICE_IDS, BATCH_SIZE, CLASSES, DATASETS
from src.training.arguments import MODELS, CLASSIFICATION_TYPES, PERFORMANCE_METRICS
import re
import torch.multiprocessing as mp
import gc
from src.training.loss_functions import init_loss_functions
from src.datasets.filters import filter_fn
from src.fingerprinting.fingerprinting import load_fingerprints, compute_mahalanobis_scores, assign_vocoders, WaveformToAvgSpec
import torch.nn as nn

from torchaudio.transforms import Spectrogram

@click.command()
@click.option('--model', type=click.Choice(MODELS), required=True, help='Model to train.')
@click.option('--classification_type', type=click.Choice(CLASSIFICATION_TYPES), required=True, help='Classification type.')
@click.option('--performance_metric', type=click.Choice(PERFORMANCE_METRICS), default="f1_score", help='Performance metric.')
#@click.option('--save_id', type=int, required=True, help='ID for saving the model.')
@click.option('--seed', type=int, default=40, help='Random seed.')
@click.option('--corruption_type', type=int, default=0, help='Evaluate under evasion attack: 0 = no evasion attack, 1 = evasion attack')
@click.option('--scale_factor', type=float, default=1.0, help='It compresses or dilates the given impulse response.') 
@click.option('--use_nn', type=int, default=1, help='Set to 1 to use a DNN for the binary classifier under the fingerprint model, 0 to disable.')
@click.option('--corpus', type=click.Choice(["ljspeech", "jsut", "asvspoof", "codecfake"]), required=True, default="ljspeech", help="Dataset corpus to use.")
@click.option('--filter_param', type=str, default=1, required=True, help="Parameter of the filter.")
@click.option('--filter_type', type=click.Choice(["low_pass_filter", "band_pass_filter"]), required=True, default="low_pass_filter", help="Type of filter to apply to the audio signal.")
@click.option('--scorefunction', type=click.Choice(["mahalanobis", "correlation"]), required=True, default="mahalanobis", help="Type of scoring function to use.")
@click.option('--nfft', type=int, default=128, help='Number of FFT points for creating the Spectrograms.')
@click.option('--hop_len', type=int, default=2, help='Hop length for creating the Spectrograms.')
@click.option('--epochs', type=int, default=10, help='Number of epochs.')
@click.option('--num_workers_opt', type=int, default=2, help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')

# By default proportional is False

def main(model, classification_type, performance_metric, seed, corruption_type, scale_factor, use_nn, corpus, filter_param, filter_type, scorefunction, nfft, hop_len, epochs, num_workers_opt):   # save_id

    set_seed(seed)
    init_loss_functions(seed)

    # Set up directory where to save model and logs
    BASE_DIR = os.getcwd()
    URL_DIR_TO_SAVE_MODELS_AND_LOGS = os.path.join(BASE_DIR, "trained_models") 
    url_dir_to_save_model = f'{URL_DIR_TO_SAVE_MODELS_AND_LOGS}/{model}/{corpus}/{seed}/{filter_type}/{classification_type}_filparm_{filter_param}_nfft_{nfft}_hop_{hop_len}'
    MEAN_STD_FOLDER_DIR = os.path.join(BASE_DIR, "mean_std_stats", corpus, filter_type, f"{classification_type}_filparm_{filter_param}_nfft_{nfft}_hop_{hop_len}") 
    
    x = []
    if isinstance(filter_param, float): 
        # Convert to int and then to string
        filter_param = int(filter_param) if filter_param.is_integer() else filter_param 
    file_in = open(f"spectral_filter_coefs/{filter_type}/{filter_param}khz.txt", 'r')

    for y in file_in.read().split('\n'):
        x.append(float(y))
    coef = torch.tensor(x)
    FILTER = filter_fn(1, coef, dev=DEV)

    AVG_SPEC =  WaveformToAvgSpec(n_fft=nfft, hop_length=hop_len, device=DEV).forward

    if not os.path.exists(url_dir_to_save_model):
        os.makedirs(url_dir_to_save_model)

    # Get Dataloader
    if corpus == "jsut":
        sample_rate = 24000
    elif corpus == "ljspeech":
        sample_rate = 22050
    else:
        sample_rate = 16000

    train_ds, validate_ds, test_ds, test_2_ds = get_datasets(
        model=model,
        classification_type=classification_type, 
        seed=seed,
        corruption_type=corruption_type, 
        scale_factor=scale_factor, 
        corpus=corpus,
        mean_std_dir=MEAN_STD_FOLDER_DIR,
        sample_rate=sample_rate,
        filter_fn=FILTER,
        AVG_SPEC=AVG_SPEC
        )

    generator = Generator().manual_seed(seed)
    num_classes = len(CLASSES[classification_type][corpus]) # int(re.findall(pattern=r'\d+', string=classification_type)[0])
    # run vocoder_fingerprint_attribution.py if model == fingerprints and classification_type is multiclass
    if model != "fingerprint" or (model == "fingerprint" and use_nn == 1):
        # Get model
        if model == "fingerprint":
            # my_model = get_model(model=model, classification_type=classification_type, num_classes=num_classes, input_size=nfft // 2 + 1)
            my_model = get_model(model=model, classification_type=classification_type, num_classes=num_classes)
        else:
            my_model = get_model(model=model, classification_type=classification_type, num_classes=num_classes)
        my_model = DataParallel(my_model, device_ids=DEVICE_IDS).to(DEV)

        # Get optimizer, scheduler and loss function
        optimizer, scheduler, loss_function = get_optimizer_scheduler_loss_function(model=model, my_model=my_model, classification_type=classification_type)

        # Set up Metrics
        if "binary" in classification_type:
            task = "binary"
            accuracy = Accuracy(task=task).to(DEV)
            f1 = F1Score(task=task).to(DEV)
            precision = Precision(task=task).to(DEV)
            recall = Recall(task=task).to(DEV)
            confusion_matrix = ConfusionMatrix(task=task).to(DEV)
            auroc = AUROC(task=task).to(DEV)
            prob_func = torch.nn.functional.sigmoid
            preds_func = lambda signals: (signals > 0.5).long()
        else:
            print(f'num_classes: {num_classes}')
            task = "multiclass"
            accuracy = Accuracy(task=task, num_classes=num_classes).to(DEV)
            f1 = F1Score(task=task, num_classes=num_classes, average="macro").to(DEV)
            precision = Precision(task=task, num_classes=num_classes, average="macro").to(DEV)
            recall = Recall(task=task, num_classes=num_classes, average="macro").to(DEV)
            confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes).to(DEV)
            auroc = AUROC(task=task, num_classes=num_classes, average="macro").to(DEV)
            prob_func = lambda signals: torch.nn.functional.softmax(signals, dim=1)
            preds_func = lambda signals: argmax(signals, dim=1)

        testing_score_df = pd.DataFrame(
            columns=["Testing_Accuracy", "Testing_F1_Score", "Testing_Precision", "Testing_Recall", "Testing_AUROC"]
        )
        
        if os.path.exists(f'{url_dir_to_save_model}/best_model.pth'):
            checkpoint = torch.load(f'{url_dir_to_save_model}/best_model.pth',
                            map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage)
            my_model.load_state_dict(checkpoint)
            print("Best model found!", url_dir_to_save_model)

        # --- DataLoader setup ---
        sampler = None
        shuffle = True
        if model == "fingerprint":
            # col_fn = fingerprints_collate_fn
            col_fn = collate_fn
        else:
            col_fn = collate_fn

        if not os.path.exists(f'{url_dir_to_save_model}/best_model.pth'):

            print(f'Initializing {model} model training...')
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE[model], num_workers=num_workers_opt, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=col_fn, shuffle=shuffle, sampler=sampler)
            validation_loader = DataLoader(validate_ds, batch_size=BATCH_SIZE[model], num_workers=num_workers_opt, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=col_fn)

            # Create performance dataframe for training/validating
            training_validating_score_df = pd.DataFrame(
                columns=["Epoch", "Training_Loss", "Validating_Loss", "Training_Accuracy", 
                        "Validating_Accuracy", "Training_F1_Score", "Validating_F1_Score", 
                        "Training_Precision", "Validating_Precision","Training_Recall", 
                        "Validating_Recall", "Training_AUROC", "Validating_AUROC"])

            # Training loop
            n_epochs = epochs
            best_score = 0
            print('Training started...')
            
            transform = Spectrogram(
                                    n_fft = int(0.025 * sample_rate),
                                    hop_length = int(0.01 * sample_rate)
                                    ).to(DEV)

            for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
                # Reset metrics
                accuracy.reset(), f1.reset(), precision.reset()
                recall.reset(), confusion_matrix.reset(), auroc.reset()
                # === Train Phase ===
                my_model.train()
                running_loss = 0.0
                train_batches = len(train_loader)

                for batch  in tqdm(train_loader, desc="Training batches"):
                    # Transfer to device
                    if model == "fingerprint":
                        # waveforms, labels, original_lens = batch
                        waveforms, labels = batch
                        waveforms, labels = waveforms.to(DEV), labels.to(DEV)
                        # inputs = waveform_to_residual(waveforms, FILTER, AVG_SPEC, original_lens)
                        '''
                        if original_lens is None:
                            original_lens = [waveforms.shape[-1]]
                        transformed_features = AVG_SPEC(waveforms, original_lens)
                        filtered_signals = FILTER.forward(waveforms)
                        transformed_filtered_features = AVG_SPEC(filtered_signals, original_lens)
                        inputs = transformed_features - transformed_filtered_features
                        '''
                        # print(waveforms.shape)
                        inputs = transform(waveforms)
                        # print(inputs.shape)
                        # print(waveforms.shape, filtered_signals.shape)
                        # print(inputs.shape)
                        # print(afgasf)
                    else:
                        waveforms, labels = batch
                        inputs, labels = waveforms.to(DEV), labels.to(DEV)
                    if "binary" in classification_type:
                        labels = labels.float().unsqueeze(1)
                    else:
                        labels = labels -1
                    # Zero gradients
                    optimizer.zero_grad()
                    # Forward pass
                    # print(inputs.shape)
                    outputs, features = my_model(inputs)
                    loss = loss_function(outputs, features, labels)
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    # Loss, predictions and probabilities
                    running_loss += loss.item()
                    probabilities = prob_func(outputs)
                    preds = preds_func(probabilities)
                    # Accumulate metrics
                    accuracy.update(preds, labels)
                    f1.update(preds, labels.long())
                    precision.update(preds, labels)
                    recall.update(preds, labels)
                    auroc.update(probabilities, labels)

                # Get scores
                training_loss = running_loss / train_batches
                training_accuracy = accuracy.compute().item()
                training_f1_score = f1.compute().item()
                training_precesion = precision.compute().item()
                training_recall = recall.compute().item()
                training_auroc = auroc.compute().item()
                # === Validation Phase ===
                # Reset metrics
                accuracy.reset(), f1.reset(), precision.reset()
                recall.reset(), confusion_matrix.reset(), auroc.reset()
                
                # LCNN: BatchNorm collapses in evaluation because its running mean and variance are inaccurate for small batches,
                #  especially after channel-halving operations like MFM, causing outputs to shrink even with dropout disabled.
                if model not in ["lcnn"]:
                    my_model.eval()

                validating_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(validation_loader, desc="Validation batches"):
                        if model == "fingerprint":
                            # waveforms, labels, original_lens = batch
                            waveforms, labels = batch
                            waveforms, labels = waveforms.to(DEV), labels.to(DEV)
                            # inputs = waveform_to_residual(waveforms, FILTER, AVG_SPEC, original_lens)
                            '''
                            if original_lens is None:
                                original_lens = [waveforms.shape[-1]]
                            transformed_features = AVG_SPEC(waveforms, original_lens)
                            filtered_signals = FILTER.forward(waveforms)
                            transformed_filtered_features = AVG_SPEC(filtered_signals, original_lens)
                            inputs = transformed_features - transformed_filtered_features
                            '''
                            inputs = transform(waveforms)
                        else:
                            waveforms, labels = batch
                            inputs, labels = waveforms.to(DEV), labels.to(DEV)
                        if "binary" in classification_type:
                            labels = labels.float().unsqueeze(1)
                        else:
                            labels = labels -1
                        # Forward pass
                        outputs, features = my_model(inputs)
                        loss = loss_function(outputs, features, labels)
                        # loss, predictions and probabilities
                        validating_loss += loss.item()                
                        probabilities = prob_func(outputs)
                        preds = preds_func(probabilities)
                        # Accumulate metrics
                        accuracy.update(preds, labels)
                        f1.update(preds, labels.long())
                        precision.update(preds, labels)
                        recall.update(preds, labels)                    
                        auroc.update(probabilities, labels)

                # Get training_validating scores
                validating_loss = validating_loss / len(validation_loader)
                validating_accuracy = accuracy.compute().item()
                validating_f1_score = f1.compute().item()
                validating_precision = precision.compute().item()
                validating_recall = recall.compute().item()
                validating_auroc = auroc.compute().item()
                # Save training_validating scores to dict
                training_validating_scores_dict = {
                    "Epoch": epoch+1,
                    "Training_Loss": training_loss,
                    "Validating_Loss": validating_loss,
                    "Training_Accuracy": training_accuracy,
                    "Validating_Accuracy": validating_accuracy,
                    "Training_F1_Score": training_f1_score,
                    "Validating_F1_Score": validating_f1_score,
                    "Training_Precision": training_precesion,
                    "Validating_Precision": validating_precision,
                    "Training_Recall": training_recall,
                    "Validating_Recall": validating_recall,
                    "Training_AUROC": training_auroc,
                    "Validating_AUROC": validating_auroc
                }
                # Add training_validating scores to dataframe 
                training_validating_score_df.loc[len(training_validating_score_df)] = training_validating_scores_dict
                # Save the best model based on validation F1 score
                metric = get_metric(performance_metric)
                if training_validating_scores_dict[metric] > best_score:
                    best_score = training_validating_scores_dict[metric]
                    print("\n\nNew best model found! Saving...")
                    save(my_model.state_dict(), f'{url_dir_to_save_model}/best_model.pth')
                    save(scheduler.state_dict(), f'{url_dir_to_save_model}/scheduler.pth')
                    save(optimizer.state_dict(), f'{url_dir_to_save_model}/optimizer.pth')
                # Scheduler step
                #if model in ["resnet", "se-resnet", "lcnn", "x-vector"]:
                scheduler.step()
                #print(f'Learning rate at epoch {epoch}: {scheduler.get_last_lr()}')
                # Print Metrics
                print("\n")
                table = [[key, value] for key, value in training_validating_scores_dict.items()]
                print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
                print("\n")
            print("\nTraining Completed.")
            del train_loader
            del validation_loader
            gc.collect()            
            # Add scores to dataframes
            training_validating_score_df.loc[len(training_validating_score_df)] = training_validating_scores_dict
            # Save scores
            training_validating_score_df.to_excel(f'{url_dir_to_save_model}/training_validating_scores.xlsx', index=False)

        # === Test Phase ===
        print("\nTesting the best model...")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=num_workers_opt, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=col_fn)
        # for i in tqdm(test_ds, desc="Testing batches"):
        #  continue
        # my_model.load_state_dict(torch.load(f'{url_dir_to_save_model}/best_model.pth'))
        checkpoint = torch.load(f'{url_dir_to_save_model}/best_model.pth',
                            map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage)
        my_model.load_state_dict(checkpoint)

        if model not in ["lcnn"]:
            my_model.eval()
        
        # Reset Metrics
        accuracy.reset(), f1.reset(), precision.reset()
        recall.reset(), confusion_matrix.reset(), auroc.reset()

        with no_grad():
            for batch in tqdm(test_loader, desc="Testing batches"):
                if model == "fingerprint":
                    # waveforms, labels, original_lens = batch
                    waveforms, labels = batch
                    waveforms, labels = waveforms.to(DEV), labels.to(DEV)
                    # inputs = waveform_to_residual(waveforms, FILTER, AVG_SPEC, original_lens)
                    '''
                    if original_lens is None:
                        original_lens = [waveforms.shape[-1]]
                    transformed_features = AVG_SPEC(waveforms, original_lens)
                    filtered_signals = FILTER.forward(waveforms)
                    transformed_filtered_features = AVG_SPEC(filtered_signals, original_lens)
                    inputs = transformed_features - transformed_filtered_features
                    '''
                    inputs = transform(waveforms)
                else:
                    waveforms, labels = batch
                    inputs, labels = waveforms.to(DEV), labels.to(DEV)
                if "binary" in classification_type:
                    labels = labels.float().unsqueeze(1)
                else:
                    labels = labels -1                    
                # Forward pass
                outputs, features = my_model(inputs)
                probabilities = prob_func(outputs)
                preds = preds_func(probabilities)
                # Update Metrics
                accuracy.update(preds, labels)
                f1.update(preds, labels.long())
                precision.update(preds, labels)
                recall.update(preds, labels)
                confusion_matrix.update(preds, labels)
                auroc.update(probabilities, labels)                
                # '''
        # Get test scores
        testing_accuracy = accuracy.compute().item()
        testing_f1_score = f1.compute().item()
        testing_precision = precision.compute().item()
        testing_recall = recall.compute().item()
        testing_confusion_matrix = confusion_matrix.compute()
        testing_auroc = auroc.compute().item()    
        # Save test scores to dict
        testing_scores_dict = {
            "Testing_Accuracy": testing_accuracy,
            "Testing_F1_Score": testing_f1_score,
            "Testing_Precision": testing_precision,
            "Testing_Recall": testing_recall,
            "Testing_AUROC": testing_auroc,
        }
        # Print test metrics
        print("\n")
        table = [[key, value] for key, value in testing_scores_dict.items()]
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
        print("\n")        
        # Add scores to dataframes
        testing_score_df.loc[len(testing_score_df)] = testing_scores_dict
        # Save scores
        testing_score_df.to_excel(f'{url_dir_to_save_model}/testing_scores_{corruption_type}_factor{scale_factor}_NN{use_nn}.xlsx', index=False)
        save_confusion_matrix_to_excel(conf_matrix=testing_confusion_matrix, destination_url=url_dir_to_save_model, classification_type=classification_type, corruption_type=corruption_type, scale_factor=scale_factor, corpus=corpus)
        save_heatmap(conf_matrix=testing_confusion_matrix.cpu().numpy(), destination_url=url_dir_to_save_model, classification_type=classification_type, corruption_type=corruption_type, scale_factor=scale_factor, corpus=corpus)
        print("Scores saved...")
    else:
        print(f'Initializing fingerprints scoring...')
        # construct command to run vocoder_fingerprint_attribution.py
        FINGERPRINT_DIR = f'{URL_DIR_TO_SAVE_MODELS_AND_LOGS}/{model}/{corpus}/{seed}/{filter_type}'
        # Load fingerprints
        fingerprints = load_fingerprints(FINGERPRINT_DIR, filter_param, scorefunction, nfft, hop_len, CLASSES[classification_type][corpus], DEV)
        all_preds = []
        all_labels = []
        print("Scoring initialized...")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=num_workers_opt, persistent_workers=True, pin_memory=False, generator=generator, collate_fn=fingerprints_collate_fn)
        label_map_inv = {v: k for k, v in DATASETS[corpus].items()}
        print(DATASETS[corpus].items())
        print(label_map_inv)        
        for batch in tqdm(test_loader, desc="Processing test samples"):
            waveforms, labels, original_lens = batch
            waveforms, labels = waveforms.to(DEV), labels.to(DEV)
            if original_lens is None:
                original_lens = [waveforms.shape[-1]]
            transformed_features = AVG_SPEC(waveforms, original_lens)
            filtered_signals = FILTER.forward(waveforms)

            transformed_filtered_features = AVG_SPEC(filtered_signals, original_lens)
            residuals = transformed_features - transformed_filtered_features
    
            if corruption_type == 1:
                orig_labels = labels
                # Random replacement
                rand_labels = torch.randint(1, num_classes + 1, labels.size(), device=labels.device)
                # Make sure replacement is not the same as original
                mask = rand_labels == labels
                while mask.any():
                    rand_labels[mask] = torch.randint(1, num_classes + 1, (mask.sum().item(),), device=labels.device)
                    mask = rand_labels == labels
                scores = evasion_attack_scores(residuals, fingerprints, orig_labels, rand_labels, label_map_inv, DEV)
                labels = rand_labels
            else:
                scores = compute_mahalanobis_scores(residuals, fingerprints, DEV)
            # print(scores)
            preds_tensor = assign_vocoders(scores)
            # print(preds_tensor)
            # print(labels , preds_tensor)
            all_preds.append(preds_tensor)
            all_labels.append(labels)

        print("Scoring finished.")
        # Convert predictions and labels to tensors
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(DEV)
        f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
        precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
        recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(DEV)
        # Shift labels to start from 0
        labels_tensor = labels_tensor - 1
        # Update metrics with final tensors
        accuracy.update(preds_tensor, labels_tensor)
        precision.update(preds_tensor, labels_tensor)
        recall.update(preds_tensor, labels_tensor)
        f1.update(preds_tensor, labels_tensor)
        confusion_matrix.update(preds_tensor, labels_tensor)        
        # Compute final metrics
        accuracy_score = accuracy.compute().item()
        precision_score = precision.compute().item()
        recall_score = recall.compute().item()
        f1_score = f1.compute().item()
        confusion_matrix_score = confusion_matrix.compute().cpu().numpy()        
        # Print metrics
        print(f"Accuracy: {accuracy_score:.4f}")
        print(f"Precision: {precision_score:.4f}")
        print(f"Recall: {recall_score:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix_score}")        
        # Save metrics to Excel file
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [accuracy_score, precision_score, recall_score, f1_score]
        }
        metrics_df = pd.DataFrame(metrics_data)
        output_dir = f'{url_dir_to_save_model}/nn_{use_nn}'
        os.makedirs(output_dir, exist_ok=True)        
        # Save confusion matrix to Excel file
        confusion_matrix_df = pd.DataFrame(confusion_matrix_score)
        if corruption_type == 1:
            confusion_matrix_df.to_excel(f'{output_dir}/evasion_confusion_matrix_{corruption_type}_factor{scale_factor}.xlsx', index=True)  
            metrics_df.to_excel(f'{output_dir}/evasion_testing_scores_{corruption_type}_factor{scale_factor}.xlsx', index=False)
        else:
            confusion_matrix_df.to_excel(f'{output_dir}/confusion_matrix_{corruption_type}_factor{scale_factor}.xlsx', index=True)      
            metrics_df.to_excel(f'{output_dir}/testing_scores_{corruption_type}_factor{scale_factor}.xlsx', index=False)
        save_heatmap(confusion_matrix_df.to_numpy(), output_dir, classification_type, corruption_type, scale_factor, corpus)
        print(f'Metrics and confusion matrix saved in {output_dir}.')
    
if __name__ == "__main__":
    main()