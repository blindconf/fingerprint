import os
import random
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
import pandas as pd
import wandb
import click
import sys
from datetime import datetime
from torch import no_grad, argmax, save
from torchmetrics import F1Score, Precision, Recall, Accuracy, ConfusionMatrix, AUROC
from torch.nn import DataParallel
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch import Generator
from src.datasets.utility import collate_fn, get_datasets, StratifiedSampler
from src.training.utility import get_model, get_optimizer_scheduler_loss_function, get_metric, save_confusion_matrix_to_excel, save_heatmap, set_seed
from src.training.invariables import DEV, DEVICE_IDS, URL_DIR_TO_SAVE_MODELS_AND_LOGS, BATCH_SIZE
from src.training.arguments import MODELS, CLASSIFICATION_TYPES, PERFORMANCE_METRICS
import re
import torch.multiprocessing as mp
import gc
from src.training.loss_functions import init_loss_functions

@click.command()
@click.option('--model', type=click.Choice(MODELS), required=True, help='Model to train.')
@click.option('--classification_type', type=click.Choice(CLASSIFICATION_TYPES), required=True, help='Classification type.')
@click.option('--performance_metric', type=click.Choice(PERFORMANCE_METRICS), default="f1_score", help='Performance metric.')
#@click.option('--save_id', type=int, required=True, help='ID for saving the model.')
@click.option('--seed', type=int, default=40, help='Random seed.')
@click.option('--proportional/--non-proportional', default=False, show_default=True, help='Only for binary classification.')

def main(model, classification_type, performance_metric, seed, proportional):   # save_id

    set_seed(seed)
    init_loss_functions(seed)
    # run vocoder_fingerprint_attribution.py if model == fingerprints and classification_type is multiclass
    if model == "fingerprints":

        print(f'Initializing fingerprints scoring...')
        # construct command to run vocoder_fingerprint_attribution.py
        command = [
            sys.executable,
            "src/training/vocoder_fingerprint_attribution.py",
            "--seed", str(seed),
            "--classification_type", classification_type,
            "--performance_metric", performance_metric
        ]

        if proportional:
            command.append("--proportional")

        # Run the other script
        subprocess.run(command)

        # Exit the current script after the other script finishes
        sys.exit(0)

    # Set up directory where to save model and logs
    if "binary" in classification_type:
        prop = "proportional" if proportional else "non-proportional"
        url_dir_to_save_model = f'{URL_DIR_TO_SAVE_MODELS_AND_LOGS}{model}/{classification_type}/{prop}/{seed}'
    else: 
        url_dir_to_save_model = f'{URL_DIR_TO_SAVE_MODELS_AND_LOGS}{model}/{classification_type}/{seed}'

    if not os.path.exists(url_dir_to_save_model):
        os.makedirs(url_dir_to_save_model)
    print(f'Initializing {model} model training...')


    # Set up wandb

    wandb.init(
        # set the wandb project where this run will be logged
        
        project= 
            f'{model}-{classification_type}-proportional-{seed}' if "binary" in classification_type and proportional 
            else f'{model}-{classification_type}-{seed}',

        # track hyperparameters and run metadata
        config={
        "architecture": f'{model}',
        "classification_type": f'{classification_type}',
        "epochs": 100,
        },
        name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )    

    # Get Dataloader
        # Set up dataloaders
    train_ds, validate_ds, test_ds = get_datasets(
        model=model,
        classification_type=classification_type, 
        proportional="proportional" if proportional else "non-proportional", 
        seed=seed)

    sampler = None
    shuffle = True
    col_fn = collate_fn
    if model == "vfd-resnet":
        train_labels = train_ds.df['label'].tolist()
        sampler = StratifiedSampler(train_labels, batch_size=BATCH_SIZE[model])
        shuffle = False
        col_fn = None

    generator = Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=col_fn, shuffle=shuffle, sampler=sampler)
    validation_loader = DataLoader(validate_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=col_fn)

    # Get model
    my_model = get_model(model=model, classification_type=classification_type)
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
        preds_func = lambda signals: (signals > 0.5).float()

    else:
        num_classes = int(re.findall(pattern=r'\d+', string=classification_type)[0])
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

    # Create performance dataframe for training/validating
    training_validating_score_df = pd.DataFrame(
        columns=["Epoch", "Training_Loss", "Validating_Loss", "Training_Accuracy", 
                 "Validating_Accuracy", "Training_F1_Score", "Validating_F1_Score", 
                 "Training_Precision", "Validating_Precision","Training_Recall", 
                 "Validating_Recall", "Training_AUROC", "Validating_AUROC"])
    testing_score_df = pd.DataFrame(
        columns=["Testing_Accuracy", "Testing_F1_Score", "Testing_Precision", "Testing_Recall", "Testing_AUROC"]
    )

    # Training loop
    epochs = 100
    best_score = 0
    print('Training started...')
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Reset metrics
        accuracy.reset(), f1.reset(), precision.reset()
        recall.reset(), confusion_matrix.reset(), auroc.reset()


        # === Train Phase ===
        my_model.train()
        running_loss = 0.0
        train_batches = len(train_loader)


        for waveforms, labels in tqdm(train_loader, desc="Training batches"):
            # Transfer to device
            if "binary" in classification_type:
                labels = labels.float().unsqueeze(1)
            waveforms, labels = waveforms.to(DEV), labels.to(DEV)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, features = my_model(waveforms)
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
            f1.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            auroc.update(probabilities, labels)

            # step for vfd-model
            #if model == "vfd-resnet":
            #    scheduler.step()


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
        
        my_model.eval()
        validating_loss = 0.0
        with no_grad():
            for waveforms, labels in tqdm(validation_loader, desc="Validation batches"):
                if "binary" in classification_type:
                    labels = labels.float().unsqueeze(1)
                waveforms, labels = waveforms.to(DEV), labels.to(DEV)

                # Forward pass
                outputs, features = my_model(waveforms)
                loss = loss_function(outputs, features, labels)

                # loss, predictions and probabilities
                validating_loss += loss.item()                
                probabilities = prob_func(outputs)
                preds = preds_func(probabilities)

                # Accumulate metrics
                accuracy.update(preds, labels)
                f1.update(preds, labels)
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


        # Add training_validating scores to dataframe and log them to wandb
        training_validating_score_df.loc[len(training_validating_score_df)] = training_validating_scores_dict
        wandb.log(training_validating_scores_dict, step=epoch+1)


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

    wandb.finish()
    print("\nTraining Completed.")

    del train_loader
    del validation_loader
    gc.collect()

    # === Test Phase ===
    print("\nTesting the best model...")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=col_fn)
    my_model.load_state_dict(torch.load(f'{url_dir_to_save_model}/best_model.pth'))
    my_model.eval()


    # Reset Metrics
    accuracy.reset(), f1.reset(), precision.reset()
    recall.reset(), confusion_matrix.reset(), auroc.reset()


    with no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing batches"):
            if "binary" in classification_type:
                labels = labels.float().unsqueeze(1)
            waveforms, labels = waveforms.to(DEV), labels.to(DEV)

     
            # Forward pass
            outputs, features = my_model(waveforms)
            probabilities = prob_func(outputs)
            preds = preds_func(probabilities)

            # Update Metrics
            accuracy.update(preds, labels)
            f1.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            confusion_matrix.update(preds, labels)
            auroc.update(probabilities, labels)

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
    training_validating_score_df.loc[len(training_validating_score_df)] = training_validating_scores_dict
    testing_score_df.loc[len(testing_score_df)] = testing_scores_dict
    

    # Save scores
    training_validating_score_df.to_excel(f'{url_dir_to_save_model}/training_validating_scores.xlsx', index=False)
    testing_score_df.to_excel(f'{url_dir_to_save_model}/testing_scores.xlsx', index=False)
    save_confusion_matrix_to_excel(conf_matrix=testing_confusion_matrix, destination_url=url_dir_to_save_model, classification_type=classification_type)
    save_heatmap(conf_matrix=testing_confusion_matrix.cpu().numpy(), destination_url=url_dir_to_save_model, classification_type=classification_type)
    print("Scores saved...")

if __name__ == "__main__":
    main()