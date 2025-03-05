
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from datetime import datetime
import gc
import os
import pickle
import re
from tabulate import tabulate
from torch.nn import DataParallel
import torch
from torchmetrics import AUROC, F1Score, Precision, Recall, Accuracy, ConfusionMatrix
import wandb
from src.training.invariables import CLASSES, DEV, DEVICE_IDS, FINGERPRINT_DIR, MEAN_STD_FOLDER_DIR, URL_DIR_TO_SAVE_MODELS_AND_LOGS, BATCH_SIZE
from src.training.utility import get_metric, get_model, get_optimizer_scheduler_loss_function, save_confusion_matrix_to_excel, save_heatmap, set_seed
from src.datasets.utility import fingerprints_collate_fn, get_datasets
from torch.utils.data import DataLoader
import argparse
from src.datasets.custom_dataset import waveform_to_residual
import pandas as pd
from src.training.arguments import CLASSIFICATION_TYPES, PERFORMANCE_METRICS
from tqdm import tqdm
import pandas as pd
import torch
from tqdm import tqdm
import re


def load_fingerprints(fingerprint_dir, classes):
    fingerprints = {}
    transformation_keywords = ["Avg_Spec", "Avg_MFCC", "Avg_Mel"]

    for vocoder_dir in sorted(os.listdir(fingerprint_dir)):
        if not vocoder_dir in classes:
            continue
        vocoder_path = os.path.join(fingerprint_dir, vocoder_dir)
        if not os.path.isdir(vocoder_path):
            continue
        
        fingerprint_files = sorted([f for f in os.listdir(vocoder_path) if f.endswith("_fingerprint.pickle")])
        invcov_files = sorted([f for f in os.listdir(vocoder_path) if f.endswith("_invcov.pickle")])

        if len(fingerprint_files) != len(invcov_files):
            raise ValueError(f"Mismatch in number of fingerprint and invcov files in {vocoder_dir}")
        
        for fp_file, invcov_file in zip(fingerprint_files, invcov_files):
            # Extract parameters from filenames
            params = {}
            transformation_type = None

            # Search for transformation keywords in the filename
            for keyword in transformation_keywords:
                if keyword in fp_file:
                    transformation_type = keyword
                    break  # Stop searching once a match is found
            
            filename_parts = fp_file.replace(".pickle", "").split("_")
            for item in filename_parts:
                key_value = item.split("=")
                if len(key_value) == 2:

                    params[key_value[0]] = key_value[1]

            params["transformation_type"] = transformation_type

            # Load fingerprint and invcov
            with open(os.path.join(vocoder_path, fp_file), "rb") as f:
                fingerprint = pickle.load(f)
            with open(os.path.join(vocoder_path, invcov_file), "rb") as f:
                invcov = pickle.load(f)

            # Initialize vocoder dictionary if not already present
            if vocoder_dir not in fingerprints:
                fingerprints[vocoder_dir] = []

            # Append the fingerprint and related data
            fingerprints[vocoder_dir].append({
                "fingerprint": fingerprint.to(DEV),
                "invcov": invcov.to(DEV),
                "params": params
            })
    return fingerprints


def mahalanobis_score(fingerprint, batch_residual, invcov):
    
    batch_size = batch_residual.shape[0]
    scores = torch.empty(batch_size, device=DEV)
    
    for i in range(batch_size):
        input_residual = batch_residual[i, :, :]
        delta = input_residual.flatten() - fingerprint.flatten()
        scores[i] = -torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))


    return scores



def compute_mahalanobis_scores(residuals, fingerprints):

    #print(signals.shape)
    num_fingerprints = len(fingerprints)
    batch_size = residuals.shape[0]

    scores = torch.zeros((batch_size, num_fingerprints), device=DEV)
    fingerprints_list = sorted(fingerprints.keys())

    for fingerprint_index, fingerprint_name in enumerate(fingerprints_list):
        for data in fingerprints[fingerprint_name]:
            fingerprint = data["fingerprint"]
            invcov = data["invcov"]

            
            fingerprint_score = mahalanobis_score(fingerprint, residuals, invcov)
            
            scores[:, fingerprint_index] = fingerprint_score
    
    return scores


def assign_vocoders(scores):
    
    # Get the best vocoder index for each sample based on highest Mahalanobis score
    best_vocoder_indices = torch.argmax(scores, dim=1)
    
    # Convert predictions and labels to tensors
    preds_tensor = best_vocoder_indices.float()
    return preds_tensor


def calculate_metrics(preds_tensor, labels_tensor, output_dir, classification_type, num_classes):
    
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(DEV)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
    precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
    recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(DEV)
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(DEV)
    
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
        "Testing_Accuracy": [accuracy_score],
        "Testing_F1_Score": [f1_score],
        "Testing_Precision": [precision_score],
        "Testing_Recall": [recall_score]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(f'{output_dir}/testing_scores.xlsx', index=False)
    
    # Save confusion matrix to Excel file
    confusion_matrix_df = pd.DataFrame(confusion_matrix_score)
    confusion_matrix_df.to_excel(f'{output_dir}/testing_confusion_matrix.xlsx', index=True)
    
    save_heatmap(confusion_matrix_df.to_numpy(), results_path, classification_type)
    print(f'Metrics and confusion matrix saved in {output_dir}.')


# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run vocoder assignment with specified parameters.")
    parser.add_argument("--classification_type", type=str, required=True, choices=CLASSIFICATION_TYPES, help="Classification type (binary or multiclass).")
    parser.add_argument("--seed", type=int, default=40, help="Seed for reproducibility.")
    #parser.add_argument('--save_id', type=int, required=True, help='ID for saving the scores.')
    parser.add_argument('--proportional', action='store_true', default=False, help="Only for binary classification.")
    parser.add_argument('--performance_metric', type=str, choices=(PERFORMANCE_METRICS), default="f1_score", help='Performance metric.')

    args = parser.parse_args()
    set_seed(args.seed)

    if "binary" in args.classification_type:
        num_classes = 1
    else:
        num_classes = int(re.findall(pattern=r'\d+', string=args.classification_type)[0])
    model = "fingerprints"

    # Load fingerprints
    fingerprints = load_fingerprints(f'{FINGERPRINT_DIR}/{str(args.seed)}/low_pass_filter', CLASSES[args.classification_type])
    

    # Load dataset loaders
    train_ds, validate_ds, test_ds = get_datasets(
        "fingerprints", 
        args.classification_type, 
        args.seed, 
        "proportional" if args.proportional else "non-proportional")
    
    generator = torch.Generator().manual_seed(args.seed)
    
    # Score/Train
    if "multiclass" in args.classification_type:    # Fingerprints Scoring without any nn models

        # Set up results
        results_path = os.path.join(URL_DIR_TO_SAVE_MODELS_AND_LOGS, f'fingerprints/{args.classification_type}/{args.seed}')
        
        os.makedirs(results_path, exist_ok=True)

        all_preds = []
        all_labels = []


        print("Scoring initialized...")

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=fingerprints_collate_fn)

        for batch in tqdm(test_loader, desc="Processing test samples"):
            waveforms, labels, original_lens = batch
            waveforms, labels = waveforms.to(DEV), labels.to(DEV)
            residuals = waveform_to_residual(waveforms, original_lens)
            #residuals = residuals.squeeze(1)    # Remove dim 1

            scores = compute_mahalanobis_scores(residuals, fingerprints)
            #print(f'fingerprints: {fingerprints}')
            #print (f'scores shape: {scores.shape}')
            preds_tensor = assign_vocoders(scores)
            all_preds.append(preds_tensor)
            all_labels.append(labels)

        print("Scoring finished.")

        # Convert predictions and labels to tensors
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        calculate_metrics(preds_tensor, labels_tensor, results_path, args.classification_type, num_classes)

    else:   # Binary setting, Fingerprints scoring with nn models
        # if not args.use_residuals:  # Mahalanbis scores
        print(f'Initializing {model} model training...')

        # Set up results
        results_path = os.path.join(URL_DIR_TO_SAVE_MODELS_AND_LOGS, f'fingerprints/{args.classification_type}/{"proportional" if args.proportional else "non-proportional"}/{args.seed}')
        
        os.makedirs(results_path, exist_ok=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=True, pin_memory=True, generator=generator, shuffle=True, collate_fn=fingerprints_collate_fn)
        validation_loader = DataLoader(validate_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=fingerprints_collate_fn)
        
        # Set up wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=f'{model}-{args.classification_type}-{args.seed}',

            # track hyperparameters and run metadata
            config={
            "architecture": f'{model}',
            "classification_type": f'{args.classification_type}',
            "epochs": 100,
            },
            name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        )


        # Get model
        my_model = get_model(model=model, classification_type=args.classification_type)
        my_model = DataParallel(my_model, device_ids=DEVICE_IDS).to(DEV)

        # Get optimizer, scheduler and loss function
        optimizer, scheduler, loss_function = get_optimizer_scheduler_loss_function(model=model, my_model=my_model, classification_type=args.classification_type)

        # Get mean and std
        dir = os.path.join(MEAN_STD_FOLDER_DIR, "binary-10")
        dir = os.path.join(dir, "proportional" if args.proportional else "non-proportional")
        dir = os.path.join(dir, "residuals")
        dir = os.path.join(dir, str(args.seed))

        with open(os.path.join(dir, 'mean.pkl'), "rb") as f:
            mean = pickle.load(f)
            mean = torch.tensor(mean).to(DEV)
            #mean.to(DEV)
        with open(os.path.join(dir, 'std.pkl'), "rb") as f:
            std = pickle.load(f)
            std = torch.tensor(std).to(DEV)
            #std.to(DEV)

        # Set up metrics
        task = "binary"
        accuracy = Accuracy(task=task).to(DEV)
        f1 = F1Score(task=task).to(DEV)
        precision = Precision(task=task).to(DEV)
        recall = Recall(task=task).to(DEV)
        confusion_matrix = ConfusionMatrix(task=task).to(DEV)
        auroc = AUROC(task=task).to(DEV)

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


            for waveforms, labels, original_lens in tqdm(train_loader, desc="Training batches"):
                
                # print(f'Residual\'s batch shape: {residuals.shape}') # Debug
                labels = labels.float().unsqueeze(1)
                waveforms, labels = waveforms.to(DEV), labels.to(DEV)

                residuals = waveform_to_residual(waveforms, original_lens)

                # Normalize
                residuals = (residuals - mean[:, None]) / std[:, None]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                
                outputs = my_model(residuals)

                loss = loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Loss, predictions and probabilities
                running_loss += loss.item()
                probabilities = torch.nn.functional.sigmoid(outputs)
                preds = (probabilities > 0.5).float()

                # Accumulate metrics
                accuracy.update(preds, labels)
                f1.update(preds, labels)
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

            my_model.eval()
            validating_loss = 0.0
            with torch.no_grad():
                for waveforms, labels, original_lens in tqdm(validation_loader, desc="Validation batches"):

                    labels = labels.float().unsqueeze(1)
                    waveforms, labels = waveforms.to(DEV), labels.to(DEV)
                    residuals = waveform_to_residual(waveforms, original_lens)
                    
                    # Normalize
                    residuals = (residuals - mean[:, None]) / std[:, None]

                    # Forward pass
                    outputs = my_model(residuals)
                    loss = loss_function(outputs, labels)

                    # loss, predictions and probabilities
                    validating_loss += loss.item()
                    probabilities = torch.nn.functional.sigmoid(outputs)
                    preds = (probabilities > 0.5).float()

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
            metric = get_metric(args.performance_metric)
            if training_validating_scores_dict[metric] > best_score:
                best_score = training_validating_scores_dict[metric]
                print("\n\nNew best model found! Saving...")

                torch.save(my_model.state_dict(), f'{results_path}/best_model.pth')
                torch.save(scheduler.state_dict(), f'{results_path}/scheduler.pth')
                torch.save(optimizer.state_dict(), f'{results_path}/optimizer.pth')

            scheduler.step()

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
        my_model.load_state_dict(torch.load(f'{results_path}/best_model.pth'))
        my_model.eval()


        # Reset Metrics
        accuracy.reset(), f1.reset(), precision.reset()
        recall.reset(), confusion_matrix.reset(), auroc.reset()

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=24, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=fingerprints_collate_fn)

        with torch.no_grad():
            for waveforms, labels, original_lens in tqdm(test_loader, desc="Testing batches"):
                labels = labels.float().unsqueeze(1)
                waveforms, labels = waveforms.to(DEV), labels.to(DEV)

                residuals = waveform_to_residual(waveforms, original_lens)

                # Normalize
                residuals = (residuals - mean[:, None]) / std[:, None]
                
                # Forward pass
                outputs = my_model(residuals)

                probabilities = torch.nn.functional.sigmoid(outputs)
                preds = (probabilities > 0.5).float()

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
        training_validating_score_df.to_excel(f'{results_path}/training_validating_scores.xlsx', index=False)
        testing_score_df.to_excel(f'{results_path}/testing_scores.xlsx', index=False)
        save_confusion_matrix_to_excel(conf_matrix=testing_confusion_matrix, destination_url=results_path, classification_type=args.classification_type)
        save_heatmap(conf_matrix=testing_confusion_matrix.cpu().numpy(), destination_url=results_path, classification_type=args.classification_type)
        print("Scores saved...")
