import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import amp
from sklearn.metrics import (
    roc_curve, confusion_matrix, cohen_kappa_score, accuracy_score,
    roc_auc_score, precision_score, recall_score, balanced_accuracy_score,
    f1_score
)
from sklearn import metrics
from NewTechnology.SynPredTest.utils import MyDataset, save_AUCs, collate
from models.model_GAT_dual_attention import MolecularInteractionModel

# Set random seeds for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(model, device, train_loader, optimizer, epoch, scaler):
    """Train the model for one epoch."""
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    start_memory = torch.cuda.memory_allocated(device)
    
    for batch_idx, (left_molecule, right_molecule, labels) in enumerate(train_loader):
        # Move data to device
        left_molecule = left_molecule.to(device)
        right_molecule = right_molecule.to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with torch.amp.autocast("cuda", enabled=True):
            predictions = model(left_molecule, right_molecule)
            loss = nn.CrossEntropyLoss()(predictions, labels)
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Log training progress
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(left_molecule.x)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Log memory usage
    end_memory = torch.cuda.memory_allocated(device)
    memory_usage_mb = (end_memory - start_memory) / (1024 ** 2)
    print(f'Training memory usage for epoch {epoch}: {memory_usage_mb:.2f} MB')

def evaluate_model(model, device, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    all_predictions = torch.Tensor()
    all_labels = torch.Tensor()
    all_predicted_labels = torch.Tensor()
    
    print(f'Evaluating on {len(test_loader.dataset)} samples...')
    with torch.no_grad():
        for left_molecule, right_molecule, labels in test_loader:
            # Move data to device
            left_molecule = left_molecule.to(device)
            right_molecule = right_molecule.to(device)
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda', enabled=True):
                predictions = model(left_molecule, right_molecule)
            
            # Process predictions
            probabilities = F.softmax(predictions, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), probabilities))
            prediction_scores = list(map(lambda x: x[1], probabilities))
            
            # Concatenate results
            all_predictions = torch.cat((all_predictions, torch.Tensor(prediction_scores)), 0)
            all_predicted_labels = torch.cat((all_predicted_labels, torch.Tensor(predicted_labels)), 0)
            all_labels = torch.cat((all_labels, labels.view(-1, 1)), 0)
    
    return all_labels.numpy().flatten(), all_predictions.numpy().flatten(), all_predicted_labels.numpy().flatten()

def calculate_metrics(true_labels, prediction_scores, predicted_labels):
    """Calculate all evaluation metrics."""
    metrics_dict = {
        'AUC': roc_auc_score(true_labels, prediction_scores),
        'PR_AUC': metrics.auc(*metrics.precision_recall_curve(true_labels, prediction_scores)[:2]),
        'BACC': balanced_accuracy_score(true_labels, predicted_labels),
        'ACC': accuracy_score(true_labels, predicted_labels),
        'KAPPA': cohen_kappa_score(true_labels, predicted_labels),
        'RECALL': recall_score(true_labels, predicted_labels),
        'PRECISION': precision_score(true_labels, predicted_labels),
        'F1': f1_score(true_labels, predicted_labels)
    }
    
    # Calculate TPR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    metrics_dict['TPR'] = tp / (tp + fn)
    
    return metrics_dict

def main():
    # Hyperparameters
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LEARNING_RATE = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 120
    NUM_FOLDS = 5
    
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'Epochs: {NUM_EPOCHS}')
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = MyDataset()
    dataset_size = len(dataset)
    fold_size = int(dataset_size / NUM_FOLDS)
    print(f'Dataset size: {dataset_size}')
    print(f'Fold size: {fold_size}')
    
    # Random shuffle indices
    indices = random.sample(range(dataset_size), dataset_size)
    
    # Cross-validation
    for fold in range(NUM_FOLDS):
        print(f'\n=== Starting Fold {fold + 1}/{NUM_FOLDS} ===')
        
        # Split data
        test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]
        
        # Create data loaders
        train_data = dataset.get_data(train_indices)
        test_data = dataset.get_data(test_indices)
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
        
        # Initialize model and training components
        model = MolecularInteractionModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler(enabled=True)
        
        # Setup result file
        result_file = f'../result/MolecularInteraction-lr{LEARNING_RATE:.4f}-fold{fold}.csv'
        with open(result_file, 'w') as f:
            f.write('Epoch,ACC,PR_AUC,AUC,BACC,PREC,TPR,KAPPA,RECALL,Precision,F1\n')
        
        # Training loop
        best_accuracy = 0
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Train and evaluate
            train_model(model, device, train_loader, optimizer, epoch + 1, scaler)
            true_labels, prediction_scores, predicted_labels = evaluate_model(model, device, test_loader)
            
            # Calculate metrics
            metrics_dict = calculate_metrics(true_labels, prediction_scores, predicted_labels)
            
            # Save results if improved
            if metrics_dict['ACC'] > best_accuracy:
                best_accuracy = metrics_dict['ACC']
                metrics_list = [
                    epoch, metrics_dict['ACC'], metrics_dict['PR_AUC'],
                    metrics_dict['AUC'], metrics_dict['BACC'], metrics_dict['PRECISION'],
                    metrics_dict['TPR'], metrics_dict['KAPPA'], metrics_dict['RECALL'],
                    metrics_dict['PRECISION'], metrics_dict['F1']
                ]
                save_AUCs(metrics_list, result_file)
            
            # Log epoch completion
            epoch_duration = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')
            print(f'Best accuracy: {best_accuracy:.4f}')

if __name__ == '__main__':
    main()