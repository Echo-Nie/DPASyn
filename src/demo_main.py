import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.model.demo_model import DemoDualStreamModel
from src.utils.util import demo_collate, DemoDataset

# Set random seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Hyperparameter search space
HYPERPARAM_SPACE = {
    'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
    'batch_size': [128, 256, 512],
    'num_epochs': [80, 80, 80, 80],
    'dropout_rate': [0.1, 0.3, 0.5]
}


def generate_random_hyperparams():
    """Generate a random set of hyperparameters from the predefined search space."""
    return {
        'lr': random.choice(HYPERPARAM_SPACE['learning_rate']),
        'batch_size': random.choice(HYPERPARAM_SPACE['batch_size']),
        'num_epochs': random.choice(HYPERPARAM_SPACE['num_epochs']),
        'dropout': random.choice(HYPERPARAM_SPACE['dropout_rate'])
    }


def train_one_epoch(model, device, data_loader, optimizer, scaler):
    """Train the model for one epoch."""
    model.train()
    for data1, data2, y in data_loader:
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            output = model(data1, data2)
            loss = nn.CrossEntropyLoss()(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def evaluate_model(model, device, data_loader):
    """Evaluate the model and return true labels and predicted probabilities."""
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for data1, data2, y in data_loader:
            data1, data2 = data1.to(device), data2.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                output = model(data1, data2)
            preds = torch.softmax(output, dim=1).cpu().numpy()
            total_preds.extend(preds[:, 1])
            total_labels.extend(y.cpu().numpy())

    return np.array(total_labels), np.array(total_preds)


def train_and_validate_with_params(params, device, dataset, num_folds=5):
    """
    Train and validate using k-fold cross-validation.
    Returns average accuracy across all folds.
    """
    fold_accuracies = []
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds

    for fold in range(num_folds):
        # Split indices into train and test
        indices = list(range(dataset_size))
        test_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = [i for i in indices if i not in test_indices]

        # Get data subsets
        train_data = dataset.get_data(train_indices)
        test_data = dataset.get_data(test_indices)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=params['batch_size'],
                                  shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_data, batch_size=params['batch_size'],
                                 shuffle=False, collate_fn=collate)

        # Build model and optimizer
        model = AttenSyn(dropout_rate=params['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        scaler = torch.amp.GradScaler(enabled=True)

        best_acc = 0.0

        # Training loop
        for epoch in range(params['num_epochs']):
            train_one_epoch(model, device, train_loader, optimizer, scaler)
            true_labels, pred_probs = evaluate_model(model, device, test_loader)
            predicted_labels = (pred_probs > 0.5).astype(int)
            current_acc = accuracy_score(true_labels, predicted_labels)

            if current_acc > best_acc:
                best_acc = current_acc

        fold_accuracies.append(best_acc)

    return np.mean(fold_accuracies)


def save_search_records(records, filename='hyperparam_search.log'):
    """Save hyperparameter search records to a log file."""
    with open(filename, 'w') as f:
        for record in records:
            line = f"Score: {record['score']:.4f} | Params: {record['params']} | Time: {record['time']:.1f}s\n"
            f.write(line)


if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = MyDataset()

    # Initialize tracking variables
    best_params = None
    best_score = 0.0
    search_records = []

    # Perform 50 iterations of random hyperparameter search
    for iteration in range(50):
        print(f"\n=== Hyperparameter Search Iteration {iteration + 1}/50 ===")

        # Generate new parameters
        current_params = generate_random_hyperparams()
        print("Current Parameters:", current_params)

        # Train and validate
        start_time = time.time()
        avg_accuracy = train_and_validate_with_params(current_params, device, dataset)
        duration = time.time() - start_time

        # Record results
        search_records.append({
            'params': current_params,
            'score': avg_accuracy,
            'time': duration
        })

        # Update best parameters
        if avg_accuracy > best_score:
            best_score = avg_accuracy
            best_params = current_params
            print(f"New Best Parameters Found! Accuracy: {avg_accuracy:.4f}")

        # Print current best
        print(f"Current Best Accuracy: {best_score:.4f}")
        print(f"Current Best Parameters: {best_params}")

    # Final output
    print("\n=== Hyperparameter Search Completed ===")
    print(f"Best Validation Accuracy: {best_score:.4f}")
    print("Best Parameter Combination:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # Save logs
    save_search_records(search_records)
