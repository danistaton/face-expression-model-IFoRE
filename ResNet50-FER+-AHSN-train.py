import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import json  # Added for saving class_to_idx mapping

from torchvision.datasets import ImageFolder

def compute_mean_std(dataset, batch_size=64, num_workers=4, transform=None):
    """
    Compute the mean and standard deviation of a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute statistics on.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of subprocesses for data loading.
        transform (callable, optional): A function/transform to apply to the samples.
            
    Returns:
        tuple: (mean, std) each as a list of per-channel values.
    """
    # Temporarily set the dataset's transform to the provided transform
    original_transform = dataset.transform
    dataset.transform = transform

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    total_samples = 0
    for images, _ in tqdm(loader, desc="Computing Mean and Std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    mean /= total_samples
    std /= total_samples

    # Restore the original transform
    dataset.transform = original_transform

    return mean.tolist(), std.tolist()

def verify_dataset_labels(dataset, num_classes, dataset_name='Dataset'):
    """
    Verify that all labels in the dataset are within the expected range.

    Args:
        dataset (torch.utils.data.Dataset): The dataset subset to verify.
        num_classes (int): The number of expected classes.
        dataset_name (str): Name of the dataset subset for logging.

    Raises:
        ValueError: If any label is outside the expected range.
    """
    print(f"\nVerifying {dataset_name} labels...")
    labels = []
    for _, label in tqdm(dataset, desc=f"Collecting labels for {dataset_name}"):
        labels.append(label)
    unique_labels = set(labels)
    print(f"Unique labels in {dataset_name}: {unique_labels}")
    if not unique_labels.issubset(set(range(num_classes))):
        print(f"Error: {dataset_name} contains labels outside the range 0-{num_classes - 1}.")
        raise ValueError(f"{dataset_name} has invalid labels.")
    else:
        print(f"All labels in {dataset_name} are within the expected range 0-{num_classes - 1}.")

class RemappedSubsetWithTransform(Dataset):
    """
    A wrapper around Subset to remap labels and apply transforms.
    """
    def __init__(self, subset, original_to_new_map, transform=None):
        """
        Initialize the RemappedSubsetWithTransform.

        Args:
            subset (torch.utils.data.Subset): The original subset.
            original_to_new_map (dict): Mapping from original labels to new labels.
            transform (callable, optional): A function/transform to apply to the samples.
        """
        self.subset = subset
        self.original_to_new_map = original_to_new_map
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]
        if label not in self.original_to_new_map:
            raise ValueError(f"Label {label} not in original_to_new_map.")
        new_label = self.original_to_new_map[label]
        if self.transform:
            data = self.transform(data)
        return data, new_label

def main():
    # ------------------------------
    # 0. Parse Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description='Train ResNet50 on FER+ Dataset with AHNS classes')
    parser.add_argument('--subset_fraction', type=float, default=1.0,
                        help='Fraction of the dataset to use for training and validation (default: 1.0)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation (default: 16)')
    parser.add_argument('--project_path', type=str, default="/Users/home/Code/iFoRe2024-local-laughing-engine",
                        help='Path to the project directory')
    args = parser.parse_args()

    subset_fraction = args.subset_fraction
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    PROJ_PATH = args.project_path

    print(f"Using subset fraction: {subset_fraction}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Project path: {PROJ_PATH}")

    print("PyTorch version:", torch.__version__)
    # Assuming torchvision was successfully installed
    try:
        import torchvision
        print("torchvision version:", torchvision.__version__)
    except ImportError:
        print("torchvision is not installed properly.")

    # ------------------------------
    # 1. Set Up Project Paths and Results Directory
    # ------------------------------
    print("\n[Step 1/19] Setting up project paths and results directory...")

    os.chdir(PROJ_PATH)

    # Image size for ResNet-50
    image_size = 224  # ResNet-50 expects 224x224 images

    # Paths to dataset
    base_path = os.path.join(PROJ_PATH, "FER+")
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'test')

    # Create Results Directory
    results_dir = os.path.join(PROJ_PATH, "ResNet50Fer+-AHNS-Train-Results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # ------------------------------
    # 2. Define Desired Classes and Class Mapping
    # ------------------------------
    print("\n[Step 2/19] Defining desired classes and class mapping...")

    # Correct class_to_idx mapping
    class_to_idx = {
        'anger': 0,
        'happiness': 1,
        'surprise': 2,
        'neutral': 3
    }

    desired_classes = list(class_to_idx.keys())
    num_classes = len(desired_classes)
    print(f"Number of classes: {num_classes}")
    print(f"Desired classes: {desired_classes}")
    print(f"class_to_idx mapping: {class_to_idx}")

    # ------------------------------
    # 3. Data Loading with PyTorch Datasets
    # ------------------------------
    print("\n[Step 3/19] Setting up data transformations and loading datasets...")

    # Do not apply any transforms here; set transform=None
    full_train_dataset = ImageFolder(
        root=train_dir,
        transform=None
    )

    full_val_dataset = ImageFolder(
        root=val_dir,
        transform=None
    )

    # ------------------------------
    # 4. Filter Datasets to Include Only Desired Classes and Remap Labels
    # ------------------------------
    print("\n[Step 4/19] Filtering datasets to include only desired classes and remapping labels...")

    def filter_dataset(full_dataset, desired_classes, class_to_idx):
        # Get the indices of the desired classes based on the dataset's class_to_idx
        desired_class_indices = [full_dataset.class_to_idx[cls_name] for cls_name in desired_classes if cls_name in full_dataset.class_to_idx]
        print(f"Desired class indices in dataset: {desired_class_indices}")

        # Map original dataset class indices to new indices using class_to_idx
        original_to_new_map = {full_dataset.class_to_idx[cls_name]: class_to_idx[cls_name] for cls_name in desired_classes}
        print(f"Original to new label mapping: {original_to_new_map}")

        # Filter the dataset samples
        filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in desired_class_indices]
        print(f"Number of samples after filtering: {len(filtered_indices)}")

        # Create a subset of the dataset
        subset = Subset(full_dataset, filtered_indices)

        return subset, original_to_new_map

    # Filter training and validation datasets
    train_subset, train_original_to_new_map = filter_dataset(full_train_dataset, desired_classes, class_to_idx)
    val_subset, val_original_to_new_map = filter_dataset(full_val_dataset, desired_classes, class_to_idx)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # Wrap the subsets with RemappedSubsetWithTransform using the correct mapping
    train_subset = RemappedSubsetWithTransform(train_subset, train_original_to_new_map, transform=None)
    val_subset = RemappedSubsetWithTransform(val_subset, val_original_to_new_map, transform=None)

    print("Label remapping completed.")

    # ------------------------------
    # 5. Save class_to_idx mapping
    # ------------------------------
    print("\n[Step 5/19] Saving class_to_idx mapping...")
    class_to_idx_path = os.path.join(results_dir, 'class_to_idx.json')
    with open(class_to_idx_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f"class_to_idx mapping saved to '{class_to_idx_path}'.")

    # ------------------------------
    # 6. Define Data Transforms with Normalization
    # ------------------------------
    print("\n[Step 6/19] Defining data transformations with normalization...")

    # Define a simple transform to convert images to tensors
    to_tensor_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Compute mean and std from training subset
    print("\n[Step 7/19] Computing mean and standard deviation from training subset...")
    mean, std = compute_mean_std(train_subset, batch_size=64, num_workers=4, transform=to_tensor_transform)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    # Save Mean and Std
    print("\n[Step 8/19] Saving mean and standard deviation...")
    mean_std = {'mean': mean, 'std': std}
    mean_std_path = os.path.join(results_dir, 'mean_std.json')
    with open(mean_std_path, 'w') as f:
        json.dump(mean_std, f)
    print(f"Mean and standard deviation saved to '{mean_std_path}'.")

    # Update transforms to include normalization
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Apply the updated transforms to the subsets
    print("Applying normalization transforms to subsets...")
    train_subset.transform = train_transforms
    val_subset.transform = val_transforms
    print("Transforms applied successfully.")

    # ------------------------------
    # 7. Verify Dataset Integrity
    # ------------------------------
    print("\n[Step 8/19] Verifying dataset integrity after filtering and remapping...")

    print("Verifying training subset labels...")
    verify_dataset_labels(train_subset, num_classes, 'Training Subset')
    print("Training subset verification passed.")

    print("Verifying validation subset labels...")
    verify_dataset_labels(val_subset, num_classes, 'Validation Subset')
    print("Validation subset verification passed.")

    # ------------------------------
    # 8. Create Data Loaders
    # ------------------------------
    print("\n[Step 9/19] Creating data loaders for subsets...")
    num_workers = 4  # Adjust based on your system

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Data loaders created successfully.")

    # ------------------------------
    # 9. Set up Device
    # ------------------------------
    print("\n[Step 10/19] Setting up device...")
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend.")
    else:
        device = torch.device("cpu")
        print("Using CPU backend.")

    # ------------------------------
    # 10. Compute Class Weights
    # ------------------------------
    print("\n[Step 11/19] Computing class weights for balanced handling...")
    # Extract labels from the training subset
    train_labels_subset = [label for _, label in train_subset]

    # Compute class weights based on the desired classes
    classes = np.arange(num_classes)  # [0, 1, 2, 3]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels_subset
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights computed: {class_weights}")
    print("Class weights moved to device.")

    # ------------------------------
    # 11. Build the Model
    # ------------------------------
    print("\n[Step 12/19] Building the ResNet50 model...")
    model_building_start_time = time.time()

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the final layer to match the number of desired classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optionally, unfreeze the last few layers for fine-tuning (e.g., layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Move the model to the device
    model = model.to(device)

    # Define the optimizer to include only the parameters that require gradients
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5  # Added weight decay for regularization
    )

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
        patience=2,
        verbose=True
    )

    model_building_end_time = time.time()
    print(f"Model built in {model_building_end_time - model_building_start_time:.2f} seconds.")

    # ------------------------------
    # 12. Define Loss Function
    # ------------------------------
    print("\n[Step 13/19] Setting up loss function with class weights...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Loss function setup completed.")

    # ------------------------------
    # 13. Initialize TensorBoard
    # ------------------------------
    print("\n[Step 14/19] Initializing TensorBoard writer...")
    tensorboard_log_dir = os.path.join(results_dir, 'runs', 'ResNet50_AHNS_experiment')
    writer = SummaryWriter(tensorboard_log_dir)
    print(f"TensorBoard writer initialized at '{tensorboard_log_dir}'.")

    # ------------------------------
    # 14. Train the Model
    # ------------------------------
    print("\n[Step 15/19] Starting training...")
    # num_epochs is set via command-line arguments

    # Initialize lists to store metrics for visualization
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_aucs = []

    # Early Stopping parameters
    best_val_auc = 0.0
    patience = 3
    trigger_times = 0

    # Initialize lists to collect labels and predictions for metrics
    all_val_labels = []
    all_val_preds = []
    all_val_probs = []

    # Initialize mixed precision scaler (optional, for performance)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Initialize list to store metrics for CSV logging
    metrics_list = []

    # Initialize training start time
    total_training_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0  # To track total samples processed

        # Initialize tqdm progress bar for training
        train_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision (if scaler is available)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    logits = outputs  # ResNet50 returns raw logits
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                # Backward + optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward and backward
                outputs = model(inputs)
                logits = outputs  # ResNet50 returns raw logits
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)

                # Backward + optimize
                loss.backward()
                optimizer.step()

            # Statistics
            batch_size_current = inputs.size(0)
            running_loss += loss.item() * batch_size_current
            running_corrects += torch.sum(preds == labels.data)
            processed_samples += batch_size_current

            # Calculate running average loss and accuracy
            avg_loss = running_loss / processed_samples
            avg_acc = running_corrects.float() / processed_samples

            # Update tqdm progress bar with running averages
            train_bar.set_postfix({'Avg Loss': f"{avg_loss:.4f}", 'Avg Acc': f"{avg_acc:.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_processed_samples = 0  # To track total validation samples processed

        all_val_labels_epoch = []
        all_val_preds_epoch = []
        all_val_probs_epoch = []

        # Initialize tqdm progress bar for validation
        val_bar = tqdm(val_loader, desc='Validation', leave=False)
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs  # ResNet50 returns raw logits
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)

            batch_size_current = inputs.size(0)
            val_running_loss += loss.item() * batch_size_current
            val_running_corrects += torch.sum(preds == labels.data)
            val_processed_samples += batch_size_current

            # Collect labels and predictions for metrics
            all_val_labels_epoch.extend(labels.cpu().numpy())
            all_val_preds_epoch.extend(preds.cpu().numpy())
            all_val_probs_epoch.extend(probs.cpu().numpy())

            # Calculate running average loss and accuracy
            val_avg_loss = val_running_loss / val_processed_samples
            val_avg_acc = val_running_corrects.float() / val_processed_samples

            # Update tqdm progress bar with running averages
            val_bar.set_postfix({'Avg Loss': f"{val_avg_loss:.4f}", 'Avg Acc': f"{val_avg_acc:.4f}"})

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.float() / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        # Aggregate for ROC-AUC
        all_val_labels.extend(all_val_labels_epoch)
        all_val_preds.extend(all_val_preds_epoch)
        all_val_probs.extend(all_val_probs_epoch)

        epoch_time = time.time() - epoch_start_time

        # Compute ROC-AUC for this epoch
        try:
            from sklearn.preprocessing import label_binarize
            y_true_binarized = label_binarize(all_val_labels_epoch, classes=range(num_classes))
            y_pred_probs = np.array(all_val_probs_epoch)
            val_auc = roc_auc_score(y_true_binarized, y_pred_probs, average='macro', multi_class='ovr')
        except ValueError:
            val_auc = 0.0  # Handle cases where ROC-AUC cannot be computed
            print("  ROC-AUC could not be computed for this epoch due to insufficient class representation.")

        val_aucs.append(val_auc)

        # Log metrics to list for CSV
        metrics_list.append({
            'Epoch': epoch + 1,
            'Train Loss': epoch_loss,
            'Train Acc': epoch_acc.item(),
            'Val Loss': val_loss,
            'Val Acc': val_acc.item(),
            'Val ROC-AUC': val_auc
        })

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc.item(), epoch)
        writer.add_scalar('Accuracy/Validation', val_acc.item(), epoch)
        writer.add_scalar('ROC_AUC/Validation', val_auc, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Time: {epoch_time:.2f}s, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val ROC-AUC: {val_auc:.4f}')

        # Scheduler step based on validation ROC-AUC
        scheduler.step(val_auc)

        # Early Stopping check based on validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_path = os.path.join(results_dir, 'best_ResNet50_AHNS_model.pth')
            torch.save(model.state_dict(), best_model_path)  # Save the best model
            trigger_times = 0
            print(f"  Validation ROC-AUC improved. Saving model to '{best_model_path}'.")
        else:
            trigger_times += 1
            print(f"  No improvement in validation ROC-AUC for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        # ------------------------------
        # 15. Evaluation After Each Epoch
        # ------------------------------
        print(f"\n[Step 16/19] Epoch {epoch+1}: Generating evaluation metrics and plots...")

        # Function to evaluate the model on a given dataloader
        def evaluate_model(model, dataloader, device, num_classes, dataset_name='Dataset'):
            model.eval()
            all_labels = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                eval_bar = tqdm(dataloader, desc=f'Evaluating {dataset_name}', leave=False)
                for inputs, labels in eval_bar:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(inputs)
                    logits = outputs  # Raw logits
                    probs = torch.softmax(logits, dim=1)
                    _, preds_batch = torch.max(logits, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds_batch.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            return all_labels, all_preds, all_probs

        # Evaluate on Training Data
        print("Evaluating on Training Data...")
        all_train_labels, all_train_preds, all_train_probs = evaluate_model(model, train_loader, device, num_classes, dataset_name='Training Dataset')
        print("Training Data Evaluation Completed.")

        # Evaluate on Validation Data
        print("Evaluating on Validation Data...")
        all_val_labels_final, all_val_preds_final, all_val_probs_final = evaluate_model(model, val_loader, device, num_classes, dataset_name='Validation Dataset')
        print("Validation Data Evaluation Completed.")

        # Generate and save reports and plots for this epoch
        epoch_results_dir = os.path.join(results_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_results_dir, exist_ok=True)

        # ------------------------------
        # 15.1. Generate Classification Reports
        # ------------------------------
        print("Generating Classification Report for Training Data...")
        train_report = classification_report(all_train_labels, all_train_preds, target_names=desired_classes, output_dict=True)
        train_report_df = pd.DataFrame(train_report).transpose()
        train_report_path = os.path.join(epoch_results_dir, f'classification_report_train_epoch_{epoch+1}.csv')
        train_report_df.to_csv(train_report_path)
        print(f"Training classification report saved to '{train_report_path}'.")

        print("Generating Classification Report for Validation Data...")
        val_report = classification_report(all_val_labels_final, all_val_preds_final, target_names=desired_classes, output_dict=True)
        val_report_df = pd.DataFrame(val_report).transpose()
        val_report_path = os.path.join(epoch_results_dir, f'classification_report_val_epoch_{epoch+1}.csv')
        val_report_df.to_csv(val_report_path)
        print(f"Validation classification report saved to '{val_report_path}'.")

        # ------------------------------
        # 15.2. Generate Confusion Matrices
        # ------------------------------
        print("Generating Confusion Matrix for Training Data...")
        train_cm = confusion_matrix(all_train_labels, all_train_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=desired_classes, yticklabels=desired_classes)
        plt.title(f'Confusion Matrix - ResNet50_AHNS (Training Data, Epoch {epoch+1})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        train_cm_path = os.path.join(epoch_results_dir, f'confusion_matrix_train_epoch_{epoch+1}.png')
        plt.savefig(train_cm_path)
        plt.close()
        print(f"Training confusion matrix saved to '{train_cm_path}'.")

        print("Generating Confusion Matrix for Validation Data...")
        val_cm = confusion_matrix(all_val_labels_final, all_val_preds_final)
        plt.figure(figsize=(10, 8))
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=desired_classes, yticklabels=desired_classes)
        plt.title(f'Confusion Matrix - ResNet50_AHNS (Validation Data, Epoch {epoch+1})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        val_cm_path = os.path.join(epoch_results_dir, f'confusion_matrix_val_epoch_{epoch+1}.png')
        plt.savefig(val_cm_path)
        plt.close()
        print(f"Validation confusion matrix saved to '{val_cm_path}'.")

        # ------------------------------
        # 15.3. Generate ROC-AUC Curves
        # ------------------------------
        print("Generating ROC-AUC Curves...")
        def plot_roc_curves(y_true, y_probs, classes, filename='roc_auc_curves.png', save_dir=epoch_results_dir):
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc

            # Binarize the true labels
            y_true_binarized = label_binarize(y_true, classes=range(len(classes)))

            # Convert y_probs to a numpy array if it's not already
            y_probs = np.array(y_probs)

            # Check if y_probs has the correct shape
            if y_probs.shape[0] != y_true_binarized.shape[0]:
                raise ValueError("Number of samples in y_true and y_probs do not match.")

            plt.figure(figsize=(10, 8))
            for i in range(len(classes)):
                # Handle cases where a class might not be present in y_true
                if np.sum(y_true_binarized[:, i]) == 0:
                    print(f"Warning: No samples found for class '{classes[i]}'. Skipping ROC curve for this class.")
                    continue
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multi-class ROC Curves - ResNet50_AHNS (Epoch {epoch+1})')
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_auc_path = os.path.join(save_dir, filename)
            plt.savefig(roc_auc_path)
            plt.close()
            print(f"ROC-AUC curves saved as '{roc_auc_path}'.")

        plot_roc_curves(all_val_labels_final, all_val_probs_final, desired_classes, filename=f'roc_auc_curves_epoch_{epoch+1}.png', save_dir=epoch_results_dir)

        # ------------------------------
        # 15.4. Generate Metrics Over Epochs Plot
        # ------------------------------
        print("Generating Metrics Over Epochs Plot...")
        def plot_metrics(metrics_df, filename='metrics_over_epochs.png', save_dir=epoch_results_dir):
            plt.figure(figsize=(12, 18))

            # Plot Loss
            plt.subplot(3, 1, 1)
            plt.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Train Loss', marker='o')
            plt.plot(metrics_df['Epoch'], metrics_df['Val Loss'], label='Val Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs - ResNet50_AHNS')
            plt.legend()

            # Plot Accuracy
            plt.subplot(3, 1, 2)
            plt.plot(metrics_df['Epoch'], metrics_df['Train Acc (%)'], label='Train Acc (%)', marker='o')
            plt.plot(metrics_df['Epoch'], metrics_df['Val Acc (%)'], label='Val Acc (%)', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy over Epochs - ResNet50_AHNS')
            plt.legend()

            # Plot ROC-AUC
            plt.subplot(3, 1, 3)
            plt.plot(metrics_df['Epoch'], metrics_df['Val ROC-AUC (%)'], label='Val ROC-AUC (%)', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('ROC-AUC (%)')
            plt.title('Validation ROC-AUC over Epochs - ResNet50_AHNS')
            plt.legend()

            plt.tight_layout()
            metrics_plot_path = os.path.join(save_dir, filename)
            plt.savefig(metrics_plot_path)
            plt.close()
            print(f"Metrics over epochs plot saved as '{metrics_plot_path}'.")

        # Convert metrics list to DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df['Train Acc (%)'] = metrics_df['Train Acc'] * 100
        metrics_df['Val Acc (%)'] = metrics_df['Val Acc'] * 100
        metrics_df['Val ROC-AUC (%)'] = metrics_df['Val ROC-AUC'] * 100

        plot_metrics(metrics_df, filename=f'metrics_over_epochs_epoch_{epoch+1}.png', save_dir=epoch_results_dir)

    # ------------------------------
    # 16. After Training: Save Metrics
    # ------------------------------
    print("\n[Step 17/19] Finalizing training and saving metrics...")

    total_training_end_time = time.time()
    print(f"\nTraining completed in {total_training_end_time - total_training_start_time:.2f}s")

    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Optionally, convert accuracies to percentages in the CSV
    metrics_df['Train Acc (%)'] = metrics_df['Train Acc'] * 100
    metrics_df['Val Acc (%)'] = metrics_df['Val Acc'] * 100
    metrics_df['Val ROC-AUC (%)'] = metrics_df['Val ROC-AUC'] * 100

    # Save training metrics to CSV with percentages
    metrics_csv_path = os.path.join(results_dir, 'training_metrics_ResNet50_AHNS.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved to '{metrics_csv_path}'.")

    if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        torch.cuda.empty_cache()

    print("\nScript completed successfully.")

    # ------------------------------
    # 17. Close TensorBoard Writer
    # ------------------------------
    writer.close()
    print("TensorBoard writer closed.")

if __name__ == "__main__":
    main()