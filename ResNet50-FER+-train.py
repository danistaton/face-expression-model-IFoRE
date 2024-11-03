import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

from torchvision.datasets import ImageFolder

def compute_mean_std(dataset, batch_size=64, num_workers=4):
    """
    Compute the mean and standard deviation of a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute statistics on.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of subprocesses for data loading.
        
    Returns:
        tuple: (mean, std) each as a list of per-channel values.
    """
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
    return mean.tolist(), std.tolist()

def main():
    # ------------------------------
    # 0. Parse Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description='Train ResNet-50 on FER+ with Proper Data Handling')
    parser.add_argument('--subset_fraction', type=float, default=0.1,
                        help='Fraction of the dataset to use for training and validation (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation (default: 16)')
    parser.add_argument('--project_path', type=str, default="/Users/home/Code/iFoRe2024-local-laughing-engine",
                        help='Path to the project directory (default: "/Users/home/Code/iFoRe2024-local-laughing-engine")')
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
    print("\n[Step 1/18] Setting up project paths and results directory...")

    os.chdir(PROJ_PATH)

    # Image size for ResNet-50
    image_size = 224  # ResNet-50 expects 224x224 images

    # Paths to dataset
    base_path = os.path.join(PROJ_PATH, "FER+")
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'test')

    # Create Results Directory
    results_dir = os.path.join(PROJ_PATH, "ResNet50Fer+-Train-Results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # ------------------------------
    # 2. Define Class Mapping
    # ------------------------------
    print("\n[Step 2/18] Defining class-to-index mapping...")
    
    # Define a fixed class_to_idx mapping (all lowercase)
    class_to_idx = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happiness': 3,
        'neutral': 4,
        'surprise': 5,
        'sadness': 6
    }

    # ------------------------------
    # 3. Data Loading with PyTorch Datasets
    # ------------------------------
    print("\n[Step 3/18] Setting up data transformations and loading datasets...")

    # Instantiate ImageFolder without subclassing
    train_dataset = ImageFolder(
        root=train_dir,
        transform=None  # Placeholder, will set later
    )

    val_dataset = ImageFolder(
        root=val_dir,
        transform=None  # Placeholder, will set later
    )

    # Manually set class_to_idx
    train_dataset.class_to_idx = class_to_idx
    train_dataset.classes = list(class_to_idx.keys())

    val_dataset.class_to_idx = class_to_idx
    val_dataset.classes = list(class_to_idx.keys())

    # Recreate samples with the new class_to_idx
    train_dataset.imgs = train_dataset.make_dataset(train_dataset.root, train_dataset.class_to_idx, train_dataset.extensions)
    train_dataset.samples = train_dataset.imgs

    val_dataset.imgs = val_dataset.make_dataset(val_dataset.root, val_dataset.class_to_idx, val_dataset.extensions)
    val_dataset.samples = val_dataset.imgs

    # Verify datasets (optional)
    def verify_dataset(dataset, dataset_name='Dataset'):
        print(f"Verifying {dataset_name} integrity...")
        for idx in tqdm(range(len(dataset)), desc=f"Verifying {dataset_name}"):
            try:
                img, label = dataset[idx]
                if label == -1:
                    print(f"Warning: Sample at index {idx} has label -1.")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")

    print("Verifying training dataset...")
    verify_dataset(train_dataset, 'Training Dataset')
    print("Training dataset verification completed.")

    print("Verifying validation dataset...")
    verify_dataset(val_dataset, 'Validation Dataset')
    print("Validation dataset verification completed.")

    # Store class names and number of classes
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    print("Class names:", class_names)

    # Number of workers for data loading
    num_workers = 4  # Adjust based on your system

    # ------------------------------
    # 4. Create Subsets (For Testing)
    # ------------------------------
    print("\n[Step 4/18] Creating subsets for training and validation...")

    random_seed = 42

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)

    # Calculate the number of samples for subset
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    num_train_subset = int(num_train * subset_fraction)
    num_val_subset = int(num_val * subset_fraction)

    # Generate random indices
    train_indices = np.random.choice(num_train, num_train_subset, replace=False)
    val_indices = np.random.choice(num_val, num_val_subset, replace=False)

    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # ------------------------------
    # 5. Define Data Transforms
    # ------------------------------
    print("\n[Step 5/18] Defining data transformations...")

    # Initial transforms without normalization
    initial_train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1)
    ])

    initial_val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Assign initial transforms to subsets
    train_subset.dataset.transform = initial_train_transforms
    val_subset.dataset.transform = initial_val_transforms

    # ------------------------------
    # 6. Compute Mean and Std from Training Subset
    # ------------------------------
    print("\n[Step 6/18] Computing mean and standard deviation from training subset...")

    # Compute mean and std using the entire training subset
    mean, std = compute_mean_std(train_subset, batch_size=64, num_workers=4)

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    # ------------------------------
    # 7. Update Data Transforms with Normalization
    # ------------------------------
    print("\n[Step 7/18] Updating data transformations with normalization...")

    # Enhanced Data Augmentation for training with normalization
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Just resizing and normalization for validation
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Assign updated transforms to subsets
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms

    # ------------------------------
    # 8. Create Data Loaders
    # ------------------------------
    print("\n[Step 8/18] Creating data loaders for subsets...")
    data_loader_start_time = time.time()
    # Create data loaders using subset datasets
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
    data_loader_end_time = time.time()
    print(f"Data loaders created in {data_loader_end_time - data_loader_start_time:.2f} seconds.")

    # ------------------------------
    # 9. Set up Device
    # ------------------------------
    print("\n[Step 9/18] Setting up device...")
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend.")
    else:
        device = torch.device("cpu")
        print("Using CPU backend.")

    # ------------------------------
    # 10. Compute Class Weights
    # ------------------------------
    print("\n[Step 10/18] Computing class weights for balanced handling...")
    # Extract labels from the subset
    train_labels_subset = [train_dataset.targets[i] for i in train_indices]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_subset),
        y=train_labels_subset
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights computed and moved to device.")

    # ------------------------------
    # 11. Build the Model
    # ------------------------------
    print("\n[Step 11/18] Building the ResNet-50FER+ model...")
    model_building_start_time = time.time()

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the final layer to match FER+ classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optionally, unfreeze the last few layers for fine-tuning
    # For example, unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Move the model to the device
    model = model.to(device)

    # Adjust the optimizer to include the newly unfrozen parameters
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
    print("\n[Step 12/18] Setting up loss function with class weights...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Loss function setup completed.")

    # ------------------------------
    # 13. Initialize TensorBoard
    # ------------------------------
    print("\n[Step 13/18] Initializing TensorBoard writer...")
    tensorboard_log_dir = os.path.join(results_dir, 'runs', 'ResNet50FER+_experiment')
    writer = SummaryWriter(tensorboard_log_dir)
    print(f"TensorBoard writer initialized at '{tensorboard_log_dir}'.")

    # ------------------------------
    # 14. Train the Model
    # ------------------------------
    print("\n[Step 14/18] Starting training...")
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
                    logits = outputs  # ResNet-50FER+ returns raw logits
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                # Backward + optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward and backward
                outputs = model(inputs)
                logits = outputs  # ResNet-50FER+ returns raw logits
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
                logits = outputs  # ResNet-50FER+ returns raw logits
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
            best_model_path = os.path.join(results_dir, 'best_ResNet50FER+_model.pth')
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
    # 15. After Training: Save Metrics and Plots
    # ------------------------------
        print("\n[Step 15/18] Finalizing training and saving metrics...")

    total_training_end_time = time.time()
    print(f"\nTraining completed in {total_training_end_time - total_training_start_time:.2f}s")

    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Optionally, convert accuracies to percentages in the CSV
    metrics_df['Train Acc (%)'] = metrics_df['Train Acc'] * 100
    metrics_df['Val Acc (%)'] = metrics_df['Val Acc'] * 100
    metrics_df['Val ROC-AUC (%)'] = metrics_df['Val ROC-AUC'] * 100

    # Save training metrics to CSV with percentages
    metrics_csv_path = os.path.join(results_dir, 'training_metrics_ResNet50FER+.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved to '{metrics_csv_path}'.")

    # ------------------------------
    # 16. Save the Trained Model
    # ------------------------------
    print("\n[Step 16/18] Saving the final trained model...")
    model_save_start_time = time.time()
    final_model_path = os.path.join(results_dir, 'ResNet50FER+_model_final.pth')
    torch.save(model.state_dict(), final_model_path)  # Saved as final model
    model_save_end_time = time.time()
    print(f"Model saved to '{final_model_path}' in {model_save_end_time - model_save_start_time:.2f} seconds.")

    # Optionally, save the optimizer state
    optimizer_state_path = os.path.join(results_dir, 'optimizer_state_ResNet50FER+.pth')
    torch.save(optimizer.state_dict(), optimizer_state_path)
    print(f"Optimizer state saved to '{optimizer_state_path}'.")

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        torch.cuda.empty_cache()

    print("\nScript completed successfully.")

    # ------------------------------
    # 17. Post-Training Evaluation (Optional)
    # ------------------------------
    print("\n[Step 17/18] Generating classification reports, confusion matrices, and stacked bar plots for both Train and Validation data...")

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
    # Note: Validation data has already been evaluated during training, but we'll re-evaluate to ensure consistency
    # Alternatively, you can reuse the collected validation predictions if desired
    all_val_labels_final, all_val_preds_final, all_val_probs_final = evaluate_model(model, val_loader, device, num_classes, dataset_name='Validation Dataset')
    print("Validation Data Evaluation Completed.")

    # Classification Report for Training Data
    print("Generating Classification Report for Training Data...")
    train_report = classification_report(all_train_labels, all_train_preds, target_names=class_names, output_dict=True)
    # Convert the report to percentages
    train_report_percentage = {}
    for key, value in train_report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            train_report_percentage[key] = {
                'precision': round(value['precision'] * 100, 2),
                'recall': round(value['recall'] * 100, 2),
                'f1-score': round(value['f1-score'] * 100, 2),
                'support': value['support']
            }
        elif key in ['macro avg', 'weighted avg']:
            train_report_percentage[key] = {
                'precision': round(value['precision'] * 100, 2),
                'recall': round(value['recall'] * 100, 2),
                'f1-score': round(value['f1-score'] * 100, 2),
                'support': value['support']
            }
        else:  # 'accuracy'
            train_report_percentage[key] = round(value * 100, 2)

    # Convert the percentage report to a DataFrame for better formatting
    train_report_df = pd.DataFrame(train_report_percentage).transpose()
    # Format the report with percentages
    train_report_df[['precision', 'recall', 'f1-score']] = train_report_df[['precision', 'recall', 'f1-score']].astype(str) + '%'

    # Save Classification Report as Image for Training Data
    def save_classification_report(report_df, filename='classification_report_train_ResNet50FER+.png', save_dir=results_dir):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Create a table
        table = ax.table(cellText=report_df.values,
                         rowLabels=report_df.index,
                         colLabels=report_df.columns,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        fig.tight_layout()
        report_path = os.path.join(save_dir, filename)
        fig.savefig(report_path, dpi=300)
        plt.close(fig)
        print(f"Training classification report saved as '{report_path}'.")

    save_classification_report(train_report_df, filename='classification_report_train_ResNet50FER+.png', save_dir=results_dir)

    # Classification Report for Validation Data
    print("Generating Classification Report for Validation Data...")
    val_report = classification_report(all_val_labels_final, all_val_preds_final, target_names=class_names, output_dict=True)
    # Convert the report to percentages
    val_report_percentage = {}
    for key, value in val_report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            val_report_percentage[key] = {
                'precision': round(value['precision'] * 100, 2),
                'recall': round(value['recall'] * 100, 2),
                'f1-score': round(value['f1-score'] * 100, 2),
                'support': value['support']
            }
        elif key in ['macro avg', 'weighted avg']:
            val_report_percentage[key] = {
                'precision': round(value['precision'] * 100, 2),
                'recall': round(value['recall'] * 100, 2),
                'f1-score': round(value['f1-score'] * 100, 2),
                'support': value['support']
            }
        else:  # 'accuracy'
            val_report_percentage[key] = round(value * 100, 2)

    # Convert the percentage report to a DataFrame for better formatting
    val_report_df = pd.DataFrame(val_report_percentage).transpose()
    # Format the report with percentages
    val_report_df[['precision', 'recall', 'f1-score']] = val_report_df[['precision', 'recall', 'f1-score']].astype(str) + '%'

    # Save Classification Report as Image for Validation Data
    save_classification_report(val_report_df, filename='classification_report_val_ResNet50FER+.png', save_dir=results_dir)

    # Confusion Matrix for Training Data
    print("Generating Confusion Matrix for Training Data...")
    train_cm = confusion_matrix(all_train_labels, all_train_preds, normalize='true') * 100  # Normalize per true class
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50FER+ (Training Data, Normalized)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    train_confusion_matrix_path = os.path.join(results_dir, 'normalized_confusion_matrix_train_ResNet50FER+.png')
    plt.savefig(train_confusion_matrix_path)
    plt.close()
    print(f"Training confusion matrix saved as '{train_confusion_matrix_path}'.")

    # Confusion Matrix for Validation Data
    print("Generating Confusion Matrix for Validation Data...")
    val_cm = confusion_matrix(all_val_labels_final, all_val_preds_final, normalize='true') * 100  # Normalize per true class
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50FER+ (Validation Data, Normalized)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    val_confusion_matrix_path = os.path.join(results_dir, 'normalized_confusion_matrix_val_ResNet50FER+.png')
    plt.savefig(val_confusion_matrix_path)
    plt.close()
    print(f"Validation confusion matrix saved as '{val_confusion_matrix_path}'.")

    # Function to plot stacked bar chart
    def plot_stacked_bar(correct, incorrect, classes, dataset_name='Dataset', filename='stacked_bar_plot.png', save_dir=results_dir):
        plt.figure(figsize=(12, 8))
        x = np.arange(len(classes))
        width = 0.6

        plt.bar(x, correct, width, label='Correctly Classified', color='bluee')
        plt.bar(x, incorrect, width, bottom=correct, label='Incorrectly Classified', color='orange')

        plt.ylabel('Percentage (%)')
        plt.xlabel('Expressions')
        plt.title(f'Classification Results - {dataset_name}')
        plt.xticks(x, classes, rotation=45)
        plt.ylim(0, 100)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Stacked bar plot for {dataset_name} saved as '{plot_path}'.")

    # Calculate correct and incorrect percentages for Training Data
    print("Generating Stacked Bar Plot for Training Data...")
    train_correct = []
    train_incorrect = []
    train_cm_counts = confusion_matrix(all_train_labels, all_train_preds)
    for i in range(num_classes):
        correct = (train_cm_counts[i, i] / train_cm_counts[i].sum()) * 100 if train_cm_counts[i].sum() > 0 else 0
        incorrect = 100 - correct
        train_correct.append(correct)
        train_incorrect.append(incorrect)

    plot_stacked_bar(train_correct, train_incorrect, class_names, dataset_name='Training Data', filename='stacked_bar_train_ResNet50FER+.png', save_dir=results_dir)

    # Calculate correct and incorrect percentages for Validation Data
    print("Generating Stacked Bar Plot for Validation Data...")
    val_correct = []
    val_incorrect = []
    val_cm_counts = confusion_matrix(all_val_labels_final, all_val_preds_final)
    for i in range(num_classes):
        correct = (val_cm_counts[i, i] / val_cm_counts[i].sum()) * 100 if val_cm_counts[i].sum() > 0 else 0
        incorrect = 100 - correct
        val_correct.append(correct)
        val_incorrect.append(incorrect)

    plot_stacked_bar(val_correct, val_incorrect, class_names, dataset_name='Validation Data', filename='stacked_bar_val_ResNet50FER+.png', save_dir=results_dir)

    # ROC-AUC Curve for Validation Data (Optional, already implemented)
    print("Generating ROC-AUC Curves...")
    def plot_roc_curves(y_true, y_probs, classes, filename='roc_auc_curves_ResNet50FER+.png', save_dir=results_dir):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np

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
        plt.title('Multi-class ROC Curves - ResNet50FER+')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_auc_path = os.path.join(save_dir, filename)
        plt.savefig(roc_auc_path)
        plt.close()
        print(f"ROC-AUC curves saved as '{roc_auc_path}'.")

    plot_roc_curves(all_val_labels_final, all_val_probs_final, class_names, filename='roc_auc_curves_ResNet50FER+.png', save_dir=results_dir)

    # ROC-AUC Over Epochs (Optional, already implemented)
    print("Generating Metrics Over Epochs Plot...")
    def plot_metrics(metrics_df, filename='metrics_over_epochs_ResNet50FER+.png', save_dir=results_dir):
        plt.figure(figsize=(12, 8))

        # Plot Loss
        plt.subplot(3, 1, 1)
        plt.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Train Loss')
        plt.plot(metrics_df['Epoch'], metrics_df['Val Loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs - ResNet50FER+')
        plt.legend()

        # Plot Accuracy
        plt.subplot(3, 1, 2)
        plt.plot(metrics_df['Epoch'], metrics_df['Train Acc (%)'], label='Train Acc')
        plt.plot(metrics_df['Epoch'], metrics_df['Val Acc (%)'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over Epochs - ResNet50FER+')
        plt.legend()

        # Plot ROC-AUC
        plt.subplot(3, 1, 3)
        plt.plot(metrics_df['Epoch'], metrics_df['Val ROC-AUC (%)'], label='Val ROC-AUC')
        plt.xlabel('Epoch')
        plt.ylabel('ROC-AUC (%)')
        plt.title('Validation ROC-AUC over Epochs - ResNet50FER+')
        plt.legend()

        plt.tight_layout()
        metrics_plot_path = os.path.join(save_dir, filename)
        plt.savefig(metrics_plot_path)
        plt.close()
        print(f"Metrics over epochs plot saved as '{metrics_plot_path}'.")

    plot_metrics(metrics_df, filename='metrics_over_epochs_ResNet50FER+.png', save_dir=results_dir)

    # Close the TensorBoard writer
    writer.close()
    print("TensorBoard writer closed.")

if __name__ == "__main__":
    main()