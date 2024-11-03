import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict
from scipy.stats import chi2_contingency
from itertools import combinations
import json  # Import json to load mean and std, and class_to_idx mapping

class SevenNationsFERDataset(Dataset):
    """
    Custom Dataset for 7NationsFER Images organized as:
    7NationsFER/<Country>/<Expression>/<image_files>
    """
    def __init__(self, root_dir, transform=None, output_dir='.'):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transformations to apply to the images.
            output_dir (str): Directory to save the expressions per country CSV.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.output_dir = output_dir
        self.image_paths = []
        self.expression_labels = []
        self.country_labels = []
        self.class_to_idx = {}
        self.countries = []
        self.corrupted_images = []
        self.skipped_expressions = set()  # To track skipped expressions

        # Initialize the dataset
        self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Initializes the dataset by traversing the directory structure,
        collecting image paths, expression labels, and country labels.
        """
        # Load class_to_idx mapping from training
        class_to_idx_path = os.path.join(self.output_dir, 'class_to_idx.json')
        if os.path.exists(class_to_idx_path):
            with open(class_to_idx_path, 'r') as f:
                self.class_to_idx = json.load(f)
            print(f"Loaded class_to_idx mapping from '{class_to_idx_path}'.")
        else:
            # Define fixed class_to_idx mapping (ensure it matches training)
            self.class_to_idx = {
                'anger': 0,
                'happiness': 1,
                'surprise': 2,
                'neutral': 3
            }
            print("Using default class_to_idx mapping.")

        # Traverse through countries
        for country in os.listdir(self.root_dir):
            country_path = os.path.join(self.root_dir, country)
            if not os.path.isdir(country_path):
                continue
            self.countries.append(country)
            # Traverse through expressions
            for expression in os.listdir(country_path):
                expression_path = os.path.join(country_path, expression)
                if not os.path.isdir(expression_path):
                    continue
                # Only process known expressions
                expression_lower = expression.lower()
                if expression_lower not in self.class_to_idx:
                    print(f"Unknown expression '{expression}' in country '{country}'. Skipping.")
                    self.skipped_expressions.add(expression)
                    continue
                # Traverse through image files
                for img_name in os.listdir(expression_path):
                    img_path = os.path.join(expression_path, img_name)
                    try:
                        # Attempt to open the image to verify it's not corrupted
                        with Image.open(img_path) as img:
                            img.verify()  # Verify that it is, in fact, an image
                        # If successful, append to dataset
                        self.image_paths.append(img_path)
                        self.expression_labels.append(self.class_to_idx[expression_lower])
                        self.country_labels.append(country)
                    except (UnidentifiedImageError, IOError, SyntaxError) as e:
                        print(f"Error loading image: {img_path} - {e}")
                        self.corrupted_images.append(img_path)

        # After initializing the dataset
        print("\n[Dataset Initialization] Summary:")
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Total expressions processed: {len(self.expression_labels)}")
        print(f"Skipped Expressions: {self.skipped_expressions}")
        print(f"Number of Countries: {len(self.countries)}")

        # Compute and print the number of expressions per country
        self._compute_and_save_expressions_per_country()

    def _compute_and_save_expressions_per_country(self):
        """
        Computes the number of images per expression for each country,
        prints the counts, and saves them to a CSV file.
        """
        # Create a reverse mapping from label index to expression name
        index_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Initialize a nested dictionary to hold counts
        counts = defaultdict(lambda: defaultdict(int))

        # Populate the counts
        for country, expression_label in zip(self.country_labels, self.expression_labels):
            expression = index_to_class.get(expression_label, "Unknown")
            counts[country][expression] += 1

        # Print the counts
        print("\n[Expressions per Country]:")
        for country, expr_dict in counts.items():
            print(f"Country: {country}")
            for expression, count in expr_dict.items():
                print(f"  {expression}: {count} images")

        # Prepare data for CSV
        data = []
        for country, expr_dict in counts.items():
            for expression, count in expr_dict.items():
                data.append({
                    'Country': country,
                    'Expression': expression,
                    'Count': count
                })

        # Create a DataFrame and save to CSV
        df_counts = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, 'expressions_per_country_test_ResNet50-7NationsFER-AHNS.csv')
        try:
            df_counts.to_csv(csv_path, index=False)
            print(f"\n[Expressions per Country] CSV saved to '{csv_path}'.")
        except Exception as e:
            print(f"Error saving expressions per country CSV: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range in SevenNationsFERDataset")
        img_path = self.image_paths[idx]
        expression_label = self.expression_labels[idx]
        country_label = self.country_labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')  # Ensure 3-channel RGB
        except (UnidentifiedImageError, IOError, SyntaxError) as e:
            print(f"Error loading image during __getitem__: {img_path} - {e}")
            # Return a tensor of zeros and label - this can be handled or skipped
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, expression_label, country_label

def main():
    # ------------------------------
    # 0. Parse Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description='Test ResNet50 on 7NationsFER Dataset with AHNS classes')
    parser.add_argument('--model_path', type=str, default='ResNet50Fer+-AHNS-Train-Results/best_ResNet50_AHNS_model.pth',
                        help='Path to the trained model file')
    parser.add_argument('--test_data_path', type=str, default='7NationsFER',
                        help='Path to the test dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--project_path', type=str, default="/Users/home/Code/iFoRe2024-local-laughing-engine",
                        help='Path to the project directory')
    parser.add_argument('--output_dir', type=str, default='ResNet50-7NationsFER-AHNS-Test-Results',
                        help='Directory to save output reports and plots')
    args = parser.parse_args()

    model_path = args.model_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    PROJ_PATH = args.project_path
    results_dir = args.output_dir

    print(f"Model Path: {model_path}")
    print(f"Test Data Path: {test_data_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Project Path: {PROJ_PATH}")
    print(f"Results Directory: {results_dir}")
    print("PyTorch version:", torch.__version__)

    # ------------------------------
    # 1. Set Up Project Paths and Results Directory
    # ------------------------------
    print("\n[Step 1/16] Setting up project paths and results directory...")

    os.chdir(PROJ_PATH)

    # Construct absolute paths using PROJ_PATH
    model_path = os.path.join(PROJ_PATH, model_path)
    test_data_path = os.path.join(PROJ_PATH, test_data_path)
    results_dir = os.path.join(PROJ_PATH, results_dir)

    # Image size for ResNet-50
    image_size = 224  # ResNet-50 expects 224x224 images

    # Create Results Directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # ------------------------------
    # 2. Load Mean and Std from Training
    # ------------------------------
    print("\n[Step 2/16] Loading mean and standard deviation from training...")
    mean_std_path = os.path.join(PROJ_PATH, 'ResNet50Fer+-AHNS-Train-Results', 'mean_std.json')
    with open(mean_std_path, 'r') as f:
        mean_std = json.load(f)
    mean = mean_std['mean']
    std = mean_std['std']
    print(f"Loaded Mean: {mean}")
    print(f"Loaded Std: {std}")

    # ------------------------------
    # 3. Load class_to_idx Mapping from Training
    # ------------------------------
    print("\n[Step 3/16] Loading class_to_idx mapping from training...")
    class_to_idx_path = os.path.join(PROJ_PATH, 'ResNet50Fer+-AHNS-Train-Results', 'class_to_idx.json')
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    print(f"Loaded class_to_idx mapping: {class_to_idx}")

    # ------------------------------
    # 4. Data Loading with Custom Dataset
    # ------------------------------
    print("\n[Step 4/16] Setting up data transformations and loading test dataset...")

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load test dataset using custom Dataset
    print("Loading test dataset...")
    test_dataset = SevenNationsFERDataset(root_dir=test_data_path, transform=test_transforms, output_dir=results_dir)
    test_dataset.class_to_idx = class_to_idx  # Ensure consistent class_to_idx mapping
    print(f"Test dataset loaded with {len(test_dataset)} images.")
    print(f"Number of corrupted images skipped: {len(test_dataset.corrupted_images)}")
    if test_dataset.corrupted_images:
        corrupted_test_path = os.path.join(results_dir, 'corrupted_images_test_ResNet50-7NationsFER-AHNS.txt')
        with open(corrupted_test_path, 'w') as f:
            for img_path in test_dataset.corrupted_images:
                f.write(f"{img_path}\n")
        print(f"List of corrupted test images saved to '{corrupted_test_path}'.")
    else:
        print("No corrupted images were found in the test dataset.")

    # Store class names and number of classes
    class_names = list(test_dataset.class_to_idx.keys())
    num_classes = len(class_names)
    print("Number of classes (Expressions):", num_classes)
    print("Class names (Expressions):", class_names)

    # Check distribution of expressions in the test set
    expression_counts_test = defaultdict(int)
    for label in test_dataset.expression_labels:
        expression_counts_test[label] += 1

    print("\n[Expression Distribution in Test Set]:")
    for expr_idx, count in expression_counts_test.items():
        if expr_idx >= len(class_names):
            print(f"Invalid class index {expr_idx}. Skipping.")
            continue
        expr_name = class_names[expr_idx]
        print(f"{expr_name}: {count} images")

    # Log skipped expressions
    if test_dataset.skipped_expressions:
        skipped_expressions_path = os.path.join(results_dir, 'skipped_expressions_test_ResNet50-7NationsFER-AHNS.txt')
        with open(skipped_expressions_path, 'w') as f:
            for expr in test_dataset.skipped_expressions:
                f.write(f"{expr}\n")
        print(f"List of skipped expressions saved to '{skipped_expressions_path}'.")
    else:
        print("No expressions were skipped during test dataset initialization.")

    # Number of workers for data loading
    num_workers = 4  # Adjust based on your system

    # ------------------------------
    # 5. Create Data Loader
    # ------------------------------
    print("\n[Step 5/16] Creating data loader for test dataset...")
    data_loader_start_time = time.time()
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    data_loader_end_time = time.time()
    print(f"Test data loader created in {data_loader_end_time - data_loader_start_time:.2f} seconds.")

    # ------------------------------
    # 6. Set up Device
    # ------------------------------
    print("\n[Step 6/16] Setting up device...")
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
    # 7. Load the Trained Model
    # ------------------------------
    print("\n[Step 7/16] Loading the trained ResNet50-AHNS model...")
    try:
        # Initialize ResNet-50 model
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # Ensure the final layer matches the number of classes

        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Trained ResNet50-AHNS model loaded and set to evaluation mode.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # ------------------------------
    # 8. Initialize TensorBoard
    # ------------------------------
    print("\n[Step 8/16] Initializing TensorBoard writer for testing...")
    tensorboard_log_dir = os.path.join(results_dir, 'runs', 'ResNet50-7NationsFER-AHNS_test_experiment')
    writer = SummaryWriter(tensorboard_log_dir)
    print(f"TensorBoard writer initialized at '{tensorboard_log_dir}'.")

    # ------------------------------
    # 9. Perform Inference on Test Data
    # ------------------------------
    print("\n[Step 9/16] Performing inference on test data...")
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_countries = []

    with torch.no_grad():
        for inputs, expression_labels, country_labels in tqdm(test_loader, desc='Testing', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            expression_labels = expression_labels.to(device, non_blocking=True)

            outputs = model(inputs)
            logits = outputs  # ResNet50-AHNS returns raw logits
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_true_labels.extend(expression_labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())
            all_countries.extend(country_labels)  # Country labels are strings

    print("Inference completed.")

    # Convert lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_pred_probs = np.array(all_pred_probs)
    all_countries = np.array(all_countries)

    # ------------------------------
    # 10. Evaluate Model Performance
    # ------------------------------
    print("\n[Step 10/16] Evaluating model performance...")

    # Classification Report
    print("Generating classification report...")
    report = classification_report(all_true_labels, all_pred_labels, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Save Classification Report
    classification_report_path = os.path.join(results_dir, 'classification_report_test_ResNet50-7NationsFER-AHNS.txt')
    with open(classification_report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to '{classification_report_path}'.")

    # ------------------------------
    # 11. Generate Confusion Matrix
    # ------------------------------
    print("\n[Step 11/16] Generating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50-AHNS (Test Dataset)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix_test_ResNet50-7NationsFER-AHNS.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to '{confusion_matrix_path}'.")

    # ------------------------------
    # 12. Generate Stacked Bar Plot
    # ------------------------------
    print("\n[Step 12/16] Generating stacked bar plot for test dataset...")

    # Function to calculate correct and incorrect percentages
    def calculate_correct_incorrect(true_labels, pred_labels):
        total = len(true_labels)
        correct = np.sum(true_labels == pred_labels)
        incorrect = total - correct
        if total == 0:
            return 0, 0
        return (correct / total) * 100, (incorrect / total) * 100

    # Calculate percentages for test dataset
    correct_pct, incorrect_pct = calculate_correct_incorrect(all_true_labels, all_pred_labels)
    print(f"Test Dataset - Correct: {correct_pct:.2f}%, Incorrect: {incorrect_pct:.2f}%")

    # Plotting the stacked bar chart
    plt.figure(figsize=(8, 6))
    splits = ['Test']
    correct_pcts = [correct_pct]
    incorrect_pcts = [incorrect_pct]

    bar_width = 0.5
    plt.bar(splits, correct_pcts, bar_width, label='Correctly Classified', color='blue')
    plt.bar(splits, incorrect_pcts, bar_width, bottom=correct_pcts, label='Incorrectly Classified', color='orange')

    plt.ylabel('Percentage (%)')
    plt.xlabel('Dataset Split')
    plt.title('Classification Results - ResNet50-AHNS')
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    stacked_bar_path = os.path.join(results_dir, 'stacked_bar_test_ResNet50-7NationsFER-AHNS.png')
    plt.savefig(stacked_bar_path)
    plt.close()
    print(f"Stacked bar plot saved to '{stacked_bar_path}'.")

    # ------------------------------
    # 13. Log Metrics to TensorBoard
    # ------------------------------
    print("\n[Step 13/16] Logging metrics to TensorBoard...")
    # Log classification report as text
    writer.add_text('Classification Report/Test', report)
    # Optionally, log overall metrics or other statistics here

    # Close the TensorBoard writer
    writer.close()
    print("TensorBoard logging completed and writer closed.")

    # ------------------------------
    # 14. Generate and Save Additional Plots
    # ------------------------------
    print("\n[Step 14/16] Generating and saving additional plots for each country...")

    # Create a DataFrame for easier manipulation
    df_test = pd.DataFrame({
        'Country': all_countries,
        'True_Label': all_true_labels,
        'Pred_Label': all_pred_labels
    })

    # Map numerical labels to class names
    label_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    df_test['True_Class'] = df_test['True_Label'].map(label_to_class)
    df_test['Pred_Class'] = df_test['Pred_Label'].map(label_to_class)

    # Get list of unique countries
    unique_countries = df_test['Country'].unique()

    for country in unique_countries:
        print(f"\nProcessing country: {country}")
        df_country = df_test[df_test['Country'] == country]
        if df_country.empty:
            print(f"No data for country: {country}. Skipping.")
            continue

        # ------------------------------
        # 14.1. Generate Classification Report for the Country
        # ------------------------------
        print(f"Generating classification report for {country}...")
        report_country = classification_report(df_country['True_Label'], df_country['Pred_Label'],
                                               target_names=class_names, zero_division=0)
        print(f"\nClassification Report for {country}:")
        print(report_country)

        # Save Classification Report for the Country
        classification_report_country_path = os.path.join(results_dir, f'classification_report_{country}_ResNet50-7NationsFER-AHNS.txt')
        with open(classification_report_country_path, 'w') as f:
            f.write(report_country)
        print(f"Classification report for {country} saved to '{classification_report_country_path}'.")

        # ------------------------------
        # 14.2. Generate Confusion Matrix for the Country
        # ------------------------------
        print(f"Generating confusion matrix for {country}...")
        cm_country = confusion_matrix(df_country['True_Label'], df_country['Pred_Label'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_country, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - ResNet50-AHNS ({country})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        confusion_matrix_country_path = os.path.join(results_dir, f'confusion_matrix_{country}_ResNet50-7NationsFER-AHNS.png')
        plt.savefig(confusion_matrix_country_path)
        plt.close()
        print(f"Confusion matrix for {country} saved to '{confusion_matrix_country_path}'.")

        # ------------------------------
        # 14.3. Generate Normalized Percentage Stacked Bar Chart for the Country
        # ------------------------------
        print(f"Generating stacked bar chart for {country}...")
        # Calculate correct and incorrect counts per expression
        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)

        for t, p in zip(df_country['True_Label'], df_country['Pred_Label']):
            expr = label_to_class[t]
            if t == p:
                correct_counts[expr] += 1
            else:
                incorrect_counts[expr] += 1

        expressions = class_names
        correct_pcts = []
        incorrect_pcts = []

        for expr in expressions:
            total = correct_counts[expr] + incorrect_counts[expr]
            if total == 0:
                correct_pcts.append(0)
                incorrect_pcts.append(0)
            else:
                correct_pcts.append((correct_counts[expr] / total) * 100)
                incorrect_pcts.append((incorrect_counts[expr] / total) * 100)

        # Plotting the stacked bar chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(expressions))
        width = 0.6

        plt.bar(x, correct_pcts, width, label='Correctly Classified', color='blue')
        plt.bar(x, incorrect_pcts, width, bottom=correct_pcts, label='Incorrectly Classified', color='orange')

        plt.ylabel('Percentage (%)')
        plt.xlabel('Expressions')
        plt.title(f'Classification Results for {country} - ResNet50-AHNS')
        plt.xticks(x, expressions, rotation=45)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        stacked_bar_country_path = os.path.join(results_dir, f'stacked_bar_{country}_ResNet50-7NationsFER-AHNS.png')
        plt.savefig(stacked_bar_country_path)
        plt.close()
        print(f"Stacked bar chart for {country} saved to '{stacked_bar_country_path}'.")

    # ------------------------------
    # 15. Generate Per-Expression, Per-Country Stacked Bar Charts
    # ------------------------------
    print("\n[Step 15/16] Generating per-expression, per-country stacked bar charts...")

    # Get list of unique expressions and countries
    unique_expressions = class_names
    unique_countries = df_test['Country'].unique()

    # Initialize a dictionary to hold counts
    expression_country_counts = {expr: defaultdict(lambda: {'correct': 0, 'incorrect': 0}) for expr in unique_expressions}

    # Populate the counts
    for _, row in df_test.iterrows():
        expr = row['True_Class']
        country = row['Country']
        if row['True_Label'] == row['Pred_Label']:
            expression_country_counts[expr][country]['correct'] += 1
        else:
            expression_country_counts[expr][country]['incorrect'] += 1

    # Generate stacked bar charts for each expression
    for expr in unique_expressions:
        print(f"\nGenerating stacked bar chart for expression: {expr}")
        countries = []
        correct_pcts = []
        incorrect_pcts = []

        for country in unique_countries:
            counts = expression_country_counts[expr][country]
            total = counts['correct'] + counts['incorrect']
            if total == 0:
                countries.append(country)
                correct_pcts.append(0)
                incorrect_pcts.append(0)
            else:
                countries.append(country)
                correct_pcts.append((counts['correct'] / total) * 100)
                incorrect_pcts.append((counts['incorrect'] / total) * 100)

        # Plotting the stacked bar chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(countries))
        width = 0.6

        plt.bar(x, correct_pcts, width, label='Correctly Classified', color='blue')
        plt.bar(x, incorrect_pcts, width, bottom=correct_pcts, label='Incorrectly Classified', color='orange')

        plt.ylabel('Percentage (%)')
        plt.xlabel('Countries')
        plt.title(f'Classification Results for "{expr}" Expression - ResNet50-AHNS')
        plt.xticks(x, countries, rotation=45)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        stacked_bar_expr_path = os.path.join(results_dir, f'stacked_bar_{expr}_ResNet50-7NationsFER-AHNS.png')
        plt.savefig(stacked_bar_expr_path)
        plt.close()
        print(f"Stacked bar chart for expression '{expr}' saved to '{stacked_bar_expr_path}'.")

    # ------------------------------
    # 16. Perform Statistical Hypothesis Tests
    # ------------------------------
    print("\n[Step 16/16] Performing statistical hypothesis tests for differences between countries...")

    # Prepare data for statistical tests
    print("Preparing data for statistical tests...")
    stats_data = {}
    for country in unique_countries:
        df_country = df_test[df_test['Country'] == country]
        if df_country.empty:
            continue
        total_samples = len(df_country)
        correct_predictions = np.sum(df_country['True_Label'] == df_country['Pred_Label'])
        incorrect_predictions = total_samples - correct_predictions
        stats_data[country] = {
            'correct': correct_predictions,
            'incorrect': incorrect_predictions,
            'total': total_samples
        }

    # Perform chi-squared tests between each pair of countries
    print("Performing pairwise chi-squared tests between countries...")
    country_pairs = list(combinations(unique_countries, 2))
    test_results = []

    for country1, country2 in country_pairs:
        data = [
            [stats_data[country1]['correct'], stats_data[country1]['incorrect']],
            [stats_data[country2]['correct'], stats_data[country2]['incorrect']]
        ]
        chi2, p, dof, ex = chi2_contingency(data, correction=False)
        result = {
            'Country1': country1,
            'Country2': country2,
            'Chi2_statistic': chi2,
            'p_value': p
        }
        test_results.append(result)
        print(f"Chi-squared test between {country1} and {country2}: Chi2 = {chi2:.4f}, p-value = {p:.4e}")

    # Save test results to CSV
    df_test_results = pd.DataFrame(test_results)
    test_results_path = os.path.join(results_dir, 'chi_squared_test_results_between_countries.csv')
    df_test_results.to_csv(test_results_path, index=False)
    print(f"\nChi-squared test results saved to '{test_results_path}'.")

    # Interpretation note
    print("\nNote: A p-value less than 0.05 indicates a statistically significant difference in the proportions of correctly classified expressions between the two countries.")

    print("\nAll statistical hypothesis tests have been performed and results have been saved successfully.")

if __name__ == "__main__":
    main()