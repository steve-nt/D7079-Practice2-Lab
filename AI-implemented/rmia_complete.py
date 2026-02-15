"""
PART I: Complete RMIA Implementation
Robust Membership Inference Attack against ResNet-18 on CIFAR-10

This script implements the RMIA attack from the paper:
"Membership Inference Attacks From First Principles" (arXiv:2112.03570)

The attack determines if a specific data sample was used in training a model
by comparing likelihood ratios between the target model and reference models.
"""

# Import necessary libraries
import numpy as np  # For numerical operations
import torch  # PyTorch framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torchvision  # Computer vision datasets and models
import torchvision.transforms as transforms  # Data preprocessing
from torchvision.models import resnet18  # ResNet-18 architecture
from torch.utils.data import DataLoader, Subset  # Data loading utilities
from sklearn.metrics import roc_curve, auc, roc_auc_score  # Evaluation metrics
import matplotlib.pyplot as plt  # Plotting library
import pickle  # For saving Python objects
import os  # Operating system interface

# Device configuration - use GPU if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_model(dataloader, epochs=10, model_name="model"):
    """
    Train a ResNet-18 model on CIFAR-10
    
    Args:
        dataloader: PyTorch DataLoader containing training data
        epochs: Number of complete passes through the dataset
        model_name: Name for logging purposes
    
    Returns:
        Trained PyTorch model
    """
    # Initialize ResNet-18 with 10 output classes (for CIFAR-10)
    model = resnet18(num_classes=10).to(device)
    
    # Cross-entropy loss - standard for classification tasks
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer - adaptive learning rate optimization
    # lr=0.001 is a common default learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()
    
    # Loop over the dataset multiple times
    for epoch in range(epochs):
        running_loss = 0.0  # Track cumulative loss
        correct = 0  # Count correct predictions
        total = 0  # Count total samples processed
        
        # Iterate through batches of data
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to GPU/CPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients (PyTorch accumulates gradients by default)
            optimizer.zero_grad()
            
            # Forward pass: compute model predictions
            outputs = model(inputs)
            
            # Calculate loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters based on gradients
            optimizer.step()
            
            # Statistics tracking
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class (highest probability)
            total += labels.size(0)  # Count samples in this batch
            correct += (predicted == labels).sum().item()  # Count correct predictions
            
            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                # Print progress statistics
                print(f'{model_name} - Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%')
                running_loss = 0.0  # Reset running loss for next 100 batches
        
        # Calculate and print accuracy for the entire epoch
        epoch_acc = 100 * correct / total
        print(f'{model_name} - Epoch [{epoch+1}/{epochs}] completed. Accuracy: {epoch_acc:.2f}%')
    
    # Return the trained model
    return model


def get_rmia_score_multi(target_model, ref_models, known_img, known_label, 
                         population_subset, gamma=1.0, a=0.3):
    """
    Calculate RMIA score using multiple reference models
    
    RMIA Concept:
    - Compares how much more likely the target model assigns to a sample vs reference models
    - Uses pairwise likelihood ratios: LR(x,z) = [Pr(x|θ) / Pr(x)] / [Pr(z|θ) / Pr(z)]
    - Score = fraction of population samples that x "dominates"
    
    Args:
        target_model: The model being attacked (trained on members)
        ref_models: List of reference models (trained on different data)
        known_img: Image to test for membership
        known_label: True label of the image
        population_subset: List of (img, label) tuples used as baseline comparison
        gamma: Threshold for determining if x dominates z (default: 1.0)
        a: Offline scaling parameter to approximate Pr(x) (default: 0.3)
    
    Returns:
        RMIA score between 0 and 1 (higher = more likely a member)
    """
    # Set all models to evaluation mode (disables dropout, batch norm uses running stats)
    target_model.eval()
    for rm in ref_models:
        rm.eval()
    
    # Disable gradient computation (saves memory and speeds up inference)
    with torch.no_grad():
        # Move image to GPU/CPU
        known_img = known_img.to(device)
        
        # Step 1: Get probability that target model assigns to the correct class
        # unsqueeze(0) adds batch dimension, softmax converts logits to probabilities
        prob_x_target = torch.softmax(target_model(known_img.unsqueeze(0)), dim=1)[0, known_label].item()
        
        # Step 2: Average predictions across all reference models to estimate Pr(x)_OUT
        # This approximates the probability distribution of models NOT trained on x
        all_ref_probs_x = []
        for rm in ref_models:
            prob = torch.softmax(rm(known_img.unsqueeze(0)), dim=1)[0, known_label].item()
            all_ref_probs_x.append(prob)
        prob_x_out = np.mean(all_ref_probs_x)  # Average over all reference models
        
        # Step 3: Offline scaling approximation (from RMIA paper Equation 5)
        # Interpolates between OUT probability and uniform distribution
        # a=0 gives uniform, a=1 gives pure OUT estimate
        pr_x = 0.5 * ((1 + a) * prob_x_out + (1 - a))
        # Add epsilon (1e-10) to avoid division by zero
        ratio_x = prob_x_target / (pr_x + 1e-10)
        
        # Step 4: Count how many population samples x "dominates"
        # x dominates z if LR(x,z) >= gamma
        count_dominated = 0
        for z_img, z_label in population_subset:
            z_img = z_img.to(device)
            
            # Get target model probability for population sample z
            prob_z_target = torch.softmax(target_model(z_img.unsqueeze(0)), dim=1)[0, z_label].item()
            
            # Average reference model predictions for z
            all_ref_probs_z = []
            for rm in ref_models:
                prob = torch.softmax(rm(z_img.unsqueeze(0)), dim=1)[0, z_label].item()
                all_ref_probs_z.append(prob)
            prob_z_out = np.mean(all_ref_probs_z)
            
            # Apply same offline scaling to z
            pr_z = 0.5 * ((1 + a) * prob_z_out + (1 - a))
            ratio_z = prob_z_target / (pr_z + 1e-10)
            
            # Check if x dominates z (likelihood ratio comparison)
            # If true, x is more "member-like" than z
            if (ratio_x / (ratio_z + 1e-10)) >= gamma:
                count_dominated += 1
        
        # Return score: proportion of population dominated by x
        # Score close to 1 = likely member, close to 0 = likely non-member
        return count_dominated / len(population_subset)


def evaluate_attack(target_model, ref_models, members, non_members, 
                    population_data, num_eval=500, population_size=1000):
    """
    Evaluate RMIA attack performance on members and non-members
    
    Computes ROC curve and AUC to measure attack effectiveness
    
    Args:
        target_model: Model being attacked
        ref_models: List of reference models
        members: Dataset of training samples (members)
        non_members: Dataset of non-training samples (non-members)
        population_data: Population dataset for baseline
        num_eval: Number of samples to evaluate from each set
        population_size: Size of population subset to use
    
    Returns:
        Dictionary with scores, labels, fpr, tpr, and auc
    """
    print(f"\nEvaluating attack with {len(ref_models)} reference models...")
    print(f"Testing on {num_eval} members and {num_eval} non-members")
    
    # Lists to store all scores and corresponding labels
    all_scores = []  # RMIA scores for each sample
    all_labels = []  # 1 for members, 0 for non-members
    
    # Sample a subset of population data for baseline comparison in RMIA
    # This serves as the "z" samples in the likelihood ratio comparisons
    population_subset = [population_data[i] for i in range(min(population_size, len(population_data)))]
    
    # Evaluate attack on member samples (should get high scores)
    print("Testing members...")
    for i in range(min(num_eval, len(members))):
        img, label = members[i]  # Get image and its true label
        # Calculate RMIA score for this member
        score = get_rmia_score_multi(target_model, ref_models, img, label, population_subset)
        all_scores.append(score)
        all_labels.append(1)  # Label 1 indicates this is a member
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_eval} members")
    
    # Evaluate attack on non-member samples (should get low scores)
    print("Testing non-members...")
    for i in range(min(num_eval, len(non_members))):
        img, label = non_members[i]  # Get image and its true label
        # Calculate RMIA score for this non-member
        score = get_rmia_score_multi(target_model, ref_models, img, label, population_subset)
        all_scores.append(score)
        all_labels.append(0)  # Label 0 indicates this is a non-member
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_eval} non-members")
    
    # Calculate ROC curve: plots True Positive Rate vs False Positive Rate
    # at different threshold values for the RMIA score
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Calculate AUC (Area Under Curve) - single number measuring attack effectiveness
    # AUC = 0.5 means random guessing, AUC = 1.0 means perfect attack
    roc_auc = auc(fpr, tpr)
    
    # Print summary statistics
    print(f"\nResults:")
    print(f"  AUC: {roc_auc:.4f}")
    # TPR at low FPR is important: shows how many members we catch with few false alarms
    print(f"  TPR at 1% FPR: {tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0:.4f}")
    print(f"  TPR at 0.1% FPR: {tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0:.4f}")
    
    # Return all results in a dictionary for later analysis
    return {
        'scores': all_scores,  # List of RMIA scores
        'labels': all_labels,  # List of ground truth labels
        'fpr': fpr,  # False positive rates for ROC curve
        'tpr': tpr,  # True positive rates for ROC curve
        'auc': roc_auc,  # Area under ROC curve
        'thresholds': thresholds  # Score thresholds corresponding to FPR/TPR
    }


def plot_roc_curves(results_dict, save_path='roc_comparison.png'):
    """
    Plot ROC curves for different attack configurations
    
    ROC (Receiver Operating Characteristic) curve shows the trade-off between
    True Positive Rate (correctly identified members) and False Positive Rate
    (non-members incorrectly identified as members) at different thresholds
    
    Args:
        results_dict: Dictionary mapping configuration names to result dictionaries
        save_path: Filename to save the plot
    """
    # Create a large figure for better visibility
    plt.figure(figsize=(10, 8))
    
    # Plot one ROC curve for each configuration
    for name, results in results_dict.items():
        plt.plot(results['fpr'], results['tpr'], 
                label=f'{name} (AUC = {results["auc"]:.4f})')
    
    # Plot diagonal line representing random guessing (AUC = 0.5)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add labels and formatting
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - RMIA Attack Performance')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_score_distribution(results, save_path='score_distribution.png'):
    """
    Plot histogram of RMIA scores for members vs non-members
    
    Good separation between distributions indicates effective attack
    
    Args:
        results: Dictionary containing scores and labels
        save_path: Filename to save the plot
    """
    # Convert to numpy arrays for easier manipulation
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    
    # Separate scores by membership status
    member_scores = scores[labels == 1]  # Scores for training samples
    non_member_scores = scores[labels == 0]  # Scores for non-training samples
    
    # Create histogram plot
    plt.figure(figsize=(10, 6))
    
    # Plot overlapping histograms with transparency
    # density=True normalizes to show probability density
    plt.hist(member_scores, bins=30, alpha=0.6, color='blue', label='Members', density=True)
    plt.hist(non_member_scores, bins=30, alpha=0.6, color='orange', label='Non-Members', density=True)
    
    # Add labels and formatting
    plt.xlabel('RMIA Score')
    plt.ylabel('Density')
    plt.title('Distribution of Membership Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Score distribution saved to {save_path}")


def create_imbalanced_dataset(dataset, imbalance_type='none'):
    """
    Create dataset with class imbalance to test TASK 1.2 Question 3
    
    Args:
        dataset: Original dataset
        imbalance_type: 'none', 'mild' (50% from classes 0-2), or 'severe' (80% from classes 0-4)
    """
    if imbalance_type == 'none':
        return dataset
    
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        
        if imbalance_type == 'mild':
            # Remove 50% from classes 0-2
            if label <= 2 and np.random.random() < 0.5:
                continue
            indices.append(i)
        elif imbalance_type == 'severe':
            # Remove 80% from classes 0-4
            if label <= 4 and np.random.random() < 0.8:
                continue
            indices.append(i)
    
    return Subset(dataset, indices)


def main():
    """Main execution function"""
    
    # Configuration
    TRAIN_EPOCHS = 10  # Increase to 50-100 for better results
    NUM_REF_MODELS = [1, 2, 4, 8]  # Test with different numbers
    NUM_EVAL_SAMPLES = 500  # Number of samples to evaluate
    POPULATION_SIZE = 1000
    
    # Data loading
    print("=" * 80)
    print("PART I: ROBUST MEMBERSHIP INFERENCE ATTACK (RMIA)")
    print("=" * 80)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("\nLoading CIFAR-10 dataset...")
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                  download=True, transform=transform)
    
    # Split dataset: 20k members, 20k non-members, 10k population
    target_train, target_test, population_data, _ = torch.utils.data.random_split(
        full_trainset, [20000, 20000, 10000, 0]
    )
    
    # Create dataloaders
    trainloader = DataLoader(target_train, batch_size=64, shuffle=True, num_workers=2)
    
    # Train target model
    print("\n" + "=" * 80)
    print("Training Target Model...")
    print("=" * 80)
    target_model = train_model(trainloader, epochs=TRAIN_EPOCHS, model_name="Target")
    
    # Save target model
    torch.save(target_model.state_dict(), 'target_model.pth')
    print("Target model saved to 'target_model.pth'")
    
    # Store results for different numbers of reference models
    all_results = {}
    
    for num_refs in NUM_REF_MODELS:
        print("\n" + "=" * 80)
        print(f"Training {num_refs} Reference Model(s)...")
        print("=" * 80)
        
        ref_models = []
        for i in range(num_refs):
            # Create separate population dataloader for each reference model
            pop_indices = np.random.choice(len(population_data), 10000, replace=False)
            pop_subset = Subset(population_data, pop_indices)
            pop_loader = DataLoader(pop_subset, batch_size=64, shuffle=True, num_workers=2)
            
            print(f"\nTraining Reference Model {i+1}/{num_refs}...")
            ref_model = train_model(pop_loader, epochs=TRAIN_EPOCHS, 
                                   model_name=f"Reference-{i+1}")
            ref_models.append(ref_model)
            
            # Save reference model
            torch.save(ref_model.state_dict(), f'ref_model_{i+1}_of_{num_refs}.pth')
        
        # Evaluate attack
        results = evaluate_attack(target_model, ref_models, target_train, target_test,
                                 population_data, num_eval=NUM_EVAL_SAMPLES,
                                 population_size=POPULATION_SIZE)
        
        all_results[f'{num_refs} Ref Model(s)'] = results
        
        # Save results
        with open(f'results_{num_refs}_refs.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    # Plot comparisons
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    
    plot_roc_curves(all_results, 'roc_comparison_all.png')
    
    # Plot score distribution for best configuration
    best_config = max(all_results.items(), key=lambda x: x[1]['auc'])
    plot_score_distribution(best_config[1], f'score_dist_{best_config[0]}.png')
    
    # === TASK 1.2 EXPERIMENTS ===
    
    # Question 3: Class Imbalance
    print("\n" + "=" * 80)
    print("TASK 1.2 - Question 3: Testing with Class Imbalance")
    print("=" * 80)
    
    imbalance_results = {}
    
    for imbalance_type in ['none', 'mild', 'severe']:
        print(f"\nTesting with {imbalance_type} imbalance...")
        
        # Create imbalanced dataset
        imbalanced_train = create_imbalanced_dataset(target_train, imbalance_type)
        imbalanced_loader = DataLoader(imbalanced_train, batch_size=64, shuffle=True, num_workers=2)
        
        # Train model on imbalanced data
        imb_model = train_model(imbalanced_loader, epochs=TRAIN_EPOCHS, 
                               model_name=f"Imbalanced-{imbalance_type}")
        
        # Use 4 reference models for this test
        num_refs = 4
        ref_models = []
        for i in range(num_refs):
            pop_indices = np.random.choice(len(population_data), 10000, replace=False)
            pop_subset = Subset(population_data, pop_indices)
            pop_loader = DataLoader(pop_subset, batch_size=64, shuffle=True, num_workers=2)
            ref_model = train_model(pop_loader, epochs=TRAIN_EPOCHS, 
                                   model_name=f"Ref-Imb-{i+1}")
            ref_models.append(ref_model)
        
        # Evaluate
        results = evaluate_attack(imb_model, ref_models, imbalanced_train, target_test,
                                 population_data, num_eval=NUM_EVAL_SAMPLES,
                                 population_size=POPULATION_SIZE)
        
        imbalance_results[f'Imbalance: {imbalance_type}'] = results
    
    # Plot imbalance comparison
    plot_roc_curves(imbalance_results, 'roc_imbalance_comparison.png')
    
    # Final summary
    print("\n" + "=" * 80)
    print("PART I COMPLETE - Summary")
    print("=" * 80)
    print("\nResults by number of reference models:")
    for name, results in all_results.items():
        print(f"  {name}: AUC = {results['auc']:.4f}")
    
    print("\nResults with class imbalance:")
    for name, results in imbalance_results.items():
        print(f"  {name}: AUC = {results['auc']:.4f}")
    
    print("\nAll models, results, and visualizations have been saved.")
    print("=" * 80)


if __name__ == '__main__':
    main()
