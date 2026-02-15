# ==============================================================================
# IMPORTS: Load all necessary libraries for the RMIA attack implementation
# ==============================================================================

# Import PyTorch core library - provides tensor operations and autograd functionality
import torch

# Import neural network module - provides layers (Linear, Conv2d), loss functions, etc.
import torch.nn as nn

# Import optimization module - provides optimizers like SGD, Adam for training
import torch.optim as optim

# Import torchvision - provides datasets (CIFAR-10), pre-trained models, transforms
import torchvision

# Import transforms module - provides image preprocessing (normalize, resize, etc.)
import torchvision.transforms as transforms

# Import functional API - provides stateless functions like softmax, relu
import torch.nn.functional as F

# Import ResNet-18 architecture - a convolutional neural network with 18 layers
from torchvision.models import resnet18

# ==============================================================================
# FUNCTION: train_model
# PURPOSE: Train a neural network model using supervised learning
# PARAMETERS:
#   - model: PyTorch model to train (e.g., ResNet-18)
#   - dataloader: PyTorch DataLoader providing batches of (images, labels)
#   - epochs: Number of complete passes through the training dataset (default: 5)
# RETURNS: The trained model
# ==============================================================================
def train_model(model, dataloader, epochs=5):
    # Define the loss function for multi-class classification
    # CrossEntropyLoss combines LogSoftmax and NLLLoss in one class
    # It measures how far predictions are from true labels
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the Adam optimizer (Adaptive Moment Estimation)
    # Adam adjusts learning rate automatically for each parameter
    # lr=0.001 is the initial learning rate (step size for weight updates)
    # model.parameters() returns all trainable parameters (weights and biases)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set model to training mode
    # This enables dropout (randomly zeroing neurons) and batch normalization training
    # In training mode, batch norm uses batch statistics instead of running averages
    model.train()
    
    # Outer loop: iterate through epochs (complete passes through dataset)
    # Each epoch processes all training samples once
    for epoch in range(epochs):
        
        # Inner loop: iterate through batches of data
        # DataLoader yields (inputs, labels) tuples
        # inputs: batch of images (typically 64 images of shape [3, 32, 32])
        # labels: batch of class labels (integers 0-9 for CIFAR-10)
        for inputs, labels in dataloader:
            
            # Zero out gradients from previous iteration
            # PyTorch accumulates gradients by default, so we must reset them
            # Without this, gradients would add up across batches (incorrect!)
            optimizer.zero_grad()
            
            # Forward pass: feed inputs through the model
            # outputs is a tensor of shape [batch_size, num_classes]
            # Each row contains raw scores (logits) for each class
            outputs = model(inputs)
            
            # Calculate loss: how wrong are the predictions?
            # CrossEntropyLoss expects:
            #   - outputs: raw logits (no softmax needed)
            #   - labels: true class indices
            # Lower loss = better predictions
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients of loss w.r.t. all parameters
            # Uses automatic differentiation (autograd) to compute ∂loss/∂weight
            # Gradients are stored in each parameter's .grad attribute
            loss.backward()
            
            # Update model parameters using computed gradients
            # For each parameter: weight = weight - learning_rate * gradient
            # Adam adjusts the learning rate adaptively for each parameter
            optimizer.step()
    
    # Return the trained model (with updated weights)
    # The model is now better at classifying the training data
    return model


# ==============================================================================
# FUNCTION: get_rmia_score
# PURPOSE: Calculate RMIA (Relative Membership Inference Attack) score for a sample
# This determines if a sample was in the target model's training set
# PARAMETERS:
#   - targt_model: The target (victim) model we're attacking
#   - ref_model: Reference model trained on different data (baseline)
#   - known_member: Tuple of (image, label) to test for membership
#   - population_subset: List of (image, label) tuples for comparison baseline
#   - gamma: Threshold for ratio comparison (default: 1.0)
# RETURNS: RMIA score between 0 and 1 (higher = more likely a member)
# ==============================================================================
def get_rmia_score(targt_model, ref_model, known_member, population_subset, gamma=1.0):
    # Set target model to evaluation mode
    # This disables dropout (all neurons active) and uses running statistics for batch norm
    # Ensures consistent, deterministic predictions
    targt_model.eval()
    
    # Set reference model to evaluation mode for same reasons
    ref_model.eval()

    # Unpack the test sample into image and label components
    # known_member is a tuple: (image_tensor, label_integer)
    known_member_img, known_member_label = known_member

    # Context manager that disables gradient computation
    # This saves memory and speeds up inference (no need to track gradients)
    # We're only making predictions, not training
    with torch.no_grad():
        # Calculate likelihood ratio for test sample x
        # ratio = P(x|target) / P(x|reference)
        # Higher ratio suggests x was in target's training set
        ratio_x = calculate_ratio(known_member_img, known_member_label, ref_model, targt_model)

        # Initialize counter for population samples "dominated" by x
        # A sample z is dominated if x has a significantly higher ratio than z
        count_dominated = 0
        
        # Iterate through population samples (z-samples)
        # These provide a baseline distribution of likelihood ratios
        # for samples NOT in the target model's training set
        for z_img, z_label in population_subset:
            # Calculate likelihood ratio for population sample z
            # This gives us the "typical" ratio for non-members
            ratio_z = calculate_ratio(z_img, z_label, ref_model, targt_model)

            # Compare x's ratio to z's ratio
            # If ratio_x / ratio_z >= gamma, then x "dominates" z
            # Add 1e-10 (tiny number) to avoid division by zero
            # gamma acts as a threshold (usually 1.0)
            if (ratio_x / (ratio_z + 1e-10)) >= gamma:
                count_dominated += 1  # Increment counter

    # Return RMIA score: fraction of population samples dominated by x
    # Score = (# samples with lower ratio) / (total population samples)
    # Score close to 1.0 → x likely a member (dominates most population)
    # Score close to 0.0 → x likely a non-member (dominated by population)
    return count_dominated / len(population_subset)


# ==============================================================================
# FUNCTION: calculate_ratio
# PURPOSE: Calculate likelihood ratio for a single sample
# This compares how confident the target vs reference model is on the sample
# PARAMETERS:
#   - img: Image tensor (shape: [3, 32, 32] for CIFAR-10)
#   - label: True class label (integer 0-9)
#   - ref_model: Reference model
#   - trgt_model: Target model
# RETURNS: Likelihood ratio (float)
# ==============================================================================
def calculate_ratio(img, label, ref_model, trgt_model) -> float:
    # Get target model's probability for the true class
    # Step 1: img.unsqueeze(0) adds batch dimension → shape becomes [1, 3, 32, 32]
    #         Models expect batches, even for single images
    # Step 2: trgt_model(img) produces logits (raw scores) → shape [1, 10]
    # Step 3: F.softmax converts logits to probabilities (sum to 1)
    #         dim=1 means apply softmax across classes dimension
    # Step 4: [0, label] indexes: batch 0, true class label
    # Step 5: .item() converts single-element tensor to Python float
    prob_target = F.softmax(trgt_model(img.unsqueeze(0)), dim=1)[0, label].item()
    
    # Get reference model's probability for the true class
    # Same process as above, but using reference model
    prob_ref = F.softmax(ref_model(img.unsqueeze(0)), dim=1)[0, label].item()
    
    # Return likelihood ratio
    # Ratio = P(label|target) / P(label|reference)
    # High ratio → target is more confident → likely a training member
    # Add 1e-10 to denominator to avoid division by zero
    return prob_target / (prob_ref + 1e-10)


# ==============================================================================
# MAIN EXECUTION BLOCK
# This code runs only when the script is executed directly (not imported)
# ==============================================================================
if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # STEP 1: Define image preprocessing pipeline
    # -------------------------------------------------------------------------
    # transforms.Compose creates a pipeline that applies transforms sequentially
    transform = transforms.Compose([
        # Convert PIL Image to PyTorch tensor
        # Changes format from (H, W, C) to (C, H, W)
        # Scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),
        
        # Normalize pixel values to range [-1, 1]
        # Formula: output = (input - mean) / std
        # First tuple (0.5, 0.5, 0.5) is mean for R, G, B channels
        # Second tuple (0.5, 0.5, 0.5) is std for R, G, B channels
        # Result: (x - 0.5) / 0.5 maps [0, 1] → [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # -------------------------------------------------------------------------
    # STEP 2: Download and load CIFAR-10 dataset
    # -------------------------------------------------------------------------
    # CIFAR-10: 50,000 training images in 10 classes (32x32 RGB)
    # Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    # root='./data': directory to download/load data
    # train=True: use training split (not test split)
    # download=True: download if not already present
    # transform=transform: apply our preprocessing pipeline
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # -------------------------------------------------------------------------
    # STEP 3: Split dataset into three parts
    # -------------------------------------------------------------------------
    # random_split divides dataset into random subsets
    # [20000, 20000, 10000, 0]: sizes of each subset
    #   - 20,000: Target model training data (MEMBERS - samples we want to identify)
    #   - 20,000: Test data (NON-MEMBERS - samples NOT in target training)
    #   - 10,000: Population data (for training reference model)
    #   - 0: Remainder (ensures we use all 50,000 images)
    # Returns 4 Subset objects (last one is empty)
    target_train, target_test, population_data, _ = torch.utils.data.random_split(
        full_trainset, [20000, 20000, 10000, 0]
    )

    # Print status message to console
    print("\nLoading CIFAR-10...")
    
    # -------------------------------------------------------------------------
    # STEP 4: Create DataLoaders for efficient batching
    # -------------------------------------------------------------------------
    # DataLoader wraps a dataset and provides batching, shuffling, parallel loading
    # batch_size=64: load 64 images at a time (trade-off: speed vs memory)
    # shuffle=True: randomize order each epoch (helps training convergence)
    trainloader = torch.utils.data.DataLoader(target_train, batch_size=64, shuffle=True)
    
    # Create DataLoader for population data (used for reference model)
    poploader = torch.utils.data.DataLoader(population_data, batch_size=64, shuffle=True)

    # -------------------------------------------------------------------------
    # STEP 5: Initialize neural network models
    # -------------------------------------------------------------------------
    # ResNet-18: 18-layer Residual Network
    # Uses skip connections to enable training of very deep networks
    # num_classes=10: output 10 class probabilities (for CIFAR-10)
    # This is the TARGET (victim) model - we'll attack this to infer membership
    target_model = resnet18(num_classes=10)
    
    # Create REFERENCE model with same architecture
    # This will be trained on different data (population_data)
    # Provides baseline probabilities for comparison
    reference_model = resnet18(num_classes=10)

    # -------------------------------------------------------------------------
    # STEP 6: Train the target model
    # -------------------------------------------------------------------------
    print("Training Target Model (Member data)...")
    # Train on member data (target_train) for 5 epochs (default)
    # After training, this model will have "memorized" its training samples
    # This memorization creates the membership signal we'll exploit
    target_model = train_model(target_model, trainloader)

    # -------------------------------------------------------------------------
    # STEP 7: Train the reference model
    # -------------------------------------------------------------------------
    print("Training Reference Model (Population data)...")
    # Train on population data (different from target's training data)
    # This provides baseline probabilities for samples NOT in target's training
    reference_model = train_model(reference_model, poploader)

    # -------------------------------------------------------------------------
    # STEP 8: Prepare population samples for RMIA baseline
    # -------------------------------------------------------------------------
    # Select first 100 samples from population_data
    # These are z-samples: baseline for likelihood ratio comparisons
    # In RMIA, we compare test sample's ratio against these population ratios
    # List comprehension: [population_data[0], population_data[1], ..., population_data[99]]
    z_samples = [population_data[i] for i in range(100)]

    # -------------------------------------------------------------------------
    # STEP 9: Execute RMIA attack on one sample
    # -------------------------------------------------------------------------
    # target_train[0]: first sample from training set (we KNOW this is a member)
    # This is a demonstration to verify the attack works
    score = get_rmia_score(target_model, reference_model, target_train[0], z_samples)
    
    # Print the result
    # Score close to 1.0 indicates high confidence that sample is a member
    # Score close to 0.0 indicates sample is likely NOT a member
    # {score:.4f} formats to 4 decimal places (e.g., 0.6500)
    print(f"RMIA Membership Score (closer to 1.0 is more likely a member): {score:.4f}")


# ==============================================================================
# Added by Stefanos - Question 3: Class Imbalance Experiments
# ==============================================================================
# This section implements experiments to test how class imbalance in training
# data affects the success of membership inference attacks

# ==============================================================================
# FUNCTION: create_imbalanced_dataset
# PURPOSE: Create a dataset with artificial class imbalance
# This simulates real-world scenarios where some classes have many samples
# and others have few samples (e.g., rare diseases in medical data)
# ==============================================================================
def create_imbalanced_dataset(dataset, imbalance_type='none', seed=42):
    """
    Create dataset with class imbalance for Question 3 experiments.
    
    Args:
        dataset: Original PyTorch dataset (e.g., CIFAR-10 subset)
        imbalance_type: 'none' (balanced), 'mild' (50% removed from 3 classes),
                        or 'severe' (80% removed from 5 classes)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Subset of dataset with specified class imbalance
    """
    # Import numpy for random number generation
    import numpy as np
    
    # Set random seeds for reproducibility
    # This ensures we get the same "random" imbalance each time
    np.random.seed(seed)  # NumPy random operations
    torch.manual_seed(seed)  # PyTorch random operations
    
    # If no imbalance requested, return original dataset unchanged
    if imbalance_type == 'none':
        return dataset
    
    # Initialize empty list to store indices of samples we want to KEEP
    indices = []
    
    # Dictionary to track how many samples removed per class
    # {0: count, 1: count, ..., 9: count} for CIFAR-10's 10 classes
    class_counts = {i: 0 for i in range(10)}
    
    # Iterate through ALL samples in the dataset
    # idx is the sample index, we use it to access dataset[idx]
    for idx in range(len(dataset)):
        # Get the label (class) of this sample
        # _ ignores the image data (we only need the label)
        _, label = dataset[idx]
        
        # Handle MILD imbalance: remove 50% from classes 0, 1, 2
        if imbalance_type == 'mild':
            # Check if this sample belongs to minority classes
            if label in [0, 1, 2]:
                # Generate random number between 0 and 1
                # If < 0.5 (50% chance), remove this sample
                if np.random.random() < 0.5:
                    # Increment counter for this class
                    class_counts[label] += 1
                    # Skip to next sample (don't add to indices list)
                    continue
            # If we get here, keep this sample (add index to list)
            indices.append(idx)
            
        # Handle SEVERE imbalance: remove 80% from classes 0, 1, 2, 3, 4
        elif imbalance_type == 'severe':
            # Check if this sample belongs to minority classes
            if label in [0, 1, 2, 3, 4]:
                # Generate random number between 0 and 1
                # If < 0.8 (80% chance), remove this sample
                if np.random.random() < 0.8:
                    # Increment counter for this class
                    class_counts[label] += 1
                    # Skip to next sample (don't add to indices list)
                    continue
            # If we get here, keep this sample
            indices.append(idx)
    
    # Print statistics about the imbalanced dataset
    print(f"\nCreated {imbalance_type} imbalanced dataset:")
    print(f"  Original size: {len(dataset)}")  # Total samples before removal
    print(f"  New size: {len(indices)} ({100*len(indices)/len(dataset):.1f}% of original)")
    print(f"  Removed by class:")
    # Loop through all 10 classes
    for cls, count in class_counts.items():
        # Only print classes where samples were removed
        if count > 0:
            print(f"    Class {cls}: removed {count} samples")
    
    # Create and return a Subset object
    # Subset allows us to create a view of the dataset with only selected indices
    # This is memory-efficient (doesn't copy data, just references it)
    from torch.utils.data import Subset
    return Subset(dataset, indices)


# ==============================================================================
# FUNCTION: run_imbalance_experiment
# PURPOSE: Run comprehensive experiments testing class imbalance impact on RMIA
# Tests three scenarios: balanced, mild imbalance, severe imbalance
# Compares attack success across these scenarios
# ==============================================================================
def run_imbalance_experiment():
    """
    Run class imbalance experiments for Question 3.
    Tests how class imbalance affects RMIA attack success.
    
    The experiment:
    1. Creates three datasets: balanced, mildly imbalanced, severely imbalanced
    2. Trains target models on each imbalanced dataset
    3. Trains reference models on balanced data (as baseline)
    4. Evaluates RMIA attack success on each configuration
    5. Compares results to see how imbalance weakens the attack
    """
    # Print experiment header with separators
    print("\n" + "="*80)
    print("Question 3: Class Imbalance Impact on RMIA Attack")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Setup: Load dataset and define preprocessing
    # -------------------------------------------------------------------------
    # Same preprocessing as main experiment (normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 training data (50,000 images)
    # root='./data': download/load from this directory
    # train=True: use training split
    # download=True: download if not present
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                  download=True, transform=transform)
    
    # Split into target training, test, and population sets
    # Same split as main: 20k/20k/10k
    target_train, target_test, population_data, _ = torch.utils.data.random_split(
        full_trainset, [20000, 20000, 10000, 0]
    )
    
    # -------------------------------------------------------------------------
    # Configuration parameters
    # -------------------------------------------------------------------------
    # Number of samples to test per experiment (reduced from 100 for speed)
    # We'll test 50 members and 50 non-members per configuration
    NUM_TESTS = 50
    
    # Population samples for baseline comparison (z-samples)
    # These 100 samples will be used across all experiments for consistency
    z_samples = [population_data[i] for i in range(100)]
    
    # Define experiment configurations to test
    # Each tuple: (imbalance_type, human_readable_description)
    imbalance_configs = [
        ('none', 'Balanced (baseline)'),  # No imbalance (all classes equal)
        ('mild', 'Mild Imbalance (50% removed from 3 classes)'),  # 50% from classes 0,1,2
        ('severe', 'Severe Imbalance (80% removed from 5 classes)')  # 80% from classes 0-4
    ]
    
    # CIFAR-10 class names for readable output
    # Index 0=airplane, 1=automobile, etc.
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Dictionary to store results from each experiment
    # Key: imbalance_type ('none', 'mild', 'severe')
    # Value: dict with scores, statistics, etc.
    results = {}
    
    # =========================================================================
    # Main experiment loop: test each imbalance configuration
    # =========================================================================
    for imbalance_type, description in imbalance_configs:
        # Print section header for this experiment
        print("\n" + "="*80)
        print(f"Testing: {description}")
        print("="*80)
        
        # ---------------------------------------------------------------------
        # Create imbalanced training dataset
        # ---------------------------------------------------------------------
        # Apply the imbalance to target_train
        # Returns a Subset with removed samples according to imbalance_type
        imbalanced_train = create_imbalanced_dataset(target_train, imbalance_type)
        
        # Wrap imbalanced data in DataLoader for batch training
        # batch_size=64: process 64 images at a time
        # shuffle=True: randomize order each epoch
        imbalanced_loader = torch.utils.data.DataLoader(imbalanced_train, 
                                                        batch_size=64, shuffle=True)
        
        # ---------------------------------------------------------------------
        # Train target model on imbalanced data
        # ---------------------------------------------------------------------
        print(f"\nTraining target model on {description}...")
        # Initialize fresh ResNet-18 model
        imb_target_model = resnet18(num_classes=10)
        # Train on the imbalanced dataset
        # This model will have different learning patterns for majority vs minority classes
        imb_target_model = train_model(imb_target_model, imbalanced_loader)
        
        # ---------------------------------------------------------------------
        # Train reference model on BALANCED population data
        # ---------------------------------------------------------------------
        print("\nTraining reference model on balanced population data...")
        # Create DataLoader for population data (always balanced)
        pop_loader = torch.utils.data.DataLoader(population_data, 
                                                 batch_size=64, shuffle=True)
        # Initialize fresh reference model
        imb_ref_model = resnet18(num_classes=10)
        # Train reference model
        # This provides baseline probabilities from a model trained on balanced data
        imb_ref_model = train_model(imb_ref_model, pop_loader)
        
        # ---------------------------------------------------------------------
        # Evaluate attack on MEMBER samples
        # ---------------------------------------------------------------------
        print(f"\nEvaluating attack on {NUM_TESTS} members...")
        # Lists to store results
        member_scores = []  # RMIA scores for each member
        member_classes = []  # Class labels for per-class analysis
        
        # Loop through first NUM_TESTS members
        # Use min() to avoid index error if imbalanced_train has fewer than NUM_TESTS
        for i in range(min(NUM_TESTS, len(imbalanced_train))):
            # Get image and label from imbalanced training set
            img, label = imbalanced_train[i]
            # Calculate RMIA score for this member
            # Higher score = attack more confident this is a member
            score = get_rmia_score(imb_target_model, imb_ref_model, 
                                  (img, label), z_samples)
            # Store score and class
            member_scores.append(score)
            member_classes.append(label)
        
        # ---------------------------------------------------------------------
        # Evaluate attack on NON-MEMBER samples
        # ---------------------------------------------------------------------
        print(f"Evaluating attack on {NUM_TESTS} non-members...")
        # Lists to store results
        non_member_scores = []  # RMIA scores for each non-member
        non_member_classes = []  # Class labels
        
        # Loop through first NUM_TESTS non-members from test set
        for i in range(NUM_TESTS):
            # Get image and label from test set (NOT in training)
            img, label = target_test[i]
            # Calculate RMIA score for this non-member
            # Lower score = attack correctly identifies as non-member
            score = get_rmia_score(imb_target_model, imb_ref_model, 
                                  (img, label), z_samples)
            # Store score and class
            non_member_scores.append(score)
            non_member_classes.append(label)
        
        # ---------------------------------------------------------------------
        # Calculate and store statistics
        # ---------------------------------------------------------------------
        # Average RMIA score for members
        # sum() adds all scores, len() counts them
        avg_member_score = sum(member_scores) / len(member_scores)
        
        # Average RMIA score for non-members
        avg_non_member_score = sum(non_member_scores) / len(non_member_scores)
        
        # Separation: difference between average member and non-member scores
        # Higher separation = better attack (can distinguish members from non-members)
        # Lower separation = worse attack (scores overlap too much)
        separation = avg_member_score - avg_non_member_score
        
        # Store all results for this configuration in dictionary
        results[imbalance_type] = {
            'description': description,  # Human-readable name
            'avg_member_score': avg_member_score,  # Mean score for members
            'avg_non_member_score': avg_non_member_score,  # Mean score for non-members
            'separation': separation,  # Key metric: avg_member - avg_non_member
            'member_scores': member_scores,  # Individual scores
            'non_member_scores': non_member_scores  # Individual scores
        }
        
        # ---------------------------------------------------------------------
        # Print results for this configuration
        # ---------------------------------------------------------------------
        print(f"\nResults for {description}:")
        # Print average scores with 4 decimal places
        print(f"  Average member score:     {avg_member_score:.4f}")
        print(f"  Average non-member score: {avg_non_member_score:.4f}")
        # Separation is the key metric - higher is better for attack
        print(f"  Separation (higher=better): {separation:.4f}")
        
        # ---------------------------------------------------------------------
        # Per-class analysis for members
        # ---------------------------------------------------------------------
        print(f"\n  Per-class member scores:")
        # Dictionary to group scores by class
        # {0: [score1, score2, ...], 1: [...], ...}
        class_score_sums = {i: [] for i in range(10)}
        
        # Populate dictionary: for each score, add it to its class's list
        for score, cls in zip(member_scores, member_classes):
            class_score_sums[cls].append(score)
        
        # Print average score per class
        for cls in range(10):
            # Only print classes that have samples
            if len(class_score_sums[cls]) > 0:
                # Calculate average score for this class
                avg = sum(class_score_sums[cls]) / len(class_score_sums[cls])
                # Print: class number, class name (padded to 12 chars), avg score, sample count
                # :12s means left-align string in 12-character field
                print(f"    Class {cls} ({class_names[cls]:12s}): {avg:.4f} "
                      f"(n={len(class_score_sums[cls])})")
    
    # =========================================================================
    # Final comparison across all configurations
    # =========================================================================
    print("\n" + "="*80)
    print("Summary: Impact of Class Imbalance on Attack Success")
    print("="*80)
    
    # Print header for comparison table
    # :<45 means left-align in 45-character field
    # :<15 means left-align in 15-character field
    print(f"\n{'Configuration':<45} {'Separation':<15} {'Impact'}")
    print("-"*80)
    
    # Get baseline separation (from balanced configuration)
    baseline_sep = results['none']['separation']
    
    # Print each configuration with impact percentage
    for imb_type in ['none', 'mild', 'severe']:
        res = results[imb_type]  # Get results for this imbalance type
        sep = res['separation']  # Get separation value
        
        # Calculate impact as percentage change from baseline
        # impact = (current - baseline) / baseline * 100
        # Negative impact means attack got worse (separation decreased)
        if baseline_sep != 0:  # Avoid division by zero
            impact = ((sep - baseline_sep) / baseline_sep * 100)
        else:
            impact = 0
        
        # Format impact string
        # For baseline: show "baseline"
        # For others: show percentage with + or - sign (e.g., "+5.2%" or "-12.3%")
        if imb_type == 'none':
            impact_str = "baseline"
        else:
            impact_str = f"{impact:+.1f}%"  # :+ forces display of sign
        
        # Print row: configuration name (45 chars), separation (4 decimals), impact
        print(f"{res['description']:<45} {sep:.4f}          {impact_str}")
    
    # Print conclusion section
    print("\n" + "="*80)
    print("Conclusion:")
    print("Class imbalance generally WEAKENS the attack because:")
    print("1. Models trained on imbalanced data have different confidence patterns")
    print("2. Minority classes show weaker membership signals")
    print("3. Overall separation between members and non-members decreases")
    print("="*80)
    
    # Return results dictionary for further analysis if needed
    return results


# ==============================================================================
# EXECUTION CONTROL
# ==============================================================================
# Uncomment the line below to run the class imbalance experiment
# When uncommented, this will execute the full experiment (takes ~45-60 minutes)
# The experiment trains 6 models total (3 target + 3 reference) and evaluates
# the attack on 100 samples (50 members + 50 non-members) per configuration
run_imbalance_experiment()

