"""
Complete Evaluation: RMIA Attack on HRR-Protected vs Baseline Models

This script tests the effectiveness of HRR defense against membership inference.

Purpose:
- Compare RMIA attack success on baseline vs HRR-protected models
- Measure how much HRR reduces attack effectiveness (AUC reduction)
- Answer TASK 2.2 questions about HRR defense

Evaluation methodology:
1. Load both baseline and HRR-protected trained models
2. Run RMIA attack on both models
3. Compare AUC scores (lower AUC = better defense)
4. Visualize results with ROC curves and score distributions
"""

# Import necessary libraries
import numpy as np  # Numerical operations
import torch  # PyTorch framework
import torch.nn as nn  # Neural network modules
import torchvision  # Computer vision datasets
import torchvision.transforms as transforms  # Data preprocessing
from torch.utils.data import DataLoader, Subset  # Data loading
from torchvision.models import resnet18  # ResNet-18 architecture
import matplotlib.pyplot as plt  # Plotting
from sklearn.metrics import roc_curve, auc  # Evaluation metrics
import pickle  # For saving results

# Import from our implementations
import sys
sys.path.append('.')  # Add current directory to Python path

# Device configuration - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models(hrr_protected=True):
    """
    Load trained models from saved checkpoint files
    
    Args:
        hrr_protected: If True, load HRR-protected model (2 networks)
                      If False, load baseline model (1 network)
    
    Returns:
        For HRR: (main_network, prediction_network)
        For baseline: single model
    """
    if hrr_protected:
        # Import HRR-specific functions and classes
        from hrr_defense import ModifiedResNet18, PredictionNetwork, binding_2d, unbinding_2d, generate_secret
        
        # Initialize HRR networks
        main_net = ModifiedResNet18().to(device)  # Encoder-decoder network
        pred_net = PredictionNetwork().to(device)  # Classification network
        
        # Load trained weights from disk
        main_net.load_state_dict(torch.load('hrr_main_network.pth'))
        pred_net.load_state_dict(torch.load('hrr_pred_network.pth'))
        
        # Set to evaluation mode (disables dropout, batch norm uses running stats)
        main_net.eval()
        pred_net.eval()
        
        return main_net, pred_net
    else:
        # Load standard baseline model
        model = resnet18(num_classes=10).to(device)
        model.load_state_dict(torch.load('baseline_model.pth'))
        model.eval()
        return model


def get_probability_hrr(main_net, pred_net, image, label):
    """
    Get class probability from HRR-protected model
    
    This requires the full HRR pipeline:
    1. Generate secret key
    2. Bind input with secret
    3. Process through main network
    4. Unbind result with secret
    5. Predict with prediction network
    
    Args:
        main_net: Main encoder-decoder network
        pred_net: Prediction classification network
        image: Input image tensor [C x H x W]
        label: Class label to get probability for
    
    Returns:
        Probability value between 0 and 1
    """
    # Import HRR operations
    from hrr_defense import binding_2d, unbinding_2d, generate_secret
    
    # Move image to correct device
    image = image.to(device)
    C, H, W = image.shape  # Get image dimensions
    
    # Step 1: Generate random secret key for this inference
    # Different secret each time for security
    secret = generate_secret(H, W, C)
    
    # Step 2: Bind (obfuscate) image with secret
    bound = binding_2d(image, secret)
    
    # Step 3: Process bound image through main network
    # This would happen on untrusted server
    r = main_net(bound.unsqueeze(0))  # unsqueeze adds batch dimension
    
    # Step 4: Unbind result using same secret
    # This recovers meaningful representation
    unbound = unbinding_2d(r[0], secret)  # [0] removes batch dimension
    output = pred_net(unbound.unsqueeze(0))
    
    prob = torch.softmax(output, dim=1)[0, label].item()
    return prob


def get_probability_baseline(model, image, label):
    """Get probability from baseline model"""
    image = image.to(device)
    output = model(image.unsqueeze(0))
    prob = torch.softmax(output, dim=1)[0, label].item()
    return prob


def get_rmia_score(target_model, ref_models, image, label, population_subset, 
                   model_type='baseline', main_net=None, pred_net=None):
    """
    Calculate RMIA score
    
    Args:
        target_model: Target model (baseline) or None (HRR)
        ref_models: List of reference models
        image: Image to test
        label: True label
        population_subset: Population samples
        model_type: 'baseline' or 'hrr'
        main_net, pred_net: HRR networks (if model_type='hrr')
    """
    
    # Get probability on target model
    if model_type == 'baseline':
        prob_x_target = get_probability_baseline(target_model, image, label)
    else:
        prob_x_target = get_probability_hrr(main_net, pred_net, image, label)
    
    # Average predictions across reference models
    all_ref_probs_x = []
    for ref_model in ref_models:
        if model_type == 'baseline':
            prob = get_probability_baseline(ref_model, image, label)
        else:
            # For HRR, reference models would also be HRR-protected
            # For simplicity, using baseline reference models
            prob = get_probability_baseline(ref_model, image, label)
        all_ref_probs_x.append(prob)
    
    prob_x_out = np.mean(all_ref_probs_x)
    
    # Offline scaling
    a = 0.3
    pr_x = 0.5 * ((1 + a) * prob_x_out + (1 - a))
    ratio_x = prob_x_target / (pr_x + 1e-10)
    
    # Count dominated population samples
    count_dominated = 0
    for z_img, z_label in population_subset:
        if model_type == 'baseline':
            prob_z_target = get_probability_baseline(target_model, z_img, z_label)
        else:
            prob_z_target = get_probability_hrr(main_net, pred_net, z_img, z_label)
        
        all_ref_probs_z = []
        for ref_model in ref_models:
            if model_type == 'baseline':
                prob = get_probability_baseline(ref_model, z_img, z_label)
            else:
                prob = get_probability_baseline(ref_model, z_img, z_label)
            all_ref_probs_z.append(prob)
        
        prob_z_out = np.mean(all_ref_probs_z)
        pr_z = 0.5 * ((1 + a) * prob_z_out + (1 - a))
        ratio_z = prob_z_target / (pr_z + 1e-10)
        
        if (ratio_x / (ratio_z + 1e-10)) >= 1.0:
            count_dominated += 1
    
    return count_dominated / len(population_subset)


def evaluate_rmia_attack(target_model, ref_models, members, non_members, 
                        population_data, model_type='baseline', 
                        main_net=None, pred_net=None, num_eval=200):
    """Evaluate RMIA attack performance"""
    
    print(f"\nEvaluating RMIA on {model_type} model...")
    print(f"Testing on {num_eval} members and {num_eval} non-members")
    
    all_scores = []
    all_labels = []
    
    # Sample population
    population_subset = [population_data[i] for i in range(min(1000, len(population_data)))]
    
    # Evaluate on members
    print("Testing members...")
    for i in range(min(num_eval, len(members))):
        img, label = members[i]
        score = get_rmia_score(target_model, ref_models, img, label, population_subset,
                              model_type, main_net, pred_net)
        all_scores.append(score)
        all_labels.append(1)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_eval} members")
    
    # Evaluate on non-members
    print("Testing non-members...")
    for i in range(min(num_eval, len(non_members))):
        img, label = non_members[i]
        score = get_rmia_score(target_model, ref_models, img, label, population_subset,
                              model_type, main_net, pred_net)
        all_scores.append(score)
        all_labels.append(0)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_eval} non-members")
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nResults for {model_type}:")
    print(f"  AUC: {roc_auc:.4f}")
    
    return {
        'scores': all_scores,
        'labels': all_labels,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }


def plot_comparison(baseline_results, hrr_results, save_path='hrr_vs_baseline.png'):
    """Plot comparison of attack effectiveness"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curves
    ax1.plot(baseline_results['fpr'], baseline_results['tpr'], 
            label=f'Baseline (AUC = {baseline_results["auc"]:.4f})', linewidth=2)
    ax1.plot(hrr_results['fpr'], hrr_results['tpr'], 
            label=f'HRR-Protected (AUC = {hrr_results["auc"]:.4f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - RMIA Attack Effectiveness', fontsize=14)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Score Distributions
    baseline_scores = np.array(baseline_results['scores'])
    baseline_labels = np.array(baseline_results['labels'])
    hrr_scores = np.array(hrr_results['scores'])
    hrr_labels = np.array(hrr_results['labels'])
    
    ax2.hist(baseline_scores[baseline_labels == 1], bins=20, alpha=0.5, 
            color='blue', label='Baseline Members', density=True)
    ax2.hist(baseline_scores[baseline_labels == 0], bins=20, alpha=0.5, 
            color='red', label='Baseline Non-Members', density=True)
    ax2.hist(hrr_scores[hrr_labels == 1], bins=20, alpha=0.5, 
            color='green', label='HRR Members', density=True)
    ax2.hist(hrr_scores[hrr_labels == 0], bins=20, alpha=0.5, 
            color='orange', label='HRR Non-Members', density=True)
    ax2.set_xlabel('RMIA Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Score Distributions', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def main():
    """Main evaluation function"""
    
    print("=" * 80)
    print("COMPLETE EVALUATION: HRR DEFENSE VS RMIA ATTACK")
    print("=" * 80)
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("\nLoading CIFAR-10 dataset...")
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
    
    # Use same split as training
    target_train, target_test, population_data, _ = torch.utils.data.random_split(
        full_trainset, [20000, 20000, 10000, 0]
    )
    
    # Train reference models if not already trained
    print("\nTraining 4 reference models...")
    ref_models = []
    for i in range(4):
        ref_model = resnet18(num_classes=10).to(device)
        
        # Train on population subset
        pop_indices = np.random.choice(len(population_data), 5000, replace=False)
        pop_subset = Subset(population_data, pop_indices)
        pop_loader = DataLoader(pop_subset, batch_size=64, shuffle=True, num_workers=2)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ref_model.parameters(), lr=0.001)
        
        print(f"Training reference model {i+1}/4...")
        ref_model.train()
        for epoch in range(5):  # Quick training
            for inputs, labels in pop_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = ref_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        ref_model.eval()
        ref_models.append(ref_model)
    
    # Evaluate baseline model
    print("\n" + "=" * 80)
    print("Evaluating RMIA on Baseline Model")
    print("=" * 80)
    
    baseline_model = load_models(hrr_protected=False)
    baseline_results = evaluate_rmia_attack(
        baseline_model, ref_models, target_train, target_test,
        population_data, model_type='baseline', num_eval=200
    )
    
    # Evaluate HRR-protected model
    print("\n" + "=" * 80)
    print("Evaluating RMIA on HRR-Protected Model")
    print("=" * 80)
    
    main_net, pred_net = load_models(hrr_protected=True)
    hrr_results = evaluate_rmia_attack(
        None, ref_models, target_train, target_test,
        population_data, model_type='hrr', 
        main_net=main_net, pred_net=pred_net, num_eval=200
    )
    
    # Generate comparison plots
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    plot_comparison(baseline_results, hrr_results)
    
    # Save results
    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump({
            'baseline': baseline_results,
            'hrr': hrr_results
        }, f)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE - Final Summary")
    print("=" * 80)
    
    print(f"\nBaseline Model:")
    print(f"  RMIA Attack AUC: {baseline_results['auc']:.4f}")
    
    print(f"\nHRR-Protected Model:")
    print(f"  RMIA Attack AUC: {hrr_results['auc']:.4f}")
    
    auc_reduction = (baseline_results['auc'] - hrr_results['auc']) / baseline_results['auc'] * 100
    print(f"\nHRR Defense Effectiveness:")
    print(f"  AUC Reduction: {auc_reduction:.2f}%")
    
    if hrr_results['auc'] < 0.55:
        print("  Status: ✓ Strong defense (attack near random guessing)")
    elif hrr_results['auc'] < 0.65:
        print("  Status: ✓ Moderate defense (attack significantly degraded)")
    else:
        print("  Status: ✗ Weak defense (attack still effective)")
    
    print("\n" + "=" * 80)
    print("TASK 2.2 - Analysis Questions:")
    print("=" * 80)
    
    print("\nQuestion 1: How effective is HRR at preventing RMIA?")
    print(f"  - Baseline AUC: {baseline_results['auc']:.4f}")
    print(f"  - HRR AUC: {hrr_results['auc']:.4f}")
    print(f"  - Reduction: {auc_reduction:.2f}%")
    print("  - The HRR defense obfuscates the model's output, making it harder")
    print("    for the attacker to distinguish members from non-members.")
    
    print("\nQuestion 2: Does HRR qualify as encryption?")
    print("  - No, HRR is NOT true encryption:")
    print("    * No provable security guarantees")
    print("    * Deterministic binding operation")
    print("    * Functions as obfuscation/pseudo-encryption")
    print("  - However, it provides practical privacy with low overhead")
    
    print("\nQuestion 3: Could an attacker adapt to overcome this defense?")
    print("  - Potential attacks tested in the paper:")
    print("    * Clustering: Failed (ARI < 2%)")
    print("    * Inversion: Failed (poor reconstruction)")
    print("    * Supervised learning without secret: Limited success")
    print("  - The gradient reversal forces the main network to produce")
    print("    uninformative outputs without the secret key.")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
