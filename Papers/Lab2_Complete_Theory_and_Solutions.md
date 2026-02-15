# Practice Lab 2: Membership Inference Attacks and Defense Mechanisms
## Complete Theory and Solutions

---

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Part I: Model Inversion with RMIA](#part-i-model-inversion-with-rmia)
4. [Part II: Obfuscation with HRR](#part-ii-obfuscation-with-hrr)
5. [References](#references)

---

## Overview

This document provides comprehensive theoretical explanations and answers to all questions posed in Practice Lab 2, which focuses on:
- **Part I**: Implementing Robust Membership Inference Attack (RMIA) against ResNet-18 models trained on CIFAR-10
- **Part II**: Implementing Holographically Reduced Representations (HRR) defense mechanism

---

## Theoretical Background

### What is Membership Inference Attack?

A **Membership Inference Attack (MIA)** attempts to determine whether a specific data sample was part of the training dataset of a machine learning model. This type of attack poses significant privacy risks, as it can reveal sensitive information about individuals whose data was used to train the model.

#### Key Concepts:
- **Target Model (θ)**: The machine learning model under attack
- **Target Data Point (x)**: The sample being tested for membership
- **Training Set (S)**: The dataset used to train the target model
- **Population Distribution (π)**: The distribution from which training data is sampled

#### MIA Game Definition:
The MIA is formulated as a hypothesis testing problem between two worlds:
- **H_in**: The data point x was in the training set of θ
- **H_out** (null hypothesis): The data point x was not in the training set of θ

---

## Part I: Model Inversion with RMIA

### Paper Reference
**"Low-Cost High-Power Membership Inference Attacks"** by Zarifzadeh et al. (2024)
- arXiv: https://arxiv.org/abs/2312.03262
- Published at ICML 2024

### 1.1 Implementation Details

#### What is RMIA?

**Robust Membership Inference Attack (RMIA)** is a state-of-the-art membership inference attack that achieves high test power with low computational overhead. Unlike previous methods, RMIA:

1. **Uses Fine-Grained Hypothesis Testing**: Instead of testing against an average null hypothesis, RMIA tests x against multiple random samples z from the population distribution
2. **Leverages Both Reference Models and Population Data**: Combines information from reference models (trained on random subsets) with population data
3. **Computes Pairwise Likelihood Ratios**: For each population sample z, computes how much more likely x is to be a member compared to z

#### Mathematical Foundation

The core of RMIA is the **likelihood ratio test** for pairs (x, z):

```
LR_θ(x, z) = Pr(θ|x) / Pr(θ|z)
```

Using Bayes' rule, this can be computed in the black-box setting as:

```
LR_θ(x, z) = [Pr(x|θ) / Pr(x)] / [Pr(z|θ) / Pr(z)]
```

Where:
- **Pr(x|θ)**: Likelihood of x given model θ (the model's prediction confidence on x)
- **Pr(x)**: Marginal probability of x (averaged over reference models)
- **Pr(z|θ)**: Likelihood of z given model θ
- **Pr(z)**: Marginal probability of z

#### RMIA Score Computation

```
Score_MIA(x; θ) = Pr_{z~π}[LR_θ(x, z) ≥ γ]
```

This measures the probability that x can γ-dominate a random sample z from the population, where γ ≥ 1 is a threshold parameter.

#### Implementation Steps

1. **Data Preparation**:
   - Split CIFAR-10 dataset into training and reserved sets
   - Training set: Used to train target model
   - Reserved set: Contains known members and non-members for evaluation

2. **Train Target Model**:
   - Architecture: ResNet-18
   - Dataset: CIFAR-10
   - Train on subset of data, reserving samples for testing

3. **Train Reference Models**:
   - Train k reference models on random subsets from population
   - These are "OUT" models where test samples are NOT included
   - Typically need 1-4 reference models for RMIA (unlike LiRA which needs many more)

4. **Compute Attack Scores**:
   ```python
   For each test sample x:
       # Compute Pr(x) from reference models
       Pr_x = average([Pr(x|θ_i) for θ_i in reference_models])
       
       # Compute likelihood ratio for target model
       LR_target = Pr(x|target_θ) / Pr_x
       
       # Sample many z from population
       for each z in population_samples:
           # Compute Pr(z) from reference models
           Pr_z = average([Pr(z|θ_i) for θ_i in reference_models])
           
           # Compute likelihood ratio for z
           LR_z = Pr(z|target_θ) / Pr_z
           
           # Check if x dominates z
           if LR_target / LR_z >= γ:
               count += 1
       
       # MIA score is fraction of dominated samples
       Score_MIA(x) = count / total_z_samples
   ```

5. **Make Predictions**:
   ```python
   if Score_MIA(x) >= β:
       return MEMBER
   else:
       return NON_MEMBER
   ```

### 1.2 Analysis and Evaluation

#### Question 1: How close do your results get to the paper?

**Expected Results from Paper**:
- CIFAR-10 with 1 reference model:
  - AUC: ~68.64%
  - TPR @ 0.01% FPR: ~1.19%
  - TPR @ 0% FPR: ~0.51%

- CIFAR-10 with 2 reference models:
  - AUC: ~70.13%
  - TPR @ 0.01% FPR: ~1.71%
  - TPR @ 0% FPR: ~0.91%

**Evaluation Metrics**:

1. **ROC Curve (Receiver Operating Characteristic)**:
   - Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
   - Shows attack performance across all threshold values β
   - Better attacks have curves closer to the top-left corner

2. **AUROC (Area Under ROC Curve)**:
   - Single number summarizing overall attack power
   - Perfect attack: AUROC = 1.0
   - Random guessing: AUROC = 0.5
   - Higher is better

3. **TPR at specific FPR values**:
   - TPR @ 0% FPR: Most strict metric, no false positives allowed
   - TPR @ 0.01% FPR: Very low FPR, relevant for reconstruction attacks
   - Measures attack power when being very conservative

**Analysis Approach**:
```
1. Train target ResNet-18 on CIFAR-10
2. Train reference models (1, 2, 4, etc.)
3. Implement RMIA attack
4. Compute metrics:
   - Plot TPR-FPR curve
   - Calculate AUROC
   - Measure TPR at FPR = 0%, 0.01%, 0.1%, 1%
5. Compare with paper results
6. Discuss any discrepancies:
   - Hyperparameter differences
   - Implementation details
   - Random seed effects
   - Model convergence
```

#### Question 2: How does the number of reference models affect the attack's success?

**Expected Behavior**:

From the RMIA paper results:

| # Reference Models | CIFAR-10 AUC | TPR @ 0.01% FPR | TPR @ 0% FPR |
|-------------------|--------------|-----------------|--------------|
| 0 (Attack-P)      | 58.19%       | 0.01%           | 0.0%         |
| 1                 | 68.64%       | 1.19%           | 0.51%        |
| 2                 | 70.13%       | 1.71%           | 0.91%        |
| 4                 | 71.02%       | 2.91%           | 2.13%        |
| 127               | 71.71%       | 4.18%           | 3.14%        |

**Key Insights**:

1. **Diminishing Returns**: 
   - Largest improvement from 0 to 1 reference model
   - Adding more models helps, but with diminishing returns
   - RMIA reaches near-peak performance with just 1-4 models

2. **Comparison with Other Attacks**:
   - **LiRA** (Carlini et al., 2022): Needs 64+ models for good performance
   - **Attack-R** (Ye et al., 2022): Benefits from more models but plateaus earlier
   - **RMIA**: Highly efficient, strong with few models

3. **Ideal Number**:
   - **Practical Setting**: 1-2 reference models (offline setting)
   - **Optimal Performance**: 4-8 models balances cost and power
   - **Maximum Performance**: 127+ models (marginal gains)

**Why RMIA is Efficient**:
- Uses population data calibration: Comparing x not just to reference models, but to population samples z
- Pairwise likelihood ratios: Each z vote provides information
- Robust scoring: Less sensitive to noise in individual reference models

**Experimental Design**:
```python
# Test different numbers of reference models
num_ref_models = [1, 2, 4, 8, 16, 32, 64, 127]

results = {}
for k in num_ref_models:
    # Train k reference models
    ref_models = train_reference_models(k)
    
    # Run RMIA attack
    attack_results = run_rmia(target_model, ref_models, test_data)
    
    # Record metrics
    results[k] = {
        'auc': compute_auc(attack_results),
        'tpr_at_0_fpr': compute_tpr_at_fpr(attack_results, fpr=0.0),
        'tpr_at_001_fpr': compute_tpr_at_fpr(attack_results, fpr=0.0001)
    }

# Plot: number of models vs performance
plot_performance_vs_num_models(results)
```

#### Question 3: What happens with class imbalance?

**Expected Effects**:

1. **Model Training Impact**:
   - Class imbalance affects model's learning
   - Model may overfit to majority classes
   - Minority class samples may be more "memorable"

2. **Attack Performance by Class**:
   ```
   Hypothesis: Samples from minority classes in training set 
   may be easier to identify as members because:
   - Model memorizes them more (due to rarity)
   - Higher likelihood ratios for minority class members
   - More distinguishable from population distribution
   ```

3. **Population Distribution Mismatch**:
   - If training data has imbalance but population data doesn't:
     - Attack may be more effective on minority class members
     - False positive rate may vary by class
   - If both have same imbalance:
     - Effect may be less pronounced
     - Attack still works but requires calibration

**Experimental Design**:

```python
# Create class-imbalanced datasets
def create_imbalanced_dataset(dataset, imbalance_ratio):
    """
    imbalance_ratio: dict mapping class -> sampling ratio
    Example: {0: 1.0, 1: 0.5, 2: 0.1, ...} 
    means keep all class 0, 50% of class 1, 10% of class 2
    """
    imbalanced_data = []
    for data, label in dataset:
        if random.random() < imbalance_ratio[label]:
            imbalanced_data.append((data, label))
    return imbalanced_data

# Test different imbalance scenarios
scenarios = [
    {'name': 'balanced', 'ratio': {i: 1.0 for i in range(10)}},
    {'name': 'mild_imbalance', 'ratio': {0:1.0, 1:0.8, 2:0.6, ...}},
    {'name': 'severe_imbalance', 'ratio': {0:1.0, 1:0.5, 2:0.1, ...}},
]

for scenario in scenarios:
    # Create imbalanced training set
    train_data = create_imbalanced_dataset(cifar10_train, scenario['ratio'])
    
    # Train target model
    target_model = train_resnet18(train_data)
    
    # Run RMIA attack
    results = run_rmia(target_model, ref_models, test_data)
    
    # Analyze by class
    for class_id in range(10):
        class_samples = [s for s in test_data if s.label == class_id]
        class_results = evaluate_attack_on_samples(results, class_samples)
        
        print(f"Class {class_id} (ratio={scenario['ratio'][class_id]}):")
        print(f"  AUC: {class_results['auc']}")
        print(f"  TPR @ 0% FPR: {class_results['tpr']}")
```

**Expected Findings**:
- Minority classes (low sampling ratio): Higher vulnerability to MIA
- Majority classes: More similar to population, harder to distinguish
- Overall attack effectiveness may decrease with severe imbalance
- Need class-specific threshold β for consistent FPR across classes

---

## Part II: Obfuscation with HRR

### Paper Reference
**"Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations"** by Alam et al. (2022)
- arXiv: https://arxiv.org/abs/2206.05893
- Published at ICML 2022

### 2.1 Defending Against the Attack

#### What is HRR?

**Holographic Reduced Representations (HRR)** is a method from symbolic AI that represents compositional structure using circular convolution. The paper adapts HRR to create a defense mechanism against inference attacks by obfuscating both inputs and outputs of a neural network.

#### Core HRR Operations

1. **Binding Operation** (⊛):
   ```
   B = x ⊛ s = F^(-1)[F(x) ⊙ F(s)]
   ```
   Where:
   - F() is the 2D Fourier Transform
   - F^(-1)() is the inverse 2D Fourier Transform
   - ⊙ is element-wise multiplication
   - x is the input image
   - s is a random secret vector
   - B is the bound (encrypted) image

2. **Unbinding Operation** (⊛†):
   ```
   x' ≈ B ⊛ s† ≈ x
   ```
   Where:
   - s† is the inverse of secret s
   - Defined as: F(s†) = 1/F(s)
   - x' is the retrieved (decrypted) image

#### Key Properties of HRR

1. **Commutative**: x ⊛ s = s ⊛ x
2. **Approximate Retrieval**: Can recover bound components with some noise
3. **Dimensionality Preserving**: Binding stays in same d-dimensional space
4. **Random Appearance**: Bound images look random, hiding original content

#### 2D HRR Adaptation

The original HRR uses 1D FFT (treating images as flattened vectors). The paper extends to **2D HRR**:

```python
def bind_2d_hrr(image, secret):
    """
    image: (H, W, C) input image
    secret: (H, W, C) random secret
    """
    # Apply 2D FFT to both
    image_freq = fft2(image)
    secret_freq = fft2(secret)
    
    # Multiply in frequency domain
    bound_freq = image_freq * secret_freq
    
    # Transform back to spatial domain
    bound_image = ifft2(bound_freq)
    
    return bound_image

def unbind_2d_hrr(bound_image, secret):
    """
    Retrieve original image using secret
    """
    # Compute inverse of secret in frequency domain
    secret_freq = fft2(secret)
    secret_inv_freq = 1.0 / secret_freq
    
    # Apply to bound image
    bound_freq = fft2(bound_image)
    retrieved_freq = bound_freq * secret_inv_freq
    
    # Transform back
    retrieved_image = ifft2(retrieved_freq)
    
    return retrieved_image
```

#### Network Architecture

The simplified HRR defense pipeline (as specified in the task):

```
Input Image (x)
    ↓
[Binding: x̂ = x ⊛ s]  ← User-side, secret s stays local
    ↓
Bound Image (x̂) → Send to untrusted server
    ↓
[ResNet-18: r = f_W(x̂)]  ← Main network on server
    ↓
Encrypted Output (r) → Return to user
    ↓
[Unbinding: r ⊛ s†]  ← User-side
    ↓
[Prediction Network: y = f_P(r ⊛ s†)]  ← User-side
    ↓
Final Prediction (y)
```

**Simplifications from Paper**:
1. **Replace U-Net with ResNet-18**: 
   - U-Net has input/output same size
   - ResNet-18 needs adaptation: add deconvolution layers to match input size
   - Or: project secret s to match ResNet-18 output dimensions

2. **Remove Adversarial Network**:
   - Paper uses f_A() with gradient reversal to enforce obfuscation
   - Simplified version: train only f_W() and f_P()
   - May sacrifice some defense strength but easier to implement

3. **Ignore Accuracy Recovery Measures**:
   - Paper uses averaging over multiple secrets to improve accuracy
   - Simplified: use single secret per prediction

#### Implementation Steps

**Step 1: Prepare ResNet-18 with Modified Architecture**

Option A - Add Deconvolution Layers:
```python
class ResNetHRR(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard ResNet-18 feature extractor
        self.resnet = models.resnet18(pretrained=False)
        # Remove final FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add deconvolution to restore spatial dimensions
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Back to 3 channels
        )
    
    def forward(self, x):
        # x: bound image (32, 32, 3) for CIFAR-10
        features = self.resnet(x)  # (512, 1, 1)
        output = self.deconv_layers(features)  # (3, 32, 32)
        return output
```

Option B - Project Secret:
```python
class ResNetHRRProjected(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(num_classes=512*4*4)
        
    def forward(self, x):
        # Output is flattened feature vector
        features = self.resnet(x)  # (512*4*4,)
        # Reshape to spatial dimensions
        output = features.view(-1, 512, 4, 4)
        return output

# Then project secret to match:
def project_secret(secret, target_shape):
    """Project secret from input shape to output shape"""
    # Use learned projection network
    projector = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 512*4*4),
        nn.Unflatten(1, (512, 4, 4))
    )
    return projector(secret)
```

**Step 2: Implement HRR Operations**

```python
import torch
import torch.fft as fft

def generate_secret(shape):
    """Generate random secret with unit magnitude in frequency domain"""
    # Sample from normal distribution
    secret = torch.randn(shape)
    
    # Project to unit magnitude in frequency domain
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    secret_magnitude = torch.abs(secret_freq)
    secret_freq_normalized = secret_freq / (secret_magnitude + 1e-8)
    secret = fft.ifft2(secret_freq_normalized, dim=(-2, -1)).real
    
    return secret

def bind_hrr(image, secret):
    """Bind image with secret using 2D HRR"""
    # FFT of both
    image_freq = fft.fft2(image, dim=(-2, -1))
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    
    # Multiply in frequency domain
    bound_freq = image_freq * secret_freq
    
    # IFFT back to spatial domain
    bound = fft.ifft2(bound_freq, dim=(-2, -1)).real
    
    return bound

def unbind_hrr(bound, secret):
    """Unbind using secret inverse"""
    # FFT of bound and secret
    bound_freq = fft.fft2(bound, dim=(-2, -1))
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    
    # Compute inverse (conjugate in freq domain)
    secret_inv_freq = torch.conj(secret_freq) / (torch.abs(secret_freq)**2 + 1e-8)
    
    # Unbind
    retrieved_freq = bound_freq * secret_inv_freq
    retrieved = fft.ifft2(retrieved_freq, dim=(-2, -1)).real
    
    return retrieved
```

**Step 3: Training Procedure**

```python
# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Generate random secret for each image
        secrets = generate_secret(images.shape).to(device)
        
        # Bind images with secrets
        bound_images = bind_hrr(images, secrets)
        
        # Forward through main network (ResNet-18)
        encrypted_outputs = main_network(bound_images)
        
        # Unbind outputs
        decrypted_outputs = unbind_hrr(encrypted_outputs, secrets)
        
        # Forward through prediction network
        predictions = prediction_network(decrypted_outputs)
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Step 4: Inference Procedure**

```python
def predict_with_hrr(image, main_network, prediction_network):
    """Make prediction on a single image with HRR obfuscation"""
    # Generate secret
    secret = generate_secret(image.shape)
    
    # Bind
    bound_image = bind_hrr(image, secret)
    
    # Send to untrusted server (main network)
    encrypted_output = main_network(bound_image)
    
    # Receive encrypted output, unbind locally
    decrypted_output = unbind_hrr(encrypted_output, secret)
    
    # Predict locally
    prediction = prediction_network(decrypted_output)
    
    return prediction
```

### 2.2 Analysis and Evaluation

#### Question 1: How effective is HRR at preventing RMIA from succeeding?

**Expected Findings**:

HRR should significantly reduce RMIA attack effectiveness because:

1. **Input Obfuscation**: Bound images x̂ = x ⊛ s look random
   - Attacker sees x̂, not x
   - Without secret s, cannot recover x
   - RMIA relies on model output Pr(x|θ), but now sees Pr(x̂|θ)

2. **Output Obfuscation**: Network output r = f_W(x̂) is also encrypted
   - Cannot directly use as likelihood Pr(x|θ)
   - Need to unbind first, which requires secret s

**Experimental Setup**:

```python
# Scenario 1: Attack on undefended model
target_model = train_resnet18(cifar10_train)
rmia_results_baseline = run_rmia_attack(target_model, test_data)

# Scenario 2: Attack on HRR-defended model
# Attacker only sees main network f_W
hrr_model = train_hrr_defense(cifar10_train)
main_network = hrr_model.get_main_network()  # Only part on server

# Attack Variant 1: RMIA on bound inputs
# Attacker intercepts x̂ and tries membership inference
bound_test_data = [(bind_hrr(x, generate_secret(x.shape)), label) 
                   for x, label in test_data]
rmia_results_bound = run_rmia_attack(main_network, bound_test_data)

# Attack Variant 2: RMIA on encrypted outputs
# Attacker observes r = f_W(x̂) and tries to infer membership
encrypted_outputs = [main_network(bind_hrr(x, generate_secret(x.shape))) 
                     for x, _ in test_data]
rmia_results_encrypted = run_rmia_attack_on_outputs(encrypted_outputs, test_data)

# Compare results
print(f"Baseline AUROC: {rmia_results_baseline['auc']}")
print(f"Attack on bound inputs AUROC: {rmia_results_bound['auc']}")
print(f"Attack on encrypted outputs AUROC: {rmia_results_encrypted['auc']}")
```

**Expected Defense Effectiveness**:

From the HRR paper results:
- **Attack-P** (no reference models): Random-level performance (~50% AUC)
- **Clustering attacks**: ARI ≤ 1.5% (near-random)
- **Unrealistic adversary** (knows everything except secrets): 
  - MNIST: 19.72% accuracy (vs 10% random)
  - CIFAR-10: 12.91% accuracy (vs 10% random)
  - Only 1.3-2× better than random guessing

**Metrics to Report**:
1. AUROC comparison (baseline vs defended)
2. TPR @ 0% FPR (should drop significantly)
3. Accuracy of defended model (should remain high)
4. Accuracy drop due to defense

#### Question 2: Does HRR qualify as encryption?

**Analysis Framework**:

Let's compare HRR with traditional encryption standards:

| Property | Strong Encryption | HRR Defense |
|----------|------------------|-------------|
| **Provable Security** | ✓ Mathematical proofs (e.g., AES, RSA) | ✗ Heuristic, empirical security |
| **Key Size** | Fixed (e.g., 128-256 bits) | Large (entire image dimensions) |
| **Deterministic Recovery** | ✓ Perfect recovery with key | ≈ Approximate recovery with noise |
| **Security Guarantees** | ✓ Computational hardness assumptions | ✗ No formal guarantees |
| **Ciphertext Indistinguishability** | ✓ Proven indistinguishable | Empirically appears random |
| **Known-Plaintext Resistance** | ✓ Secure even with known pairs | ? Unclear, untested |
| **Multiple Encryption** | ✓ Can chain/nest | ✓ Can bind multiple times |
| **Key Management** | ✓ Well-established protocols | ✗ New secret per prediction |

**Arguments FOR "Encryption-like"**:

1. **One-Time Pad Analogy**:
   - HRR creates "pseudo one-time pad"
   - Each prediction uses new random secret s
   - Bound image x̂ = x ⊛ s appears random
   - Without s, infinite possible (x, s) pairs produce same x̂

2. **Obfuscation Properties**:
   - Visual inspection: bound images look like random noise
   - Statistical tests: clustering fails to extract information
   - Adversarial resistance: even powerful adversaries struggle

3. **Reversibility**:
   - With secret s: can recover x from x̂
   - Without secret s: cannot recover x
   - Acts like symmetric encryption

**Arguments AGAINST "True Encryption"**:

1. **No Provable Security**:
   - Paper explicitly states: "emphatically stress this is not strong encryption"
   - Security is empirical, not mathematically proven
   - Could be broken by unknown attack

2. **Approximate Recovery**:
   - Unbinding has noise: x' ≈ x, not x' = x
   - True encryption must be lossless
   - Noise accumulates with network processing

3. **Not Uniformly Random**:
   - Bound images not uniformly distributed
   - Different from true one-time pad
   - Potential statistical vulnerabilities

4. **Linear Operation**:
   - FFT-based binding is linear operation
   - Could be vulnerable to linear algebraic attacks
   - Modern encryption uses nonlinear operations

5. **Contextual Security**:
   - Paper's threat model: hide task from untrusted server
   - Not designed for: adversarial decryption attempts
   - Different goals than cryptographic encryption

**Conclusion**:

**HRR is NOT true encryption**, but rather a **"pseudo-encryption" or "obfuscation" mechanism**:

✓ **What it does well**:
- Practical runtime efficiency (5000× faster than HE)
- Empirical protection against current MIA attacks
- Task-level obfuscation (hide what model does)
- Works with standard deep learning operations

✗ **What it lacks**:
- Formal security proofs
- Guarantees against all possible attacks
- Standards compliance (no cryptographic certification)
- Suitability for highly sensitive data

**Recommendation**: Use HRR when:
- Performance is critical
- Privacy is important but not mission-critical
- Threat model is inference attacks, not determined adversary
- Can accept heuristic vs provable security

Do NOT use HRR when:
- Handling highly sensitive data (medical, financial, etc.)
- Require regulatory compliance (GDPR, HIPAA, etc.)
- Adversary has strong motivation and resources
- Need provable security guarantees

#### Question 3: Could an attacker adapt their strategy to overcome this defense?

**Potential Attack Strategies**:

**Strategy 1: Model Inversion Attack**

Attempt to recover secret s or original input x from bound image x̂:

```python
# Attack: Optimize secret to maximize likelihood
def invert_secret(bound_image, main_network, population_data):
    """
    Try to find secret that makes bound_image look realistic
    """
    # Initialize random secret guess
    secret_guess = torch.randn(bound_image.shape, requires_grad=True)
    optimizer = torch.optim.Adam([secret_guess], lr=0.01)
    
    for iteration in range(1000):
        # Unbind with guessed secret
        retrieved = unbind_hrr(bound_image, secret_guess)
        
        # Loss: how realistic does retrieved image look?
        # Use FID score against population
        fid_loss = compute_fid(retrieved, population_data)
        
        # Or: how well does it fit the model output?
        output = main_network(bind_hrr(retrieved, secret_guess))
        consistency_loss = mse(output, original_encrypted_output)
        
        loss = fid_loss + consistency_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return unbind_hrr(bound_image, secret_guess)
```

**Expected Effectiveness**: Low
- Paper tested this: "Examples...showing inverting original images is highly challenging"
- High-dimensional optimization problem (32×32×3 = 3072 dimensions)
- Many local minima
- FID score is weak guidance signal

**Strategy 2: Multiple Query Attack**

If attacker can query same image multiple times with different secrets:

```python
# Collect multiple encrypted versions of same image
def multiple_query_attack(image, main_network, num_queries=100):
    """
    Query same image multiple times, each with different secret
    Try to extract information from ensemble
    """
    encrypted_outputs = []
    for _ in range(num_queries):
        secret = generate_secret(image.shape)
        bound = bind_hrr(image, secret)
        output = main_network(bound)
        encrypted_outputs.append(output)
    
    # Try to find commonality across outputs
    # Idea: the image is same, only secrets differ
    avg_output = torch.mean(torch.stack(encrypted_outputs), dim=0)
    
    # Does avg_output reveal information about image?
    return avg_output
```

**Expected Effectiveness**: Low to Medium
- Paper notes: "the same can not be said for the adversary" (Fig 6)
- Averaging doesn't significantly help attacker
- Secrets are independent, outputs remain obfuscated
- Would need to compromise user's system to intercept secrets

**Strategy 3: Training Surrogate Attack Model**

Train a model to predict class from encrypted outputs:

```python
# Collect training data: (encrypted_output, true_label) pairs
training_pairs = []
for image, label in leaked_training_data:
    # Assume attacker knows training data and labels
    secret = generate_secret(image.shape)
    bound = bind_hrr(image, secret)
    encrypted_output = main_network(bound)
    training_pairs.append((encrypted_output, label))

# Train surrogate classifier
surrogate_model = train_classifier(training_pairs)

# Test on target samples
accuracy = test(surrogate_model, test_encrypted_outputs)
```

**Expected Effectiveness**: Very Low
- This is what "Adversarial Network" f_A() in paper does
- Paper results show f_A() achieves only 1.3-2× better than random guessing
- MNIST: 19.72% (random: 10%)
- CIFAR-10: 12.91% (random: 10%)
- Network is trained to resist this attack (via gradient reversal in full version)

**Strategy 4: Statistical Analysis / Frequency Domain Attack**

Since HRR operates in frequency domain, attempt frequency-based attack:

```python
def frequency_domain_attack(bound_image, population_images):
    """
    Analyze frequency spectrum of bound image
    Compare with frequency spectra of population
    """
    # Compute FFT of bound image
    bound_freq = fft.fft2(bound_image)
    bound_magnitude = torch.abs(bound_freq)
    bound_phase = torch.angle(bound_freq)
    
    # Compute statistics over population
    population_freq = [fft.fft2(img) for img in population_images]
    pop_mag_mean = torch.mean(torch.stack([torch.abs(f) for f in population_freq]))
    pop_mag_std = torch.std(torch.stack([torch.abs(f) for f in population_freq]))
    
    # Test if bound_magnitude deviates from population
    anomaly_score = torch.abs(bound_magnitude - pop_mag_mean) / pop_mag_std
    
    return anomaly_score
```

**Expected Effectiveness**: Unknown (Not tested in paper)
- Theoretical weakness: HRR is linear in frequency domain
- Magnitude of x̂_freq = |x_freq × s_freq|
- If s has unit magnitude, |x̂_freq| ≈ |x_freq|
- Could potentially leak some information

**Defense Improvement**: Use non-unit magnitude secrets or add noise

**Strategy 5: Side-Channel Attacks**

Attack the implementation rather than cryptographic properties:

- Timing attacks: measure computation time to infer input properties
- Power analysis: if on hardware, measure power consumption
- Cache attacks: exploit CPU cache behavior
- Memory access patterns: observe memory reads/writes

**Expected Effectiveness**: Depends on implementation
- Not specific to HRR, applies to any defense
- Can be mitigated with constant-time implementations
- Out of scope for this lab

**Strategy 6: Adaptive RMIA**

Modify RMIA to work on bound images:

```python
def adaptive_rmia(bound_image, main_network, bound_population, ref_models):
    """
    Run RMIA on bound images directly
    Hypothesis: bound images from training set still distinguishable
    """
    # Compute likelihood on bound image
    Pr_x_bound = main_network(bound_image)
    
    # Compute average likelihood over reference models
    Pr_x_ref = np.mean([ref_model(bound_image) for ref_model in ref_models])
    
    # Compute LR
    LR_target = Pr_x_bound / Pr_x_ref
    
    # Compare with bound population samples
    scores = []
    for z_bound in bound_population:
        Pr_z_bound = main_network(z_bound)
        Pr_z_ref = np.mean([ref_model(z_bound) for ref_model in ref_models])
        LR_z = Pr_z_bound / Pr_z_ref
        
        if LR_target / LR_z >= gamma:
            scores.append(1)
        else:
            scores.append(0)
    
    return np.mean(scores)
```

**Expected Effectiveness**: Low to Medium
- Main network f_W() was trained to be uninformative without secret
- Paper shows clustering on r (network output) fails
- But: not directly tested against RMIA
- Worth implementing to test empirically

**Summary of Attack Strategies**:

| Attack Strategy | Expected Success | Difficulty | Countermeasure |
|-----------------|------------------|------------|----------------|
| Model Inversion | Low | High | High-dim optimization hard |
| Multiple Queries | Low-Medium | Medium | Independent secrets |
| Surrogate Model | Very Low | Low | Gradient reversal (in full version) |
| Frequency Analysis | Unknown | Medium | Non-unit magnitude secrets |
| Side-Channel | Varies | High | Constant-time implementation |
| Adaptive RMIA | Low-Medium | Medium | Output obfuscation |

**Conclusion**: While no defense is perfect, HRR shows strong empirical resistance to adaptive attacks. The main vulnerabilities are:
1. Theoretical: no formal security proofs
2. Practical: effectiveness depends on implementation details
3. Adaptive: sophisticated adversaries might find subtle leakage

For mission-critical applications, combine HRR with differential privacy or use provably secure methods (FHE, SMC) despite their cost.

---

## Implementation Checklist

### Part I: RMIA Attack

- [ ] Load CIFAR-10 dataset
- [ ] Split into training and reserved sets (for members/non-members)
- [ ] Train target ResNet-18 model
- [ ] Train 1, 2, 4, 8, 16 reference models
- [ ] Implement RMIA score computation:
  - [ ] Compute Pr(x|θ) using model confidence
  - [ ] Compute Pr(x) by averaging over reference models
  - [ ] Implement pairwise likelihood ratio
  - [ ] Compute MIA score as fraction of dominated population samples
- [ ] Implement attack evaluation:
  - [ ] Compute TPR and FPR for various thresholds β
  - [ ] Plot ROC curve
  - [ ] Calculate AUROC
  - [ ] Measure TPR at FPR = 0%, 0.01%, 0.1%
- [ ] Experiment with different numbers of reference models
- [ ] Test with class-imbalanced training data
- [ ] Document results and compare with paper

### Part II: HRR Defense

- [ ] Implement 2D HRR operations:
  - [ ] Secret generation with unit magnitude projection
  - [ ] Binding function
  - [ ] Unbinding function
- [ ] Adapt ResNet-18 architecture:
  - [ ] Option A: Add deconvolution layers for U-Net-like structure
  - [ ] Option B: Project secret to match output dimensions
- [ ] Implement training procedure:
  - [ ] Generate random secret per image
  - [ ] Bind images with secrets
  - [ ] Forward through main network
  - [ ] Unbind outputs
  - [ ] Forward through prediction network
  - [ ] Compute loss and backpropagate
- [ ] Implement inference procedure:
  - [ ] Bind image with secret locally
  - [ ] Send to main network (simulated untrusted server)
  - [ ] Unbind output locally
  - [ ] Make prediction
- [ ] Test defense against RMIA:
  - [ ] Run RMIA on defended model
  - [ ] Compare with baseline (no defense)
  - [ ] Measure attack effectiveness drop
- [ ] Analyze encryption properties
- [ ] Test potential adaptive attacks:
  - [ ] Model inversion
  - [ ] Multiple queries
  - [ ] Surrogate classifier
  - [ ] Adaptive RMIA
- [ ] Document results and conclusions

---

## Code Structure Suggestion

```
lab2/
├── data/
│   └── cifar10/          # CIFAR-10 dataset
├── models/
│   ├── target_model.py   # ResNet-18 for target
│   ├── reference_models.py  # Reference models for RMIA
│   └── hrr_defense.py    # HRR-protected ResNet-18
├── attacks/
│   ├── rmia.py           # RMIA implementation
│   └── adaptive_attacks.py  # Adaptive attacks on HRR
├── defenses/
│   ├── hrr_ops.py        # HRR binding/unbinding
│   └── train_hrr.py      # Training with HRR
├── experiments/
│   ├── exp1_rmia_baseline.py     # RMIA on undefended model
│   ├── exp2_num_references.py    # Vary # reference models
│   ├── exp3_class_imbalance.py   # Class imbalance experiments
│   ├── exp4_hrr_defense.py       # HRR defense evaluation
│   └── exp5_adaptive_attacks.py  # Test adaptive attacks
├── utils/
│   ├── metrics.py        # AUROC, TPR, FPR computation
│   ├── visualization.py  # Plot ROC curves
│   └── data_utils.py     # Data loading and preprocessing
├── results/
│   ├── figures/          # Generated plots
│   └── tables/           # Results tables
├── requirements.txt
└── README.md
```

---

## References

### Primary Papers

1. **RMIA Paper**: 
   - Zarifzadeh, S., Liu, P., & Shokri, R. (2024). "Low-Cost High-Power Membership Inference Attacks." 
   - *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*
   - arXiv: https://arxiv.org/abs/2312.03262
   - **Key Contributions**: Introduces robust membership inference attack with fine-grained hypothesis testing, achieves high power with few reference models

2. **HRR Defense Paper**:
   - Alam, M. M., Raff, E., Oates, T., & Holt, J. (2022). "Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations."
   - *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*
   - arXiv: https://arxiv.org/abs/2206.05893
   - **Key Contributions**: Adapts HRR to 2D for CNNs, creates practical pseudo-encryption defense, 5000× faster than alternatives

### Related Work on Membership Inference

3. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). "Membership Inference Attacks Against Machine Learning Models." *IEEE S&P*

4. Carlini, N., et al. (2022). "Membership Inference Attacks From First Principles." *IEEE S&P* (LiRA attack)

5. Ye, J., et al. (2022). "Enhanced Membership Inference Attacks against Machine Learning Models." *CCS* (Attack-R and Attack-P)

6. Yeom, S., et al. (2018). "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." *CSF*

### Related Work on Defenses

7. Carlini, N., et al. (2021). "Is Private Learning Possible with Instance Encoding?" *IEEE S&P* (InstaHide and its failure)

8. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." *CCS* (DP-SGD)

9. Gilad-Bachrach, R., et al. (2016). "CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy." *ICML* (Homomorphic Encryption)

### Related Work on HRR

10. Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks*

11. Ganesan, A., et al. (2021). "Learning with Holographic Reduced Representations." *NeurIPS*

### Additional Resources

12. Pytorch Documentation: https://pytorch.org/docs/
13. ResNet Paper: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*
14. CIFAR-10 Dataset: Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images."

---

## Conclusion

This lab explores the tension between model utility and privacy:

**Part I (RMIA)**: Demonstrates that even with sophisticated defenses, machine learning models can leak information about their training data. RMIA shows that membership inference attacks can be:
- Powerful (high TPR at low FPR)
- Efficient (needs only 1-4 reference models)
- Robust (works across different settings)

**Part II (HRR)**: Shows that practical obfuscation defenses are possible but come with tradeoffs:
- Not provably secure (heuristic defense)
- Practical and efficient (5000× faster than HE)
- Empirically effective (reduces attack success significantly)
- Accuracy cost (some loss in model performance)

**Key Takeaways**:
1. Privacy in ML is an ongoing arms race between attacks and defenses
2. Practical defenses must balance security, efficiency, and accuracy
3. No single defense is perfect; defense-in-depth is recommended
4. Understanding attack mechanisms is crucial for building robust defenses
5. Provable security (DP, FHE) vs practical security (HRR) have different use cases

**Future Directions**:
- Combine HRR with differential privacy for stronger guarantees
- Develop formal security proofs for HRR-like defenses
- Design more efficient homomorphic encryption schemes
- Create standardized privacy auditing frameworks
- Study privacy-utility tradeoffs more deeply

---

*This document provides theoretical foundations and implementation guidance for Practice Lab 2. Actual implementation should follow these guidelines while adapting to specific computational constraints and requirements.*

**Good luck with your implementation!**
