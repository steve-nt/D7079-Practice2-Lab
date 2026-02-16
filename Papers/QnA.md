# Practice Lab 2: Q&A Document
## Membership Inference Attacks (RMIA) and HRR Defense Mechanism

---

## Table of Contents
1. [General Questions about the Assignment](#general-questions)
2. [PART I: RMIA Implementation Questions](#part-i-rmia)
3. [PART II: HRR Defense Questions](#part-ii-hrr)
4. [Theoretical Concepts](#theoretical-concepts)
5. [Implementation Details](#implementation-details)
6. [Troubleshooting & Common Issues](#troubleshooting)

---

## General Questions about the Assignment

### Q1: What is the overall objective of this lab?
**A:** The lab has two main objectives:
- **Part I**: Implement and evaluate Robust Membership Inference Attack (RMIA) against a ResNet-18 model trained on CIFAR-10 to understand privacy risks in machine learning
- **Part II**: Implement Holographically Reduced Representations (HRR) defense mechanism and test its effectiveness against RMIA

This lab teaches you about the privacy-utility trade-off in machine learning and demonstrates both attack and defense perspectives.

### Q2: Why should we care about membership inference attacks?
**A:** Membership inference attacks reveal whether specific data points were used in training a model, which poses serious privacy risks:
- **Medical data**: Reveals if someone's health records were used
- **Financial data**: Exposes if someone's financial information was in training set
- **Personal data**: Can violate GDPR and other privacy regulations
- **Discrimination**: Can reveal sensitive attributes about individuals

Understanding these attacks is crucial for:
1. Privacy risk assessment
2. Regulatory compliance (GDPR, HIPAA)
3. Building privacy-preserving ML systems
4. Responsible AI development

### Q3: What are the key papers I need to read?
**A:** Two primary papers:
1. **RMIA Paper**: "Low-Cost High-Power Membership Inference Attacks" (Zarifzadeh et al., 2024) - https://arxiv.org/abs/2312.03262
2. **HRR Paper**: "Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations" (Alam et al., 2022) - https://arxiv.org/abs/2206.05893

### Q4: What deliverables are expected?
**A:** You need to submit:
1. **Code**: Working implementation of both RMIA attack and HRR defense
2. **Report** with three sections:
   - **Methodology**: Implementation details and experimental design
   - **Results**: Quantitative findings with visualizations (ROC curves, tables)
   - **Analysis**: Answers to all questions posed in the tasks

### Q5: Can I use AI-generated code?
**A:** Only if you can explain exactly what it does. From the assignment:
- Don't use AI-generated code unless you understand it completely
- You must be able to present the whole report
- Remove any experimental or commented-out code

---

## PART I: RMIA Implementation Questions

### Q6: What is a Membership Inference Attack (MIA)?
**A:** A membership inference attack tries to determine whether a specific data point was part of a model's training set.

**Formal Definition**: Given:
- A trained model θ
- A data point x
- Background knowledge about the population distribution π

The attack outputs: MEMBER or NON-MEMBER

**Why it works**: 
- Models typically perform better on training data
- Training data often has distinct statistical properties
- Model's confidence/loss patterns differ for members vs non-members

### Q7: How does RMIA differ from previous MIA methods?
**A:** RMIA (Robust Membership Inference Attack) improves upon previous methods:

| Feature | LiRA (Carlini 2022) | Attack-R (Ye 2022) | RMIA (Zarifzadeh 2024) |
|---------|-------------------|-------------------|---------------------|
| **Test Type** | Average-case | Population-based | Fine-grained pairwise |
| **Reference Models** | 64-256+ needed | 16-127 | 1-4 sufficient |
| **Computation Cost** | High | Medium | Low |
| **TPR @ 0% FPR** | Lower | Medium | Highest |
| **Robustness** | Unstable | Moderate | Very stable |

**Key Innovation**: RMIA tests x against **many random population samples z** rather than testing against an average null hypothesis.

### Q8: What is the mathematical foundation of RMIA?
**A:** RMIA is based on likelihood ratio tests for pairs (x, z):

**Step 1: Pairwise Likelihood Ratio**
```
LR_θ(x, z) = Pr(θ|x) / Pr(θ|z)
```

**Step 2: Apply Bayes Rule** (for black-box access):
```
LR_θ(x, z) = [Pr(x|θ) / Pr(x)] / [Pr(z|θ) / Pr(z)]
```

Where:
- **Pr(x|θ)**: Model's confidence on x (SoftMax output)
- **Pr(x)**: Average probability of x over reference models
- **Pr(z|θ)**: Model's confidence on z
- **Pr(z)**: Average probability of z over reference models

**Step 3: Compose Multiple Tests**
```
Score_MIA(x; θ) = Pr_{z~π}[LR_θ(x, z) ≥ γ]
```

This measures what fraction of random population samples z are dominated by x (i.e., x has a larger likelihood ratio than z).

### Q9: How do I implement Pr(x) computation?
**A:** Pr(x) is computed by averaging the model's output over reference models:

```python
def compute_marginal_probability(x, reference_models):
    """
    Compute Pr(x) by averaging over reference models
    
    Args:
        x: Input data point
        reference_models: List of models trained on random subsets
    
    Returns:
        Pr(x): Marginal probability of x
    """
    probabilities = []
    
    for ref_model in reference_models:
        with torch.no_grad():
            # Forward pass
            output = ref_model(x)
            # Get probability (SoftMax)
            prob = torch.softmax(output, dim=-1)
            # Get probability of true class
            prob_x = prob[x.label]
            probabilities.append(prob_x)
    
    # Average over all reference models
    Pr_x = torch.mean(torch.stack(probabilities))
    
    return Pr_x
```

**Important**: For offline attacks, use only OUT reference models (models that don't include x in training).

### Q10: What is the difference between online and offline attacks?
**A:** 

**Offline Attack**:
- All reference models are pre-trained
- Models do NOT include the target data point
- More practical and cost-effective
- Only uses OUT reference models
- RMIA excels in this setting

**Online Attack**:
- Trains reference models for each query
- Half include target data (IN models)
- Half exclude target data (OUT models)
- Very expensive (impractical for real-world use)
- Higher attack power but not worth the cost

**Recommendation**: Focus on offline attack for practical privacy auditing.

### Q11: What is Task 1.1 asking for?
**A:** Task 1.1 requires implementing the RMIA attack:

**Steps**:
1. **Data Preparation**:
   - Load CIFAR-10 dataset
   - Split into: training set + reserved set (for testing membership)
   - Reserved set should contain both known members and non-members

2. **Train Target Model**:
   - Architecture: ResNet-18
   - Train on partial CIFAR-10 data
   - Keep some data reserved for testing

3. **Train Reference Models**:
   - Train k reference models (k=1, 2, 4, etc.)
   - Each trained on random subset from population
   - These are OUT models (don't include test samples)

4. **Implement RMIA Score Calculation**:
   ```python
   def compute_rmia_score(x, target_model, ref_models, population_samples, gamma=1.0):
       # Compute Pr(x|target_model)
       Pr_x_target = get_model_confidence(x, target_model)
       
       # Compute Pr(x) from reference models
       Pr_x = compute_marginal_prob(x, ref_models)
       
       # Likelihood ratio for x
       LR_x = Pr_x_target / Pr_x
       
       # Count how many z samples x dominates
       count = 0
       for z in population_samples:
           Pr_z_target = get_model_confidence(z, target_model)
           Pr_z = compute_marginal_prob(z, ref_models)
           LR_z = Pr_z_target / Pr_z
           
           if LR_x / LR_z >= gamma:
               count += 1
       
       # MIA score is fraction of dominated samples
       score = count / len(population_samples)
       return score
   ```

5. **Make Predictions**:
   ```python
   def predict_membership(score, threshold_beta):
       if score >= threshold_beta:
           return "MEMBER"
       else:
           return "NON-MEMBER"
   ```

**Note**: You do NOT need to implement online attack mode.

### Q12: What is Task 1.2 asking for?
**A:** Task 1.2 requires experimental analysis to answer three questions:

**Question 1: How close to paper results?**
- Evaluate with FPR vs TPR curves
- Compute AUROC
- Measure TPR at specific FPR values (0%, 0.01%, 0.1%)
- Compare with paper's reported results
- Discuss any discrepancies

**Question 2: Effect of number of reference models?**
- Test with k = 1, 2, 4, 8, 16, 32, 64, 127
- Plot: number of models vs AUC
- Find "ideal number" that balances cost and performance
- Analyze diminishing returns

**Question 3: Effect of class imbalance?**
- Create imbalanced training sets (e.g., keep 100% of class 0, 50% of class 1, 10% of class 2)
- Test attack effectiveness per class
- Analyze if minority classes are more vulnerable
- Discuss implications

### Q13: How do I evaluate the attack performance?
**A:** Use multiple metrics:

**1. ROC Curve (Receiver Operating Characteristic)**:
```python
def plot_roc_curve(member_scores, non_member_scores):
    # Combine scores and labels
    scores = member_scores + non_member_scores
    labels = [1]*len(member_scores) + [0]*len(non_member_scores)
    
    # Compute FPR and TPR for different thresholds
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Plot
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
    return fpr, tpr
```

**2. AUROC (Area Under ROC Curve)**:
```python
auc_score = auc(fpr, tpr)
```
- Perfect attack: AUC = 1.0
- Random guessing: AUC = 0.5
- Higher is better

**3. TPR at Specific FPR**:
```python
def compute_tpr_at_fpr(member_scores, non_member_scores, target_fpr=0.01):
    scores = np.concatenate([member_scores, non_member_scores])
    labels = np.concatenate([np.ones(len(member_scores)), 
                            np.zeros(len(non_member_scores))])
    
    # Find threshold that gives target FPR
    sorted_non_member = np.sort(non_member_scores)[::-1]
    threshold_idx = int(target_fpr * len(non_member_scores))
    threshold = sorted_non_member[threshold_idx]
    
    # Compute TPR at this threshold
    tpr = np.mean(member_scores >= threshold)
    
    return tpr
```

**Important FPR values to report**:
- TPR @ 0% FPR (most strict, no false positives)
- TPR @ 0.01% FPR (very low, relevant for reconstruction attacks)
- TPR @ 0.1% FPR
- TPR @ 1% FPR

### Q14: What results should I expect from RMIA?
**A:** Based on the paper (CIFAR-10 models):

| # Reference Models | AUROC | TPR @ 0.01% FPR | TPR @ 0% FPR |
|-------------------|-------|-----------------|--------------|
| 0 (no ref models) | 58.19% | 0.01% | 0.0% |
| 1 | 68.64% | 1.19% | 0.51% |
| 2 | 70.13% | 1.71% | 0.91% |
| 4 | 71.02% | 2.91% | 2.13% |
| 127 | 71.71% | 4.18% | 3.14% |

**Observations**:
- Biggest jump from 0 to 1 reference model
- Diminishing returns after 4-8 models
- RMIA is very efficient (good performance with few models)

**Comparison with other attacks (with 1 reference model)**:
- LiRA: AUC = 53.2%, TPR@0%FPR = 0.25%
- Attack-R: AUC = 63.65%, TPR@0%FPR = 0.02%
- RMIA: AUC = 68.64%, TPR@0%FPR = 0.51%

### Q15: How many population samples (z) do I need?
**A:** From the paper's supplementary experiments:

| # Population Samples | AUROC | Notes |
|---------------------|-------|-------|
| 25 | 58.92% | Too few, unstable |
| 250 | 63.28% | Minimum viable |
| 1,250 | 64.75% | Good |
| 2,500 | 65.08% | Recommended (10% of training set) |
| 25,000 | 65.46% | Maximum (100% of training set) |

**Recommendation**: Use 2,500-5,000 population samples for stable results without excessive computation.

### Q16: What is the γ (gamma) parameter?
**A:** γ is the threshold for the pairwise likelihood ratio test:
```
LR_θ(x, z) ≥ γ
```

**Meaning**: How much larger must x's likelihood ratio be compared to z's to consider x as "dominating" z?

**Values**:
- γ = 1.0: x just needs to be more likely than z (recommended default)
- γ = 2.0: x needs to be 2× more likely than z (more conservative)
- γ < 1.0: Not recommended (makes test too lenient)

**Impact**: Higher γ → lower scores → lower TPR and FPR. However, paper shows RMIA is not very sensitive to γ (Figure 7 in paper).

**Recommendation**: Use γ = 1.0 or γ = 2.0. Test both and report results.

### Q17: What is the β (beta) threshold parameter?
**A:** β is the decision threshold for membership prediction:
```
if Score_MIA(x) ≥ β:
    return MEMBER
else:
    return NON-MEMBER
```

**Calibration**: The attack is calibrated such that when γ=1, the expected FPR ≈ 1 - β.

Example:
- β = 0.99 → Expected FPR ≈ 1%
- β = 0.999 → Expected FPR ≈ 0.1%
- β = 1.0 → Expected FPR ≈ 0%

**To get ROC curve**: Vary β from 0 to 1 and compute TPR and FPR at each value.

---

## PART II: HRR Defense Questions

### Q18: What is Holographic Reduced Representation (HRR)?
**A:** HRR is a method from symbolic AI that uses circular convolution to represent compositional structures. It has two key operations:

**1. Binding (⊛)**: Combines two vectors
```
B = x ⊛ s = F^(-1)[F(x) ⊙ F(s)]
```
Where:
- F() is Fourier Transform
- F^(-1)() is inverse Fourier Transform
- ⊙ is element-wise multiplication

**2. Unbinding (⊛†)**: Retrieves original vector
```
x' ≈ B ⊛ s†
```
Where s† is the inverse of s, defined as: F(s†) = 1/F(s)

**Properties**:
- Commutative: x ⊛ s = s ⊛ x
- Approximate retrieval: x' ≈ x (not perfect due to noise)
- Dimensionality preserving: B has same dimensions as x
- Random appearance: Bound result looks like random noise

### Q19: Why use 2D HRR instead of 1D HRR?
**A:** The paper extends HRR from 1D to 2D to work better with CNNs:

**1D HRR**:
- Treats image as flattened vector (32×32×3 = 3072 dimensions)
- Uses 1D FFT
- Loses spatial structure information
- Not aligned with how CNNs process images

**2D HRR**:
- Operates directly on 2D spatial dimensions
- Uses 2D FFT
- Preserves spatial structure
- Binding operation equivalent to 2D convolution
- Compatible with CNN architectures

**Implementation**:
```python
# 1D HRR (original)
def bind_1d(x, s):
    x_flat = x.flatten()
    s_flat = s.flatten()
    x_freq = fft.fft(x_flat)
    s_freq = fft.fft(s_flat)
    bound_freq = x_freq * s_freq
    bound = fft.ifft(bound_freq).real
    return bound.reshape(x.shape)

# 2D HRR (paper's approach)
def bind_2d(x, s):
    x_freq = fft.fft2(x, dim=(-2, -1))  # FFT on H, W dimensions
    s_freq = fft.fft2(s, dim=(-2, -1))
    bound_freq = x_freq * s_freq
    bound = fft.ifft2(bound_freq, dim=(-2, -1)).real
    return bound
```

### Q20: What is Task 2.1 asking for?
**A:** Implement HRR defense with these simplifications:

**Original Paper Pipeline**:
```
Input → Bind → U-Net → Unbind → Prediction Network
                ↓
        Adversarial Network (with gradient reversal)
```

**Simplified Pipeline (for this lab)**:
```
Input → Bind → ResNet-18 → Unbind → Prediction Network
```

**Simplifications**:
1. **Replace U-Net with ResNet-18**:
   - U-Net has equal input/output dimensions
   - ResNet-18 needs adaptation (add deconvolution or project secret)
   
2. **Remove Adversarial Network**:
   - Original uses gradient reversal to enforce obfuscation
   - Simplified: skip this (may reduce defense effectiveness slightly)
   
3. **Ignore Accuracy Recovery**:
   - Original averages multiple predictions with different secrets
   - Simplified: use single secret per prediction

### Q21: How do I adapt ResNet-18 for HRR?
**A:** Two approaches:

**Option A: Add Deconvolution Layers** (Recommended)
```python
class ResNet18HRR(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-18 backbone (remove final FC layer)
        self.resnet = models.resnet18(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # Output: (batch, 512, H/32, W/32)
        
        # Deconvolution to restore spatial dimensions
        self.decoder = nn.Sequential(
            # Upsample 1: (512, 1, 1) → (256, 2, 2)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Upsample 2: (256, 2, 2) → (128, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Upsample 3: (128, 4, 4) → (64, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Upsample 4: (64, 8, 8) → (32, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Upsample 5: (32, 16, 16) → (3, 32, 32) for CIFAR-10
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        features = self.resnet(x)  # (batch, 512, 1, 1)
        output = self.decoder(features)  # (batch, 3, 32, 32)
        return output
```

**Option B: Project Secret to Match Output**
```python
class ResNet18WithProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        # Modify final layer to output (512, 4, 4) instead of class logits
        self.resnet.fc = nn.Linear(512, 512*4*4)
    
    def forward(self, x):
        features = self.resnet(x)  # (batch, 512*4*4)
        output = features.view(-1, 512, 4, 4)  # Reshape
        return output

# Secret projection network
class SecretProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 512*4*4),
            nn.Unflatten(1, (512, 4, 4))
        )
    
    def forward(self, secret):
        return self.projector(secret)
```

**Recommendation**: Use Option A (deconvolution) as it's closer to the U-Net architecture.

### Q22: How do I implement HRR binding and unbinding?
**A:** Complete implementation:

```python
import torch
import torch.fft as fft
import numpy as np

def generate_secret(shape, device='cuda'):
    """
    Generate random secret with unit magnitude in frequency domain
    
    Args:
        shape: Tuple (C, H, W) for image dimensions
        device: 'cuda' or 'cpu'
    
    Returns:
        secret: Random secret tensor with unit magnitude projection
    """
    # Sample from normal distribution
    secret = torch.randn(shape, device=device)
    
    # Project to unit magnitude in frequency domain
    # This ensures good reconstruction properties
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    secret_magnitude = torch.abs(secret_freq)
    
    # Normalize to unit magnitude
    secret_freq_normalized = secret_freq / (secret_magnitude + 1e-8)
    
    # Transform back to spatial domain
    secret = fft.ifft2(secret_freq_normalized, dim=(-2, -1)).real
    
    return secret

def bind_hrr(image, secret):
    """
    Bind image with secret using 2D HRR
    
    Args:
        image: Input image tensor (C, H, W) or (B, C, H, W)
        secret: Secret tensor (same shape as image)
    
    Returns:
        bound: Bound (encrypted) image
    """
    # Apply 2D FFT to both image and secret
    image_freq = fft.fft2(image, dim=(-2, -1))
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    
    # Multiply in frequency domain (circular convolution)
    bound_freq = image_freq * secret_freq
    
    # Transform back to spatial domain
    bound = fft.ifft2(bound_freq, dim=(-2, -1)).real
    
    return bound

def unbind_hrr(bound, secret):
    """
    Unbind using secret inverse
    
    Args:
        bound: Bound image tensor
        secret: Secret tensor used for binding
    
    Returns:
        retrieved: Retrieved (decrypted) image
    """
    # Apply 2D FFT
    bound_freq = fft.fft2(bound, dim=(-2, -1))
    secret_freq = fft.fft2(secret, dim=(-2, -1))
    
    # Compute inverse in frequency domain
    # inverse of complex number: conjugate / magnitude^2
    secret_inv_freq = torch.conj(secret_freq) / (torch.abs(secret_freq)**2 + 1e-8)
    
    # Unbind (multiply with inverse)
    retrieved_freq = bound_freq * secret_inv_freq
    
    # Transform back to spatial domain
    retrieved = fft.ifft2(retrieved_freq, dim=(-2, -1)).real
    
    return retrieved

# Test the implementation
if __name__ == "__main__":
    # Test with CIFAR-10 image size
    image = torch.randn(3, 32, 32)
    secret = generate_secret((3, 32, 32))
    
    # Bind
    bound = bind_hrr(image, secret)
    print(f"Bound image looks random: mean={bound.mean():.4f}, std={bound.std():.4f}")
    
    # Unbind
    retrieved = unbind_hrr(bound, secret)
    
    # Check reconstruction error
    mse = torch.mean((image - retrieved)**2)
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Should be close to 0 for good HRR implementation")
```

### Q23: How do I train the HRR-defended model?
**A:** Training procedure:

```python
def train_hrr_model(train_loader, main_network, prediction_network, 
                    optimizer, criterion, device, num_epochs=100):
    """
    Train HRR-defended model
    
    Args:
        train_loader: DataLoader for training data
        main_network: ResNet-18 with deconvolution (on "server")
        prediction_network: Classification network (on "client")
        optimizer: Optimizer for both networks
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: 'cuda' or 'cpu'
        num_epochs: Number of training epochs
    """
    main_network.train()
    prediction_network.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Generate random secrets for each image
            secrets = torch.stack([
                generate_secret((3, 32, 32), device) 
                for _ in range(batch_size)
            ])
            
            # Bind images with secrets
            bound_images = bind_hrr(images, secrets)
            
            # Forward through main network (simulated "untrusted server")
            encrypted_outputs = main_network(bound_images)
            
            # Unbind outputs (back on "client")
            decrypted_outputs = unbind_hrr(encrypted_outputs, secrets)
            
            # Forward through prediction network
            predictions = prediction_network(decrypted_outputs)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%')

# Example usage
main_network = ResNet18HRR().to(device)
prediction_network = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512),
    nn.ReLU(),
    nn.Linear(512, 10)  # 10 classes for CIFAR-10
).to(device)

# Combine parameters for optimization
params = list(main_network.parameters()) + list(prediction_network.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
criterion = nn.CrossEntropyLoss()

train_hrr_model(train_loader, main_network, prediction_network, 
                optimizer, criterion, device, num_epochs=100)
```

### Q24: What is Task 2.2 asking for?
**A:** Answer three questions about HRR defense:

**Question 1: How effective is HRR at preventing RMIA?**
- Test RMIA attack on HRR-defended model
- Compare with baseline (undefended model)
- Metrics: AUROC, TPR@FPR
- Expected: Significant reduction in attack success

**Question 2: Does HRR qualify as encryption?**
- Analyze properties vs true encryption
- Compare with AES, RSA standards
- Discuss: provable security, key management, recovery perfection
- Argue both sides (encryption-like vs not true encryption)
- Provide nuanced conclusion

**Question 3: Could attacker adapt to overcome defense?**
- Propose potential attack strategies:
  - Model inversion
  - Multiple queries
  - Surrogate classifier
  - Frequency domain analysis
  - Adaptive RMIA
- Test at least 2-3 adaptive attacks
- Discuss success rate and defense robustness

### Q25: How do I test HRR defense against RMIA?
**A:** Experimental setup:

```python
# Scenario 1: Baseline (no defense)
target_model_baseline = train_resnet18(cifar10_train)
rmia_results_baseline = run_rmia_attack(
    target_model=target_model_baseline,
    reference_models=train_reference_models(k=4),
    member_data=member_samples,
    non_member_data=non_member_samples
)

# Scenario 2: With HRR defense
# Train HRR-defended model
main_network, prediction_network = train_hrr_defended_model(cifar10_train)

# Attack Variant 1: RMIA on bound inputs
# Attacker intercepts x̂ = x ⊛ s (bound images)
def attack_on_bound_inputs():
    member_scores = []
    non_member_scores = []
    
    for x, label in test_data:
        # Bind with random secret (attacker doesn't know original x)
        s = generate_secret(x.shape)
        x_bound = bind_hrr(x, s)
        
        # Run RMIA on bound image
        score = compute_rmia_score(
            x_bound, 
            main_network,  # Only this part visible to attacker
            reference_models,
            population_samples
        )
        
        if label == 'member':
            member_scores.append(score)
        else:
            non_member_scores.append(score)
    
    return evaluate_attack(member_scores, non_member_scores)

# Attack Variant 2: RMIA on encrypted outputs
# Attacker observes r = f_W(x̂) and tries membership inference
def attack_on_encrypted_outputs():
    member_scores = []
    non_member_scores = []
    
    for x, label in test_data:
        s = generate_secret(x.shape)
        x_bound = bind_hrr(x, s)
        
        # Get encrypted output (what attacker sees)
        encrypted_output = main_network(x_bound)
        
        # Try to use encrypted output for RMIA
        # Compute "likelihood" based on encrypted output statistics
        score = compute_score_from_encrypted_output(
            encrypted_output,
            reference_encrypted_outputs
        )
        
        if label == 'member':
            member_scores.append(score)
        else:
            non_member_scores.append(score)
    
    return evaluate_attack(member_scores, non_member_scores)

# Compare results
print("Baseline Attack (no defense):")
print(f"  AUROC: {rmia_results_baseline['auc']:.4f}")
print(f"  TPR @ 0% FPR: {rmia_results_baseline['tpr_0']:.4f}")

print("\nAttack on Bound Inputs:")
results_bound = attack_on_bound_inputs()
print(f"  AUROC: {results_bound['auc']:.4f}")
print(f"  TPR @ 0% FPR: {results_bound['tpr_0']:.4f}")

print("\nAttack on Encrypted Outputs:")
results_encrypted = attack_on_encrypted_outputs()
print(f"  AUROC: {results_encrypted['auc']:.4f}")
print(f"  TPR @ 0% FPR: {results_encrypted['tpr_0']:.4f}")
```

**Expected Results** (based on HRR paper):
- Baseline AUROC: ~70%
- With HRR AUROC: ~50% (random guessing)
- Defense effectiveness: Significant reduction

### Q26: Does HRR qualify as true encryption?
**A:** **Short Answer: No**, it's "pseudo-encryption" or "obfuscation", not true encryption.

**Detailed Analysis**:

**Arguments FOR encryption-like properties**:
1. ✓ One-time pad analogy (new secret per use)
2. ✓ Visual randomness (bound images look like noise)
3. ✓ Reversibility (can recover with secret)
4. ✓ Empirical security (attacks fail in practice)

**Arguments AGAINST true encryption**:
1. ✗ No provable security (heuristic only)
2. ✗ Approximate recovery (x' ≈ x, not x' = x)
3. ✗ Not uniformly random (statistical patterns exist)
4. ✗ Linear operation (potentially vulnerable to algebraic attacks)
5. ✗ No cryptographic certification

**Conclusion**: 
- Use HRR when: performance critical, privacy important but not mission-critical
- DON'T use HRR for: medical records, financial data, regulated data, highly sensitive information
- For mission-critical: Use differential privacy or homomorphic encryption

**Paper's Own Statement**: "We emphatically stress this is not strong encryption, but empirically we observe a realistic adversary's attacks are at random-guessing performance."

---

## Theoretical Concepts

### Q27: What is the threat model for MIA?
**A:** The adversary (attacker) has:

**Access**:
- Black-box query access to target model
- Knowledge of model architecture
- Sample from population distribution π
- (Sometimes) Access to training algorithm details

**Goal**:
- Determine if specific data point x was in training set

**Does NOT have**:
- Model weights (white-box attacks are separate category)
- Training data itself
- Unlimited query budget (costs money in practice)

**Realistic Assumption**: The attacker is a privacy auditor or researcher, not a malicious hacker with unlimited resources.

### Q28: Why does membership leakage occur?
**A:** Several reasons:

1. **Overfitting**: Models memorize training data
   - Training loss << Test loss
   - Model fits training data "too well"
   
2. **Statistical Differences**: 
   - Training data has distinct distribution
   - Models learn these specific patterns
   
3. **Gradient-Based Learning**:
   - SGD updates based on training samples
   - Leaves "signatures" of training data
   
4. **High Model Capacity**:
   - Large networks (like ResNet-18) can memorize
   - More parameters → more memorization potential

5. **Uneven Learning**:
   - Some samples harder to learn (outliers)
   - These "memorable" samples are easier to identify

### Q29: What are the privacy implications?
**A:** 

**Individual Privacy**:
- Reveals someone's data was used
- Can expose sensitive attributes
- Violates privacy expectations

**Regulatory Compliance**:
- GDPR: Right to be forgotten
- HIPAA: Protected health information
- CCPA: Consumer privacy rights

**Legal Risks**:
- Lawsuits for privacy violations
- Regulatory fines
- Reputational damage

**Technical Implications**:
- Need for privacy-preserving ML
- Differential privacy mechanisms
- Secure multi-party computation

### Q30: What is differential privacy and how does it relate?
**A:** **Differential Privacy (DP)** provides formal privacy guarantees:

**Definition**: A randomized algorithm A is ε-differentially private if for any two datasets D and D' differing by one record, and for any output set S:

```
Pr[A(D) ∈ S] ≤ e^ε × Pr[A(D') ∈ S]
```

**Key Points**:
- ε (epsilon) controls privacy level (smaller = more private)
- Provides mathematical guarantee against all attacks
- Adds calibrated noise during training

**DP-SGD Algorithm**:
```python
def dp_sgd_step(model, batch, epsilon, delta):
    # Compute gradients
    gradients = compute_gradients(model, batch)
    
    # Clip gradients to bound sensitivity
    clipped_grads = clip_gradients(gradients, max_norm=C)
    
    # Add Gaussian noise
    noise = sample_gaussian_noise(sigma, dimension)
    private_grads = clipped_grads + noise
    
    # Update model
    model.update(private_grads)
```

**Relation to MIA**:
- DP provides protection against MIA
- But: accuracy-privacy tradeoff
- DP-SGD can be slow and reduce accuracy
- HRR offers alternative (heuristic) protection

### Q31: What are related attacks?
**A:** 

**1. Model Inversion Attack**:
- Goal: Reconstruct training data from model
- Example: Recover face images from face recognition model
- More severe than MIA (gets actual data, not just membership)

**2. Attribute Inference Attack**:
- Goal: Infer sensitive attributes of training data
- Example: Infer gender, race from model behavior
- Can work even without membership knowledge

**3. Property Inference Attack**:
- Goal: Infer properties of training dataset
- Example: "Was training data from hospital X?"
- Aggregate information rather than individual

**4. Data Extraction Attack**:
- Goal: Extract verbatim training data
- Common in language models (GPT, etc.)
- Model "memorizes" specific training examples

**Comparison**:
```
Attack Type          | Severity | Difficulty | Defense
---------------------|----------|------------|----------
Membership Inference | Medium   | Easy       | DP, HRR
Model Inversion      | High     | Medium     | DP, Output perturbation
Attribute Inference  | Medium   | Medium     | DP, Fairness constraints
Data Extraction      | Very High| Easy       | DP, Deduplication
```

---

## Implementation Details

### Q32: What PyTorch version and libraries do I need?
**A:** 

**Required**:
```
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
matplotlib >= 3.4.0
scikit-learn >= 0.24.0
```

**Installation**:
```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

**Imports for RMIA**:
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
```

**Imports for HRR**:
```python
import torch
import torch.fft as fft  # Important: use torch.fft, not numpy.fft
import torch.nn.functional as F
```

### Q33: How do I structure my code?
**A:** Recommended structure:

```
lab2/
├── data/
│   └── cifar10/              # Downloaded automatically
├── models/
│   ├── resnet.py             # Standard ResNet-18
│   ├── resnet_hrr.py         # HRR-adapted ResNet-18
│   └── prediction_net.py     # Prediction network for HRR
├── attacks/
│   ├── rmia.py               # RMIA implementation
│   ├── utils.py              # Helper functions (Pr computation, etc.)
│   └── adaptive_attacks.py   # Adaptive attacks for Part II
├── defenses/
│   ├── hrr.py                # HRR operations (bind, unbind, generate_secret)
│   └── train_hrr.py          # Training procedure for HRR
├── experiments/
│   ├── train_models.py       # Train target and reference models
│   ├── run_rmia.py           # Run RMIA experiments
│   ├── test_hrr.py           # Test HRR defense
│   └── analyze_results.py    # Generate plots and tables
├── utils/
│   ├── data_loader.py        # CIFAR-10 loading with splits
│   ├── metrics.py            # AUROC, TPR, FPR calculations
│   └── visualization.py      # ROC curves, bound images
├── results/
│   ├── figures/              # Generated plots
│   └── tables/               # Results as CSVs
├── configs/
│   └── config.py             # Hyperparameters
├── main_part1.py             # Run Part I experiments
├── main_part2.py             # Run Part II experiments
└── README.md
```

### Q34: How do I split CIFAR-10 for the experiments?
**A:** 

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

def load_cifar10_with_split(data_dir='./data', train_size=25000, 
                            reserved_size=25000, seed=42):
    """
    Load CIFAR-10 and split for MIA experiments
    
    Args:
        data_dir: Directory to store data
        train_size: Number of samples for training target model
        reserved_size: Number of samples reserved for testing membership
        seed: Random seed for reproducibility
    
    Returns:
        train_data: For training target model (known members)
        reserved_members: Known members not in training
        reserved_non_members: Known non-members
        test_data: Standard test set
    """
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full training set
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    # Load test set
    test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Split training set
    total_train = len(full_train)  # 50,000
    indices = np.random.permutation(total_train)
    
    # Indices for different splits
    train_indices = indices[:train_size]
    reserved_member_indices = indices[train_size:train_size + reserved_size]
    
    # Create subsets
    train_data = Subset(full_train, train_indices)
    reserved_members = Subset(full_train, reserved_member_indices)
    
    # Use test set as non-members (they were never seen during training)
    reserved_non_members = test_data
    
    return train_data, reserved_members, reserved_non_members, test_data

# Example usage
train_data, reserved_members, reserved_non_members, test_data = \
    load_cifar10_with_split(train_size=25000, reserved_size=5000)

print(f"Training data: {len(train_data)} samples")
print(f"Reserved members: {len(reserved_members)} samples")
print(f"Reserved non-members: {len(reserved_non_members)} samples")
print(f"Test data: {len(test_data)} samples")

# Create data loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
member_loader = DataLoader(reserved_members, batch_size=100, shuffle=False, num_workers=4)
non_member_loader = DataLoader(reserved_non_members, batch_size=100, shuffle=False, num_workers=4)
```

### Q35: How do I train ResNet-18 on CIFAR-10?
**A:** 

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def train_resnet18(train_loader, num_epochs=100, device='cuda', 
                  lr=0.1, momentum=0.9, weight_decay=5e-4):
    """
    Train ResNet-18 on CIFAR-10
    
    Args:
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
    
    Returns:
        model: Trained ResNet-18 model
    """
    # Initialize model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                         momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Step scheduler
        scheduler.step()
        
        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%')
    
    return model

# Train target model
target_model = train_resnet18(
    train_loader, 
    num_epochs=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Save model
torch.save(target_model.state_dict(), 'checkpoints/target_model.pth')

# Test accuracy
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

test_accuracy = test_model(target_model, test_loader, device)
```

### Q36: How do I train reference models efficiently?
**A:** 

```python
def train_reference_models(num_models, train_size=25000, 
                          population_data=None, device='cuda',
                          save_dir='./reference_models'):
    """
    Train multiple reference models for RMIA
    
    Args:
        num_models: Number of reference models to train
        train_size: Size of each training set
        population_data: Full population data to sample from
        device: 'cuda' or 'cpu'
        save_dir: Directory to save models
    
    Returns:
        reference_models: List of trained models
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    reference_models = []
    
    for i in range(num_models):
        print(f"\n{'='*50}")
        print(f"Training Reference Model {i+1}/{num_models}")
        print(f"{'='*50}")
        
        # Sample random subset from population
        indices = np.random.choice(
            len(population_data), 
            size=train_size, 
            replace=False
        )
        ref_train_data = Subset(population_data, indices)
        ref_train_loader = DataLoader(
            ref_train_data, 
            batch_size=128, 
            shuffle=True, 
            num_workers=4
        )
        
        # Train model
        ref_model = train_resnet18(
            ref_train_loader,
            num_epochs=100,  # Can reduce for faster training
            device=device
        )
        
        # Save model
        model_path = os.path.join(save_dir, f'ref_model_{i}.pth')
        torch.save(ref_model.state_dict(), model_path)
        print(f"Saved to {model_path}")
        
        reference_models.append(ref_model)
    
    return reference_models

# Example: Train 4 reference models
reference_models = train_reference_models(
    num_models=4,
    train_size=25000,
    population_data=full_train_dataset,
    device='cuda'
)
```

**Optimization Tip**: Train reference models in parallel if you have multiple GPUs:
```python
import torch.multiprocessing as mp

def train_single_ref_model(gpu_id, model_id, train_data):
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    model = train_resnet18(train_data, device=device)
    torch.save(model.state_dict(), f'ref_model_{model_id}.pth')

# Spawn processes (assuming 4 GPUs)
processes = []
for i in range(4):
    p = mp.Process(target=train_single_ref_model, args=(i, i, train_data))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

---

## Troubleshooting & Common Issues

### Q37: My RMIA attack has AUC close to 0.5 (random guessing). What's wrong?
**A:** Common issues:

**Problem 1: Wrong Pr(x) Computation**
```python
# WRONG: Using target model for Pr(x)
Pr_x = target_model(x).softmax(dim=-1)[label]

# CORRECT: Average over reference models
Pr_x = np.mean([ref_model(x).softmax(dim=-1)[label] 
                for ref_model in reference_models])
```

**Problem 2: Not Using SoftMax**
```python
# WRONG: Using raw logits
Pr_x = target_model(x)[label]

# CORRECT: Use SoftMax probabilities
Pr_x = target_model(x).softmax(dim=-1)[label]
```

**Problem 3: Threshold Not Calibrated**
```python
# Make sure to vary β from 0 to 1 for ROC curve
thresholds = np.linspace(0, 1, 1000)
for beta in thresholds:
    # Compute TPR and FPR at this threshold
    predictions = (scores >= beta)
    ...
```

**Problem 4: Too Few Population Samples**
```python
# Use at least 2,500 population samples
num_population = 2500  # Not 100
```

### Q38: My HRR reconstruction has high MSE. Is this normal?
**A:** 

**Normal MSE Range**:
- Good HRR: MSE < 0.01
- Acceptable: MSE < 0.1
- Too high: MSE > 1.0

**If MSE is too high, check**:

1. **Secret Generation**:
```python
# Make sure you're using unit magnitude projection
secret_freq = fft.fft2(secret, dim=(-2, -1))
secret_freq = secret_freq / (torch.abs(secret_freq) + 1e-8)
secret = fft.ifft2(secret_freq, dim=(-2, -1)).real
```

2. **Unbinding Formula**:
```python
# Correct inverse
secret_inv_freq = torch.conj(secret_freq) / (torch.abs(secret_freq)**2 + 1e-8)

# NOT this (wrong)
secret_inv_freq = 1.0 / secret_freq
```

3. **FFT Dimensions**:
```python
# For (C, H, W) images, FFT over last two dims
fft.fft2(image, dim=(-2, -1))

# NOT this (over channel dimension)
fft.fft2(image, dim=(0, 1))
```

### Q39: My HRR-defended model has very low accuracy. How to fix?
**A:** 

**Expected Accuracy Drop**:
- Baseline ResNet-18 on CIFAR-10: ~93%
- With HRR defense: ~78-83% (10-15% drop is normal)
- With averaging k=10 secrets: ~84-88%

**If accuracy is below 70%**:

1. **Check Unbinding**:
```python
# Test unbinding quality
image = train_data[0][0]
secret = generate_secret(image.shape)
bound = bind_hrr(image, secret)
retrieved = unbind_hrr(bound, secret)
mse = ((image - retrieved)**2).mean()
print(f"Unbinding MSE: {mse}")  # Should be < 0.01
```

2. **Check Network Architecture**:
- Make sure output of main network matches input dimensions
- For CIFAR-10: input (3, 32, 32) → output (3, 32, 32)

3. **Train Longer**:
```python
# HRR models need more epochs
num_epochs = 150  # Instead of 100
```

4. **Use Learning Rate Warmup**:
```python
# Start with lower LR
for epoch in range(10):
    lr = 0.01 * (epoch + 1) / 10  # Warmup
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### Q40: How long should training take?
**A:** 

**Typical Training Times** (on single NVIDIA RTX 3090):

| Task | Time | Notes |
|------|------|-------|
| Target ResNet-18 (100 epochs) | ~2 hours | CIFAR-10, batch size 128 |
| Reference model (100 epochs) | ~2 hours each | Same as target |
| 4 reference models | ~8 hours | Can parallelize |
| HRR-defended model (150 epochs) | ~4 hours | Slower due to FFT ops |
| RMIA evaluation | ~30 mins | For 5000 test samples |
| Total for Part I | ~10-12 hours | Serial execution |
| Total for Part II | ~6-8 hours | Serial execution |

**Speed-Up Tips**:
1. Use mixed precision training: `torch.cuda.amp`
2. Increase batch size (if memory allows)
3. Train reference models in parallel (multiple GPUs)
4. Use fewer epochs for reference models (80 instead of 100)
5. Cache model outputs to avoid recomputation

### Q41: My code is running out of memory. How to fix?
**A:** 

**Memory-Saving Strategies**:

1. **Reduce Batch Size**:
```python
# If OOM with batch_size=128
batch_size = 64  # or 32
```

2. **Use Gradient Accumulation**:
```python
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Clear Cache**:
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

4. **Use torch.no_grad() for Evaluation**:
```python
with torch.no_grad():  # Don't store gradients
    for images, labels in test_loader:
        outputs = model(images)
        ...
```

5. **Process in Smaller Batches**:
```python
# For RMIA evaluation, process 100 samples at a time
for i in range(0, len(test_data), 100):
    batch = test_data[i:i+100]
    scores = compute_rmia_scores(batch)
    all_scores.extend(scores)
```

### Q42: How do I debug my RMIA implementation?
**A:** 

**Step-by-Step Debugging**:

1. **Verify Model Outputs**:
```python
# Test target model
x = test_data[0][0].unsqueeze(0).to(device)
output = target_model(x)
prob = output.softmax(dim=-1)
print(f"Output shape: {output.shape}")  # Should be (1, 10)
print(f"Probabilities sum: {prob.sum()}")  # Should be ~1.0
print(f"Max probability: {prob.max()}")  # Should be 0.5-0.99
```

2. **Check Reference Models**:
```python
# Reference models should give different outputs
outputs_ref = [ref_model(x).softmax(dim=-1) for ref_model in reference_models]
print(f"Number of ref models: {len(outputs_ref)}")
print(f"Output variance: {torch.stack(outputs_ref).var()}")  # Should be > 0
```

3. **Verify Likelihood Ratios**:
```python
# LR should be different for members vs non-members
member_x = member_data[0][0].unsqueeze(0).to(device)
non_member_x = non_member_data[0][0].unsqueeze(0).to(device)

lr_member = compute_likelihood_ratio(member_x, target_model, reference_models)
lr_non_member = compute_likelihood_ratio(non_member_x, target_model, reference_models)

print(f"LR (member): {lr_member}")  # Should be > 1.0
print(f"LR (non-member): {lr_non_member}")  # Often < 1.0
```

4. **Check Score Distribution**:
```python
# Scores should differ for members vs non-members
member_scores = [compute_rmia_score(x, ...) for x in member_samples[:100]]
non_member_scores = [compute_rmia_score(x, ...) for x in non_member_samples[:100]]

print(f"Member scores: mean={np.mean(member_scores):.3f}, std={np.std(member_scores):.3f}")
print(f"Non-member scores: mean={np.mean(non_member_scores):.3f}, std={np.std(non_member_scores):.3f}")

# Member mean should be > Non-member mean
```

5. **Visualize Score Distributions**:
```python
plt.hist(member_scores, bins=50, alpha=0.5, label='Members')
plt.hist(non_member_scores, bins=50, alpha=0.5, label='Non-members')
plt.xlabel('RMIA Score')
plt.ylabel('Frequency')
plt.legend()
plt.title('Score Distribution')
plt.show()
```

### Q43: How do I create good visualizations for my report?
**A:** 

**Essential Plots**:

**1. ROC Curve**:
```python
def plot_roc_curve(results_dict, save_path='roc_curve.pdf'):
    """
    Plot ROC curves for multiple attacks
    
    Args:
        results_dict: Dictionary mapping attack name to (fpr, tpr)
    """
    plt.figure(figsize=(8, 6))
    
    for name, (fpr, tpr) in results_dict.items():
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: RMIA Performance', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**2. Number of Models vs Performance**:
```python
def plot_num_models_vs_performance(num_models_list, auc_scores, 
                                  tpr_scores, save_path='models_vs_perf.pdf'):
    """
    Plot how attack performance varies with number of reference models
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUC vs num models
    ax1.plot(num_models_list, auc_scores, marker='o', linewidth=2)
    ax1.set_xlabel('Number of Reference Models', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('AUROC vs Number of Reference Models', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # TPR @ 0% FPR vs num models
    ax2.plot(num_models_list, tpr_scores, marker='s', linewidth=2, color='red')
    ax2.set_xlabel('Number of Reference Models', fontsize=12)
    ax2.set_ylabel('TPR @ 0% FPR', fontsize=12)
    ax2.set_title('TPR at 0% FPR vs Number of Reference Models', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**3. Class Imbalance Analysis**:
```python
def plot_class_imbalance_results(class_labels, member_tpr, non_member_fpr,
                                sampling_ratios, save_path='class_imbalance.pdf'):
    """
    Show how attack performance varies by class with imbalance
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_labels))
    width = 0.35
    
    ax.bar(x - width/2, member_tpr, width, label='TPR (Members)', alpha=0.8)
    ax.bar(x + width/2, non_member_fpr, width, label='FPR (Non-members)', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Attack Performance by Class (with imbalance)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend(fontsize=10)
    
    # Add sampling ratios as text
    for i, ratio in enumerate(sampling_ratios):
        ax.text(i, -0.05, f'{ratio:.1%}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**4. HRR Bound Images Visualization**:
```python
def visualize_hrr_binding(original_images, save_path='hrr_visualization.pdf'):
    """
    Show original, bound, and retrieved images
    """
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    
    for i in range(5):
        # Original
        img = original_images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # Bound
        secret = generate_secret(original_images[i].shape)
        bound = bind_hrr(original_images[i], secret)
        bound_img = bound.cpu().numpy().transpose(1, 2, 0)
        bound_img = (bound_img - bound_img.min()) / (bound_img.max() - bound_img.min())
        axes[1, i].imshow(bound_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Bound (Encrypted)', fontsize=12)
        
        # Retrieved
        retrieved = unbind_hrr(bound, secret)
        retrieved_img = retrieved.cpu().numpy().transpose(1, 2, 0)
        retrieved_img = (retrieved_img - retrieved_img.min()) / (retrieved_img.max() - retrieved_img.min())
        axes[2, i].imshow(retrieved_img)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Retrieved', fontsize=12)
    
    plt.suptitle('HRR Binding and Unbinding Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Final Tips

### Q44: What are the most common mistakes students make?
**A:** 

**Top 10 Mistakes**:

1. ✗ Not using SoftMax for probabilities
2. ✗ Computing Pr(x) from target model instead of reference models
3. ✗ Using too few population samples (< 1000)
4. ✗ Not setting random seeds (results not reproducible)
5. ✗ Forgetting to call `model.eval()` during testing
6. ✗ Using wrong dimensions for 2D FFT
7. ✗ Not normalizing CIFAR-10 images properly
8. ✗ Plotting ROC in wrong scale (should be log-scale for FPR)
9. ✗ Not saving models (have to retrain everything)
10. ✗ Leaving experimental code in final submission

### Q45: How should I structure my report?
**A:** 

**Report Structure**:

**1. Methodology (2-3 pages)**:
- Overview of RMIA attack
- Implementation details:
  - Data splits
  - Model architectures
  - Training procedures
  - Attack algorithm
- Overview of HRR defense
- Implementation details:
  - HRR operations
  - Network modifications
  - Training procedure
- Experimental design:
  - Metrics used
  - Number of runs
  - Hyperparameters

**2. Results (3-4 pages)**:
- Part I Results:
  - Table: RMIA performance vs paper
  - Figure: ROC curves
  - Figure: Number of models vs performance
  - Figure: Class imbalance effects
  - Table: Detailed metrics (AUC, TPR@FPR)
- Part II Results:
  - Table: HRR defense effectiveness
  - Figure: ROC comparison (with/without defense)
  - Figure: Bound image examples
  - Table: Adaptive attack results

**3. Analysis (2-3 pages)**:
- Answer all questions from tasks:
  - Task 1.2: Q1, Q2, Q3
  - Task 2.2: Q1, Q2, Q3
- Discuss discrepancies from paper
- Explain unexpected results
- Provide insights and observations

**Total: 7-10 pages + code**

### Q46: What should I include in my code submission?
**A:** 

**Code Requirements**:
1. ✓ All source code files (organized in folders)
2. ✓ README.md with:
   - Setup instructions
   - How to run experiments
   - Expected outputs
   - Dependencies
3. ✓ requirements.txt with all dependencies
4. ✓ Pre-trained model checkpoints (if possible)
5. ✓ Scripts to reproduce all results
6. ✓ Configuration files (hyperparameters)
7. ✓ NO commented-out experimental code
8. ✓ NO AI-generated code you can't explain

**Code Quality**:
- Clear variable names
- Docstrings for functions
- Comments for complex logic
- Type hints where appropriate
- Follows PEP 8 style guide

### Q47: How do I make my results reproducible?
**A:** 

```python
def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of all scripts
set_random_seeds(42)
```

**Also Include**:
- Python version
- PyTorch version
- CUDA version
- GPU model
- Exact hyperparameters used
- Training time and convergence

---

## Summary Checklist

### Part I Checklist:
- [ ] Load and split CIFAR-10 correctly
- [ ] Train target ResNet-18 model
- [ ] Train 1, 2, 4, (8, 16) reference models
- [ ] Implement Pr(x) computation from reference models
- [ ] Implement pairwise likelihood ratio calculation
- [ ] Implement RMIA score computation
- [ ] Evaluate attack: compute ROC curve, AUROC, TPR@FPR
- [ ] Test different numbers of reference models
- [ ] Test with class-imbalanced training data
- [ ] Compare results with paper
- [ ] Create all visualizations
- [ ] Write methodology section
- [ ] Write results section
- [ ] Answer all three questions in Task 1.2

### Part II Checklist:
- [ ] Implement HRR binding operation (2D FFT)
- [ ] Implement HRR unbinding operation
- [ ] Implement secret generation with unit magnitude
- [ ] Adapt ResNet-18 (add deconv or project secret)
- [ ] Implement prediction network
- [ ] Train HRR-defended model
- [ ] Test HRR defense against RMIA
- [ ] Visualize bound/retrieved images
- [ ] Test at least 2 adaptive attacks
- [ ] Analyze: Does HRR qualify as encryption?
- [ ] Write methodology section
- [ ] Write results section
- [ ] Answer all three questions in Task 2.2

### Report Checklist:
- [ ] Methodology section complete (2-3 pages)
- [ ] Results section complete with figures/tables (3-4 pages)
- [ ] Analysis section answers all questions (2-3 pages)
- [ ] All figures have captions
- [ ] All tables have captions
- [ ] References cited properly
- [ ] No spelling/grammar errors
- [ ] PDF formatted nicely

### Code Checklist:
- [ ] Code organized in folders
- [ ] README.md with instructions
- [ ] requirements.txt included
- [ ] All experimental code removed
- [ ] Code has comments and docstrings
- [ ] Can reproduce all results
- [ ] Random seeds set for reproducibility

---

**Good luck with your implementation! If you have more questions, refer back to the papers and this document.**
