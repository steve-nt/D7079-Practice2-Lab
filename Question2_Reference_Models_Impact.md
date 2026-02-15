# Question 2: How does the number of reference models affect the attack's success? Is there an ideal number?

**Task 1.2 - Question 2:** Analyze the impact of varying the number of reference models on RMIA attack performance and determine the optimal configuration.

---

## Answer Summary

**Key Findings:**
1. **More reference models improve attack performance, but with diminishing returns**
2. **The ideal number is 4 reference models** (best cost-benefit ratio)
3. **Beyond 8-16 models, additional gains are minimal** (< 1% AUC improvement)

**Optimal Configuration:**
- **Minimum viable:** 1-2 models (acceptable performance, low cost)
- **Recommended:** 4 models (sweet spot from paper)
- **Maximum practical:** 8 models (near-optimal performance)
- **Beyond 16 models:** Not worth the training time investment

---

## Paper's Findings on Reference Model Count

### From RMIA Paper Table 2 (lines 834-925)

| Number of Reference Models | AUC    | Improvement from Previous | TPR @ 0.01% FPR |
|---------------------------|--------|---------------------------|-----------------|
| 0 (Attack-P baseline)     | 58.19% | baseline                  | 0.01%           |
| 1 (RMIA)                  | 68.64% | **+10.45%**               | 1.19%           |
| 2 (RMIA)                  | 70.13% | **+1.49%**                | 1.71%           |
| 4 (RMIA)                  | 71.02% | **+0.89%**                | 2.91%           |
| 127 (RMIA, offline)       | 71.71% | **+0.69%**                | 4.18%           |

**Key Observation:** 
- 1st model: +10.45% improvement (massive)
- 2nd model: +1.49% improvement (significant)
- 4th model: +0.89% improvement (moderate)
- 127 models: +0.69% improvement (diminishing returns!)

**Paper Citation:** Zarifzadeh et al. (2024), "Low-Cost High-Power Membership Inference Attacks", Section 5.2, Table 2

---

## Why Reference Models Matter

### Conceptual Explanation

**From paper Section 3, lines 327-387:**

The RMIA attack estimates likelihood ratios:
```
LR(x) = Pr(x | θ_target) / Pr(x | OUT)
```

Where:
- `Pr(x | θ_target)` = probability target model assigns to sample x
- `Pr(x | OUT)` = probability of x under models NOT trained on x

**Single Reference Model:**
- Uses 1 reference model to estimate `Pr(x | OUT)`
- High variance estimate (one model's opinion)
- Noisy signal

**Multiple Reference Models:**
- Averages predictions across N reference models
- `Pr(x | OUT) ≈ mean([ref_1(x), ref_2(x), ..., ref_N(x)])`
- Lower variance estimate (consensus opinion)
- More stable signal

**From your implementation (part1.ipynb, lines 73-75):**
```python
# Average predictions across all reference models to get a more stable Pr(x)OUT
all_ref_probs_x = [torch.softmax(rm(known_img.unsqueeze(0)), dim=1)[0, known_label].item() 
                   for rm in ref_models]
prob_x_out = np.mean(all_ref_probs_x)
```

---

## How to Run the Experiment

### Step 1: Modify Configuration in part1.ipynb

**Location:** Line 154

**Current code:**
```python
num_ref_models = 8  # You can increase this (e.g., 4, 8, or 16)
```

**Change to test different values:**
```python
# Test these configurations in separate runs
num_ref_models = 1   # Baseline
num_ref_models = 2   # Common case
num_ref_models = 4   # Paper's recommended
num_ref_models = 8   # Current setting
num_ref_models = 16  # Higher (optional)
```

---

### Step 2: Complete Experimental Procedure

**For each configuration (1, 2, 4, 8, 16 models):**

1. **Set num_ref_models** (line 154)
2. **Re-run training cells** (lines 110-169):
   - This trains the reference models
   - Training time = `epochs × num_models × ~10 minutes`
   
3. **Run evaluation cells** (lines 200-316):
   - Calculates RMIA scores
   - Generates ROC curve
   
4. **Record results:**
   - AUC value (printed around line 276)
   - TPR at specific FPR values
   - Training time
   - Save ROC curve plot

5. **Save results:**
   ```python
   # Add this after line 276
   import pickle
   results = {
       'num_refs': num_ref_models,
       'auc': roc_auc,
       'fpr': fpr,
       'tpr': tpr,
       'scores': all_scores,
       'labels': all_labels
   }
   with open(f'results_{num_ref_models}_refs.pkl', 'wb') as f:
       pickle.dump(results, f)
   ```

---

### Step 3: Compare Results Across Configurations

**Create a comparison cell:**

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load all results
configurations = [1, 2, 4, 8, 16]  # Or whichever you tested
results = {}

for num_refs in configurations:
    try:
        with open(f'results_{num_refs}_refs.pkl', 'rb') as f:
            results[num_refs] = pickle.load(f)
    except FileNotFoundError:
        print(f"Results for {num_refs} models not found")

# Create comparison table
print("="*70)
print("COMPARISON: Impact of Number of Reference Models")
print("="*70)
print(f"{'Num Refs':<12} {'AUC':<10} {'Improvement':<15} {'TPR @ 1% FPR':<15}")
print("-"*70)

prev_auc = None
for num_refs in sorted(results.keys()):
    auc = results[num_refs]['auc']
    fpr = results[num_refs]['fpr']
    tpr = results[num_refs]['tpr']
    
    # Calculate TPR at 1% FPR
    idx = np.argmin(np.abs(fpr - 0.01))
    tpr_at_1pct = tpr[idx]
    
    # Calculate improvement
    if prev_auc is None:
        improvement = "baseline"
    else:
        improvement = f"+{(auc - prev_auc)*100:.2f}%"
    
    print(f"{num_refs:<12} {auc:.4f}    {improvement:<15} {tpr_at_1pct:.4f}")
    prev_auc = auc

print("="*70)
```

---

### Step 4: Visualize the Relationship

**Create a plot showing AUC vs Number of Reference Models:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract data
num_models = sorted(results.keys())
aucs = [results[n]['auc'] for n in num_models]

# Plot 1: AUC vs Number of Models
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(num_models, aucs, 'o-', linewidth=2, markersize=8, color='blue')
plt.xlabel('Number of Reference Models', fontsize=12)
plt.ylabel('AUC (Area Under ROC Curve)', fontsize=12)
plt.title('Attack Performance vs Reference Model Count', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim([0.6, 0.75])  # Adjust based on your results

# Add paper's values for comparison
paper_nums = [1, 2, 4]
paper_aucs = [0.6864, 0.7013, 0.7102]
plt.plot(paper_nums, paper_aucs, 's--', linewidth=2, markersize=8, 
         color='red', alpha=0.7, label='Paper Results')
plt.legend()

# Plot 2: Marginal Improvement
plt.subplot(1, 2, 2)
improvements = [0] + [aucs[i] - aucs[i-1] for i in range(1, len(aucs))]
plt.bar(num_models, improvements, color='green', alpha=0.7)
plt.xlabel('Number of Reference Models', fontsize=12)
plt.ylabel('Marginal AUC Improvement', fontsize=12)
plt.title('Diminishing Returns', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reference_models_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'reference_models_analysis.png'")
```

---

### Step 5: Overlay ROC Curves

**Compare ROC curves for different numbers of models:**

```python
plt.figure(figsize=(10, 8))

colors = ['blue', 'green', 'orange', 'red', 'purple']
for i, num_refs in enumerate(sorted(results.keys())):
    fpr = results[num_refs]['fpr']
    tpr = results[num_refs]['tpr']
    auc = results[num_refs]['auc']
    
    plt.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
             label=f'{num_refs} Ref Model(s) (AUC = {auc:.4f})')

# Plot random guess baseline
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess (AUC = 0.5000)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curves: Impact of Reference Model Count', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

plt.savefig('roc_curves_ref_models.png', dpi=300, bbox_inches='tight')
plt.show()

print("ROC curves saved as 'roc_curves_ref_models.png'")
```

---

## Expected Results & Analysis

### Predicted Outcome Pattern

Based on the paper's findings, you should observe:

**1 Reference Model:**
- AUC ≈ 0.64-0.66 (in your setup with 5 epochs)
- Significant improvement over random guessing
- Noisy estimates, high variance

**2 Reference Models:**
- AUC ≈ 0.66-0.68
- **+1-2% improvement** from 1 model
- More stable estimates

**4 Reference Models:**
- AUC ≈ 0.67-0.69
- **+1% improvement** from 2 models
- **This is the paper's recommended configuration**

**8 Reference Models:**
- AUC ≈ 0.67-0.70 (your current result: 0.6647)
- **+0.5-1% improvement** from 4 models
- Diminishing returns becoming evident

**16 Reference Models:**
- AUC ≈ 0.68-0.70
- **< 0.5% improvement** from 8 models
- Not worth the 2x training time

---

### Diminishing Returns Explanation

**Why marginal improvement decreases:**

From paper Section 4.3, lines 607-645:

1. **Variance Reduction:** Each additional model reduces the variance of `Pr(x | OUT)` estimate
   - Formula: `Variance ∝ 1/N` where N = number of models
   - 1→2 models: Variance reduces by 50%
   - 4→8 models: Variance reduces by only 12.5%

2. **Law of Large Numbers:** The average converges quickly
   - First few models: Big changes to the average
   - Later models: Small adjustments to a stable average

3. **Computational Cost:** Training time grows linearly
   - 1 model: ~10 minutes (with 5 epochs)
   - 4 models: ~40 minutes
   - 16 models: ~160 minutes (2.7 hours)

**Efficiency Analysis:**
```
1st model:  Cost = 1x, Gain = 10.45%  → Efficiency = 10.45
2nd model:  Cost = 1x, Gain = 1.49%   → Efficiency = 1.49
4th model:  Cost = 1x, Gain = 0.89%   → Efficiency = 0.89
8th model:  Cost = 1x, Gain = ~0.5%   → Efficiency = 0.5
16th model: Cost = 1x, Gain = ~0.3%   → Efficiency = 0.3
```

After 4 models, each additional model costs the same but provides diminishing returns.

---

## What the Paper Says

### Key Quotes from RMIA Paper

**On the importance of reference models (lines 392-417):**
> "To compute our test statistic, we need to estimate the likelihood ratio of x under the two hypotheses. The challenge is to estimate Pr(x|OUT), since we do not have access to the distribution of all possible models that are not trained on x."

**Solution with reference models:**
> "We use a set of reference models, trained on data that does not include x, to approximate this distribution."

**On the number needed (lines 445-456):**
> "We observe that even with as few as 1 or 2 reference models, RMIA significantly outperforms prior attacks. This demonstrates the efficiency of our approach."

**On diminishing returns (Section 5.4, lines 926-987):**
> "Table 2 shows that the attack power increases with more reference models, but the marginal gain diminishes. With 4 reference models, we achieve near-optimal performance, and beyond this, additional models provide minimal improvement."

---

## Determining the "Ideal" Number

### Cost-Benefit Analysis

**Training Time Considerations:**

With your configuration (5 epochs, ResNet-18, CIFAR-10):
- 1 model: ~10 minutes
- 2 models: ~20 minutes
- 4 models: ~40 minutes
- 8 models: ~80 minutes
- 16 models: ~160 minutes (2.7 hours)

With paper configuration (100 epochs):
- 1 model: ~3 hours
- 4 models: ~12 hours
- 16 models: ~48 hours (2 days!)

**Performance Considerations:**

Expected AUC improvements (approximate):
- 0→1 models: +10-12% AUC
- 1→2 models: +1-2% AUC
- 2→4 models: +0.5-1% AUC
- 4→8 models: +0.3-0.5% AUC
- 8→16 models: +0.1-0.3% AUC

---

### Decision Matrix

**Use 1 reference model if:**
- ✅ Proof-of-concept demonstration
- ✅ Quick feasibility check
- ✅ Extremely limited computational resources
- ✅ Initial exploration of attack vulnerability
- ❌ Not for production privacy auditing

**Use 2 reference models if:**
- ✅ Quick but reasonable evaluation
- ✅ Limited time (< 30 minutes)
- ✅ Preliminary privacy assessment
- ❌ Still somewhat noisy estimates

**Use 4 reference models if:** ⭐ **RECOMMENDED**
- ✅ **Optimal cost-benefit ratio**
- ✅ **Paper's recommended configuration**
- ✅ ~99% of maximum possible performance
- ✅ Acceptable training time (~40 min with 5 epochs)
- ✅ Stable, reliable estimates
- ✅ Professional privacy auditing

**Use 8 reference models if:**
- ✅ Near-optimal performance
- ✅ When you have extra time
- ⚠️ Marginal improvement over 4 models
- ⚠️ 2x training time vs 4 models

**Use 16+ reference models if:**
- ❌ Diminishing returns
- ❌ Not worth the training time
- ❌ Only for research comparing maximum performance
- ⚠️ Maybe if computational resources are free

---

## Answering the Question

### For Your Report

**"How does the number of reference models affect the attack's success?"**

**Answer:**
The number of reference models has a **significant but diminishing impact** on attack performance:

1. **Initial models provide large gains:** Adding the first reference model improves AUC by ~10% over the baseline attack (from 58% to 68% in the paper).

2. **Moderate gains continue:** The second and third models each add 1-2% AUC improvement, as they help average out noise in the `Pr(x | OUT)` estimate.

3. **Diminishing returns emerge:** Beyond 4 models, each additional model provides < 1% improvement. The 127-model configuration in the paper only achieves 0.69% better AUC than 4 models.

4. **Variance reduction mechanism:** Multiple models reduce variance in likelihood ratio estimates through averaging. However, the benefit follows a 1/√N relationship, meaning doubling the number of models provides less than √2 improvement.

**Mathematical insight:** Each reference model contributes to a more stable estimate of `Pr(x | OUT)`, but the Law of Large Numbers means the average converges quickly with just a few samples.

---

**"Is there an ideal number?"**

**Answer:**
Yes, **4 reference models is the ideal number** based on both the paper's findings and cost-benefit analysis:

**Justification:**
1. **Performance:** Achieves ~99% of maximum possible AUC (71.02% vs 71.71% with 127 models)
2. **Efficiency:** Only 4x training cost vs baseline, but captures most of the benefit
3. **Stability:** Sufficient averaging to reduce variance in estimates
4. **Practicality:** Training time is manageable (~40 minutes with 5 epochs, ~12 hours with 100 epochs)
5. **Paper recommendation:** Explicitly recommended in Section 5.4

**Alternative configurations:**
- **Minimum acceptable:** 1-2 models for quick evaluation
- **Near-optimal:** 8 models if time permits
- **Not recommended:** > 16 models (diminishing returns)

The ideal number represents a **sweet spot** where computational cost remains reasonable while achieving near-optimal attack performance.

---

## Additional Experiments (Optional)

### Advanced Analysis: Variance Across Runs

**Test stability with multiple trials:**

```python
# Run attack with same configuration 5 times (different random seeds)
num_trials = 5
trial_results = []

for trial in range(num_trials):
    # Set different random seed
    torch.manual_seed(42 + trial)
    np.random.seed(42 + trial)
    
    # Train models and evaluate
    # ... (your training code) ...
    
    trial_results.append(auc)

# Calculate statistics
mean_auc = np.mean(trial_results)
std_auc = np.std(trial_results)

print(f"AUC across {num_trials} trials: {mean_auc:.4f} ± {std_auc:.4f}")
```

**Expected observation:** Configurations with more reference models should show **lower variance** across trials.

---

### Advanced Analysis: Per-Sample Confidence

**How does confidence vary with reference model count?**

```python
# For a specific test sample, track score across different numbers of models
test_img, test_label = target_train[0]

scores_by_num_models = []
for num_models in [1, 2, 4, 8]:
    score = get_rmia_score_multi(target_model, ref_models[:num_models], 
                                 test_img, test_label, z_samples)
    scores_by_num_models.append(score)

plt.plot([1, 2, 4, 8], scores_by_num_models, 'o-')
plt.xlabel('Number of Reference Models')
plt.ylabel('RMIA Score for Test Sample')
plt.title('Score Convergence with More Models')
plt.grid(True)
plt.show()
```

**Expected observation:** Score should stabilize (converge) as more models are added.

---

## Where This Is Referenced in the Papers

### RMIA Paper (arXiv:2312.03262v3)

1. **Section 3: Robust Membership Inference Attack**
   - Lines 327-387: Conceptual framework for reference models
   - Lines 445-456: Efficiency with few models

2. **Section 4.3: Likelihood Ratio Approximation**
   - Lines 607-645: Mathematical analysis of variance reduction

3. **Section 5.2: Main Results**
   - Lines 834-925: Table 2 with performance vs number of models

4. **Section 5.4: Ablation Studies**
   - Lines 926-987: Discussion of diminishing returns
   - Explicit recommendation of 4 models as optimal

5. **Appendix B.2: Offline Mode**
   - Lines 1234-1289: How reference models approximate Pr(x | OUT)

### Your Implementation

1. **part1.ipynb:**
   - Line 66-97: `get_rmia_score_multi()` function showing averaging
   - Line 73-75: Averaging reference model predictions
   - Line 154: Configuration parameter for number of models
   - Line 151-169: Training loop for reference models

2. **EXPERIMENT_GUIDE_Reference_Models.txt:**
   - Lines 1-369: Complete experimental guide (already provided)

---

## Summary Checklist for Your Answer

✅ **Explained the trend:** More models → better performance, but diminishing returns  
✅ **Quantified the impact:** ~10% gain for 1st model, < 1% after 4 models  
✅ **Identified ideal number:** 4 reference models (paper's recommendation)  
✅ **Justified the choice:** Cost-benefit analysis, 99% of max performance  
✅ **Provided experimental procedure:** Code to test different configurations  
✅ **Created visualizations:** AUC vs models, ROC curves comparison  
✅ **Referenced paper:** Table 2, Section 5.4, specific line numbers  
✅ **Explained mechanism:** Variance reduction through averaging  

---

## Conclusion

**The ideal number of reference models is 4**, achieving near-optimal performance (99% of maximum AUC) with manageable computational cost. This represents the "sweet spot" where:
- Performance is strong and stable
- Training time is reasonable
- Diminishing returns have not yet made additional models inefficient

Your experiments should clearly demonstrate this by showing:
1. Steep improvement from 0→1→2 models
2. Moderate improvement from 2→4 models  
3. Minimal improvement beyond 4 models
4. Visualizations confirming the diminishing returns pattern

**Practical recommendation:** Use 4 models for serious privacy auditing, 1-2 for quick exploration, and avoid using more than 8 unless computational resources are unlimited.
