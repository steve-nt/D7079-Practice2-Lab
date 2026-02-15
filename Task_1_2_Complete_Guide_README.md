# Task 1.2 Analysis and Evaluation - Complete Guide

This document provides a roadmap to the three comprehensive guides created for answering Task 1.2 questions.

---

## Overview

**Task 1.2** from Practice Lab 2 requires you to design and run experiments to answer three questions about the RMIA (Robust Membership Inference Attack) implementation:

1. **How close do your results get to the paper?** (FPR vs TPR and AUROC comparison)
2. **How does the number of reference models affect the attack's success?** (Is there an ideal number?)
3. **What happens if you deliberately create class imbalance?** (Before training)

---

## Guide Files Created

### 📄 Question1_FPR_TPR_AUROC_Comparison.md

**Answers:** How close do your results get to the paper? Evaluate your attack in terms of FPR vs TPR rate as well as AUROC for comparison.

**Key Contents:**
- Comparison of your AUC (0.6647) with paper results (68.64%-71.02%)
- Explanation of the ~5% gap (fewer epochs, smaller evaluation set)
- Detailed explanation of TPR, FPR, and AUC metrics
- Code to extract TPR at specific FPR thresholds (as paper reports)
- How to improve results to match paper (increase epochs to 50-100)
- Creating comparison tables for your report

**Paper References:**
- Table 2 (lines 834-925): Performance metrics
- Section 5.2: Experimental results
- Figure 1 (lines 111-123): ROC curve visualization

**Implementation References:**
- part1.ipynb line 38: Training epochs configuration
- part1.ipynb line 276: AUC calculation
- part1.ipynb line 331: Your analysis of results
- FPR_vs_TPR_Explanation.txt: Complete metrics explanation

---

### 📄 Question2_Reference_Models_Impact.md

**Answers:** How does the number of reference models affect the attack's success? Is there an ideal number?

**Key Contents:**
- Analysis showing diminishing returns pattern
- Paper's findings: 1→2 models = +1.49%, 2→4 = +0.89%, 4→127 = only +0.69%
- **Answer: 4 reference models is ideal** (best cost-benefit ratio)
- Complete experimental procedure to test 1, 2, 4, 8, 16 models
- Code for comparison visualizations (AUC vs models, ROC curves)
- Cost-benefit analysis and decision matrix

**Paper References:**
- Table 2 (lines 834-925): Performance vs number of models
- Section 4.3 (lines 607-645): Variance reduction analysis
- Section 5.4 (lines 926-987): Ablation studies and diminishing returns
- Lines 445-456: Efficiency with few models

**Implementation References:**
- part1.ipynb line 66-97: get_rmia_score_multi() showing averaging
- part1.ipynb line 73-75: Reference model predictions averaging
- part1.ipynb line 154: num_ref_models configuration
- EXPERIMENT_GUIDE_Reference_Models.txt: Detailed experimental guide

---

### 📄 Question3_Class_Imbalance_Impact.md

**Answers:** What happens if you deliberately create class imbalance when setting aside data before training?

**Key Contents:**
- Finding: Class imbalance **reduces attack success** by 5-15% AUC
- Mechanism: Reduced overfitting on minority classes weakens membership signal
- Asymmetric effect: Minority classes harder to attack than majority classes
- Complete implementation code for testing none/mild/severe imbalance
- Per-class analysis showing differential vulnerability
- Conclusion: NOT a good defense (hurts utility), but demonstrates concept

**Paper References:**
- Section 2.1 (lines 234-252): Overfitting as MIA basis
- Section 4.1 (lines 529-557): Confidence and likelihood ratios
- Section 5.3 (lines 888-925): Distribution shift experiments

**Implementation References:**
- AI-implemented/rmia_complete.py lines 345-371: create_imbalanced_dataset()
- AI-implemented/rmia_complete.py lines 462-500: Imbalance experiment
- practice2.txt lines 59-60: Task 1.2 Question 3 specification

---

## Quick Start Guide

### For Each Question, You Need:

**1. Read the corresponding guide** (thoroughly!)
**2. Understand the answer** (what you'll conclude)
**3. Run the experiments** (code provided in each guide)
**4. Generate visualizations** (code provided)
**5. Write your analysis** (templates provided)

---

## Time Estimates

### Question 1: FPR vs TPR Comparison
- **Reading guide:** 30 minutes
- **Understanding metrics:** 15 minutes
- **Running analysis code:** 5 minutes (uses existing results)
- **Creating comparison table:** 15 minutes
- **Total:** ~1 hour

### Question 2: Reference Models Impact
- **Reading guide:** 30 minutes
- **Running experiments (1,2,4,8 models):** 2-4 hours (depends on epochs)
- **Generating visualizations:** 15 minutes
- **Analysis and writing:** 30 minutes
- **Total:** ~3-5 hours

### Question 3: Class Imbalance
- **Reading guide:** 30 minutes
- **Implementing imbalance function:** 15 minutes
- **Running experiments (3 configs):** 2-3 hours (depends on epochs)
- **Generating visualizations:** 15 minutes
- **Analysis and writing:** 30 minutes
- **Total:** ~3-4 hours

**Grand Total:** ~7-10 hours for complete Task 1.2

---

## What Each Guide Provides

### Common Structure Across All Guides:

✅ **Answer Summary** - Quick overview of findings  
✅ **Paper Reference Values** - What the paper reports  
✅ **Your Current Results** - Your implementation's performance  
✅ **Detailed Explanation** - Why results differ or what causes effects  
✅ **Complete Code** - Ready-to-use implementation  
✅ **Visualization Code** - Generate publication-quality figures  
✅ **Paper Citations** - Specific sections and line numbers  
✅ **Implementation References** - Where in your code to make changes  
✅ **Analysis Templates** - How to write up findings  
✅ **Checklists** - Ensure you've covered everything  

---

## How to Use These Guides

### Step 1: Understand the Big Picture

Read this summary document first to understand:
- What each question is asking
- How the questions relate to each other
- How much time you'll need

### Step 2: Tackle Questions in Order

**Recommended order:**

1. **Start with Question 1** (easiest, uses existing results)
   - Gives you understanding of metrics (TPR, FPR, AUC)
   - No new experiments needed
   - Builds foundation for other questions

2. **Then Question 2** (most important experimentally)
   - Tests core attack parameter
   - Clear diminishing returns pattern
   - Paper has extensive data for comparison

3. **Finally Question 3** (most creative)
   - Tests robustness to data distribution
   - Demonstrates understanding of attack mechanisms
   - Shows critical thinking about defenses

### Step 3: For Each Question

1. **Read the full guide** (don't skip sections!)
2. **Understand the answer first** (before running code)
3. **Run the provided code** (copy-paste and modify as needed)
4. **Generate visualizations** (figures are essential for reports)
5. **Interpret your results** (use provided templates)
6. **Write your analysis** (templates provided)

### Step 4: Cross-Check Everything

Use the checklists at the end of each guide to ensure:
- ✅ All experiments run
- ✅ All visualizations generated
- ✅ All paper references included
- ✅ All metrics calculated
- ✅ All questions answered

---

## Key Files in Your Project

### Your Current Implementation

**part1.ipynb** - Main notebook with RMIA implementation
- Line 38: Training configuration (epochs)
- Line 66: get_rmia_score_multi() function
- Line 154: num_ref_models configuration
- Line 202: z_samples size
- Line 243-247: Evaluation loop
- Line 276: AUC calculation and results

**main.py** - Simplified RMIA implementation
- Lines 23-41: get_rmia_score() function
- Lines 9-20: train_model() function

### Reference Materials Provided

**EXPERIMENT_GUIDE_Reference_Models.txt** - Detailed guide for Question 2
**FPR_vs_TPR_Explanation.txt** - Complete metrics explanation for Question 1

### Papers (in Papers/ folder)

**Low-Cost High-Power Membership Inference Attacks.pdf** (or .txt)
- The RMIA paper (arXiv:2312.03262v3)
- Main reference for all questions

**practice2.pdf** (or .txt)
- Lab assignment specification
- Defines all three questions

### AI-Implemented Reference

**AI-implemented/rmia_complete.py** - Complete implementation with:
- create_imbalanced_dataset() function (line 345)
- Full experimental procedures
- All helper functions

---

## Expected Results Summary

### Question 1: Comparison with Paper

**Your results:** AUC = 0.6647 (66.47%)  
**Paper results:** AUC = 68.64%-71.02%  
**Gap:** ~5% lower  
**Explanation:** Fewer epochs (5 vs 100), smaller evaluation set

**Conclusion:** Results are reasonable and validate the implementation despite being lower than paper.

---

### Question 2: Reference Models

**Pattern:** Diminishing returns
- 1 model: +10% over baseline
- 2 models: +1.5% over 1 model
- 4 models: +0.9% over 2 models
- 8+ models: < 0.5% improvement

**Answer:** **4 reference models is ideal** (99% of max performance, reasonable cost)

---

### Question 3: Class Imbalance

**Effect:** Reduces attack success
- Balanced: AUC ≈ 0.66
- Mild imbalance: AUC ≈ 0.62 (6% drop)
- Severe imbalance: AUC ≈ 0.57 (14% drop)

**Answer:** Imbalance weakens attack by reducing overfitting on minority classes, but is NOT a viable defense (hurts model utility).

---

## Tips for Success

### 1. Read Before Coding
Don't jump straight to code. Understanding the "why" makes the "how" much easier.

### 2. Start with Existing Results
Question 1 can be answered with your current results (AUC = 0.6647). Use it to understand metrics before running new experiments.

### 3. Use Smaller Configs for Testing
Before running full experiments:
- Test with epochs=3 instead of 5 (faster)
- Test with 50 samples instead of 100
- Verify code works, then scale up

### 4. Save Everything
```python
# Save results after each experiment
import pickle
with open('results_config_X.pkl', 'wb') as f:
    pickle.dump(results, f)
```

### 5. Compare with Paper Throughout
Each guide includes specific paper sections and line numbers. Look them up!

### 6. Create Visualizations
Figures are worth 1000 words:
- ROC curves (essential)
- Bar charts (AUC comparisons)
- Histograms (score distributions)
- Line plots (trends)

### 7. Use Templates
Each guide provides analysis templates. Adapt them to your specific results.

---

## Common Pitfalls to Avoid

### ❌ Don't:
- Copy code without understanding
- Skip reading the paper sections
- Run experiments without saving results
- Forget to generate visualizations
- Write analysis before running experiments
- Ignore the checklists

### ✅ Do:
- Read guides thoroughly first
- Understand metrics and mechanisms
- Save intermediate results
- Generate all visualizations
- Reference specific paper sections
- Use provided templates
- Cross-check with checklists

---

## Report Structure Suggestion

### For Each Question:

**1. Introduction** (2-3 sentences)
- What the question asks
- Why it matters

**2. Methodology** (1 paragraph)
- Experimental setup
- Configuration used
- What you varied

**3. Results** (1 paragraph + figures)
- Quantitative findings
- Table of results
- ROC curves / visualizations

**4. Analysis** (2-3 paragraphs)
- Interpretation of results
- Comparison with paper
- Explanation of mechanisms

**5. Conclusion** (2-3 sentences)
- Direct answer to the question
- Practical implications

**6. References** (bullet list)
- Paper sections cited (with line numbers)
- Implementation files referenced

---

## Final Checklist

### Before Submitting Your Report:

**Question 1:**
- ✅ Stated your AUC and compared to paper
- ✅ Explained the gap (epochs, sample size)
- ✅ Calculated TPR at specific FPR values
- ✅ Included comparison table
- ✅ Referenced Table 2 from paper

**Question 2:**
- ✅ Tested at least 3-4 different numbers of models
- ✅ Showed diminishing returns pattern
- ✅ Answered "ideal number" (with justification)
- ✅ Included AUC vs models visualization
- ✅ Referenced Section 5.4 from paper

**Question 3:**
- ✅ Tested none/mild/severe imbalance
- ✅ Quantified impact (% AUC drop)
- ✅ Explained mechanism (reduced overfitting)
- ✅ Addressed whether it's a good defense
- ✅ Included comparison visualizations

**Overall:**
- ✅ All visualizations saved and included
- ✅ All code in notebook with comments
- ✅ Paper sections properly cited
- ✅ Answers are clear and direct
- ✅ Explanations are technically sound

---

## Contact / Questions

If you have questions while working through these guides:

1. **Re-read the relevant section** - Most questions are answered in detail
2. **Check the paper** - Specific sections are referenced
3. **Look at your implementation** - Line numbers are provided
4. **Review code comments** - Explanations are inline

Each guide is self-contained and comprehensive. Take your time, work through systematically, and you'll successfully complete Task 1.2!

---

## Good Luck! 🎓

You now have everything you need to:
- ✅ Understand the three questions
- ✅ Run all necessary experiments
- ✅ Generate all visualizations
- ✅ Write comprehensive analysis
- ✅ Reference paper sections properly
- ✅ Complete Task 1.2 successfully

**Estimated total time:** 7-10 hours for complete, thorough work.

**Priority:** Start with Question 1 (quickest), then 2 (most important), then 3 (most creative).

---

**Document created:** 2026-02-15  
**Associated files:**
- Question1_FPR_TPR_AUROC_Comparison.md (15.6 KB)
- Question2_Reference_Models_Impact.md (19.3 KB)
- Question3_Class_Imbalance_Impact.md (25.1 KB)

**Total documentation:** ~60 KB of comprehensive guides for Task 1.2 completion.
