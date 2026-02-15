import json

# Load the notebook
with open(r'C:\Users\snten\Desktop\D7079-Practice2\part1.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Markdown cells to insert
markdown_cells = [
    # After cell 2 (training reference models)
    {
        "insert_after": 2,
        "content": {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Training Multiple Reference Models\n",
                "\n",
                "**Purpose:** Reference models provide baseline probability estimates\n",
                "\n",
                "**Configuration:**\n",
                "- Train 8 reference models (paper uses 1-127 models)\n",
                "- Each model trained on different random subset of population data\n",
                "- Use 50% of population data per model for diversity\n",
                "\n",
                "**Why Multiple Models:**\n",
                "- Averaging reduces variance in probability estimates\n",
                "- More stable likelihood ratio calculations\n",
                "- Improves attack accuracy (though diminishing returns after 4 models)"
            ]
        }
    },
    # After cell 3 (single sample test)
    {
        "insert_after": 3,
        "content": {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Single Sample Test with Different Numbers of Reference Models\n",
                "\n",
                "**Experiment:** Test attack on one member sample using 1, 2, 4, and 8 reference models\n",
                "\n",
                "**Purpose:** Demonstrate how multiple reference models affect score stability\n",
                "\n",
                "**Steps:**\n",
                "1. Select first sample from training set (known member)\n",
                "2. Select 100 population samples as baseline (z-samples)\n",
                "3. Calculate RMIA score using different numbers of reference models\n",
                "4. Compare scores to see effect of averaging\n",
                "\n",
                "**Expected:** Scores should stabilize with more models"
            ]
        }
    },
    # After cell 4 (comprehensive evaluation)
    {
        "insert_after": 4,
        "content": {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Comprehensive Attack Evaluation on 200 Samples\n",
                "\n",
                "**Test Set:**\n",
                "- 100 member samples (from target training set)\n",
                "- 100 non-member samples (from test set)\n",
                "\n",
                "**Process:**\n",
                "1. For each sample, compute RMIA score using all 8 reference models\n",
                "2. Label=1 for members, Label=0 for non-members\n",
                "3. Store scores and labels for ROC/AUC analysis\n",
                "\n",
                "**Goal:** Evaluate how well the attack distinguishes members from non-members\n",
                "\n",
                "**Note:** This takes ~22 minutes due to 200 samples × 100 population comparisons × 8 models"
            ]
        }
    },
    # After cell 5 (ROC/AUC calculation)
    {
        "insert_after": 5,
        "content": {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ROC Curve and AUC Calculation\n",
                "\n",
                "**ROC (Receiver Operating Characteristic) Curve:**\n",
                "- X-axis: False Positive Rate (FPR) - non-members incorrectly classified as members\n",
                "- Y-axis: True Positive Rate (TPR) - members correctly identified\n",
                "- Shows trade-off between correctly identifying members vs. false alarms\n",
                "\n",
                "**AUC (Area Under Curve):**\n",
                "- Single metric summarizing attack performance\n",
                "- AUC = 0.50: Random guessing (no better than coin flip)\n",
                "- AUC = 1.00: Perfect attack (100% accuracy)\n",
                "- AUC = 0.66: Our result - attack works significantly better than random\n",
                "\n",
                "**Comparison to Paper:**\n",
                "- Paper achieves 71.02% AUC with 4 models\n",
                "- Our 66.47% is lower due to:\n",
                "  - Fewer training epochs (5 vs. 100)\n",
                "  - Smaller evaluation set (200 vs. thousands)\n",
                "  - Different model architecture"
            ]
        }
    },
    # After cell 6 (histogram)
    {
        "insert_after": 6,
        "content": {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Score Distribution Visualization\n",
                "\n",
                "**Histogram Analysis:**\n",
                "- Blue bars: Member scores (should be higher)\n",
                "- Orange bars: Non-member scores (should be lower)\n",
                "- X-axis: RMIA membership score (0 to 1)\n",
                "- Y-axis: Density (normalized frequency)\n",
                "\n",
                "**Interpretation:**\n",
                "- Members cluster toward right (higher scores)\n",
                "- Non-members cluster toward left (lower scores)\n",
                "- Clear separation indicates attack is effective\n",
                "- Overlap shows some uncertainty (attack not perfect)\n",
                "\n",
                "**Key Insight:** The distribution separation validates that RMIA can distinguish between members and non-members better than random guessing."
            ]
        }
    }
]

# Sort in reverse order to maintain correct indices when inserting
markdown_cells.sort(key=lambda x: x['insert_after'], reverse=True)

# Insert markdown cells
for md_cell in markdown_cells:
    idx = md_cell['insert_after']
    if idx < len(notebook['cells']):
        notebook['cells'].insert(idx + 1, md_cell['content'])

# Save the modified notebook
with open(r'C:\Users\snten\Desktop\D7079-Practice2\part1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Markdown comments added successfully!")
print(f"Total cells now: {len(notebook['cells'])}")
