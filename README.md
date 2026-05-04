# 🛒 Food Delivery Pattern Analysis
### Data Mining Project — Instacart Market Basket & Amazon Reviews

---

##  Project Overview

This project explores **customer ordering behavior** using real-world grocery data from the **Instacart Market Basket Analysis Dataset**. It applies core data mining techniques to uncover purchasing patterns, rank popular products, and analyze customer sentiment from product reviews.

The project is divided into two main tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Transaction Mining — Association Rule Mining + PageRank |
| **Task 2** | BERT-Based Sentiment Analysis of Product Reviews |

---

## 👥 Team Roles

| Person | Role | Responsibility |
|--------|------|----------------|
| Person 1 | Data Engineer | Data Collection + Preprocessing |
| Person 2 | Mining Specialist| Association Rule Mining (FP-Growth)|
| Person 3 | Graph Analyst | Link Analysis (PageRank) |
| Person 4 | Visualizer | Visualization + Report |
| Person 5 | NLP Engineer | BERT Sentiment Analysis (Task 2) |

---

##  Project Structure

```
 food-delivery-pattern-analysis/
│
├──  01_preprocessing.ipynb           # Person 1 — Data loading & preprocessing
├──  02_association_mining.ipynb      # Person 2 — FP-Growth & association rules
├──  03_pagerank.ipynb                # Person 3 — Graph construction & PageRank
├──  04_visualization.ipynb           # Person 4 — All visualizations
├──  05_bert_sentiment.ipynb          # Person 5 — BERT sentiment model
│
├──  output/
│   ├── frequent_combinations.csv       # Association rules (Person 2 → Person 4)
│   └── meal_ranking.csv                # PageRank scores (Person 3 → Person 4)
│
└── 📄 README.md
```

---

## 🗂️ Datasets

### Task 1 — Instacart Market Basket Analysis
> Source: [Kaggle — yasserh/instacart-online-grocery-basket-analysis-dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)

| File | Description |
|------|-------------|
| `orders.csv` | Customer order metadata (3.4M rows) |
| `order_products__prior.csv` | Products in each order (32.4M rows) |
| `products.csv` | Product names and IDs (49K products) |
| `aisles.csv` | Product category aisles |
| `departments.csv` | Higher-level product groupings |

### Task 2 — Amazon Fine Food Reviews
> Source: [Kaggle — snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

| Detail | Value |
|--------|-------|
| Total reviews | 568,454 |
| Columns used | `Score`, `Summary`, `Text` |
| Balanced sample | 30,000 (10K per class) |

---

##  Task 1 — Transaction Mining

### Step 1: Data Preprocessing *(Person 1)*
- Loaded and merged all CSV files
- Sampled **50,000 orders** for computational efficiency
- Filtered to products appearing in **≥ 200 orders** → 322 unique items
- Built basket lists and applied **one-hot encoding** via `TransactionEncoder`
- Result: **31,273 baskets** ready for mining

---

### Step 2: Association Rule Mining *(Person 2)*

**Algorithm:** FP-Growth (faster than Apriori — uses a compressed FP-Tree, no repeated scans)

**Parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `min_support` | `0.02` | Item must appear in ≥ 2% of baskets (~625 baskets) |
| `max_len` | `3` | Capture up to 3-item bundles (actionable & computationally efficient) |
| `min_lift` | `1.2` | Items must be 20% more likely to co-occur than by chance |

**Results:**

| Metric | Value |
|--------|-------|
| Frequent itemsets found | 76 |
| Association rules generated | 10 |
| High-confidence rules (≥ 25%) | 3 |

**Top Association Rules by Lift:**

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| Bag of Organic Bananas | Organic Hass Avocado | 0.029 | 0.174 | **1.77** |
| Organic Hass Avocado | Bag of Organic Bananas | 0.029 | 0.295 | **1.77** |
| Strawberries | Banana | 0.021 | 0.327 | 1.58 |
| Organic Avocado | Banana | 0.026 | 0.320 | 1.54 |
| Organic Strawberries | Bag of Organic Bananas | 0.029 | 0.239 | 1.44 |

**Key Insight:** All top rules involve organic produce items, revealing a clear **"healthy organic shopper"** behavior pattern. These pairs can directly power a *"Frequently Bought Together"* recommendation feature.

**Output file:** `output/frequent_combinations.csv`

---

### Step 3: Graph-Based Analysis — PageRank *(Person 3)*

- Built a **co-purchase graph**: nodes = products, edges = co-purchase relationships
- Graph size: **322 nodes**, **46,748 edges**
- Applied **PageRank** (α = 0.85, weighted by co-purchase frequency)

**Top 5 Products by PageRank:**

| Rank | Product | PageRank Score |
|------|---------|----------------|
| 1 | Banana | 0.0283 |
| 2 | Bag of Organic Bananas | 0.0238 |
| 3 | Organic Strawberries | 0.0198 |
| 4 | Organic Baby Spinach | 0.0186 |
| 5 | Organic Hass Avocado | 0.0166 |

**Output file:** `output/meal_ranking.csv`

---

### Step 4: Visualization *(Person 4)*

Three visualizations were produced:

1. **Top 15 Products by PageRank** — Horizontal bar chart
2. **Support vs. Confidence Scatter Plot** — Bubble size = lift value
3. **Co-purchase Network Graph** — Top 30 products, edge width = co-purchase frequency

Additionally includes:
- Label distribution before/after balancing (Task 2)
- Confusion matrix heatmap (Task 2)
- Per-class classification metrics bar chart (Task 2)

---

##  Task 2 — BERT Sentiment Analysis *(Person 5)*

**Objective:** Classify Amazon product reviews into Positive / Neutral / Negative sentiments.

**Model:** `bert-base-uncased` from HuggingFace Transformers

### Pipeline Summary

| Step | Details |
|------|---------|
| Label mapping | Score 1–2 → Negative · Score 3 → Neutral · Score 4–5 → Positive |
| Balancing | 10,000 samples × 3 classes = 30,000 total |
| Tokenization | `max_length=128`, padding + truncation |
| Train/Test split | 80% / 20% (24,000 / 6,000) |
| Epochs | 5 |
| Batch size | 16 |
| Hardware | Tesla T4 GPU |
| Training time | ~29 minutes |

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **82.38%** |
| Negative F1-score | 0.82 |
| Neutral F1-score | 0.76 |
| Positive F1-score | 0.90 |

### Interactive Demo
A **Gradio UI** was built for live sentiment prediction — users can enter any product review and receive an instant Positive / Neutral / Negative prediction.

---

##  Requirements

```
pandas
numpy
mlxtend
networkx
matplotlib
seaborn
scikit-learn
transformers==5.7.0
datasets
torch
gradio
kagglehub
```

Install all at once:
```bash
pip install pandas numpy mlxtend networkx matplotlib seaborn scikit-learn transformers datasets torch gradio kagglehub
```

---

##  Key Findings

- **Organic produce dominates** all top association rules — banana, avocado, strawberry, and spinach combinations appear repeatedly
- **Banana** is the single most important product in the co-purchase network (highest PageRank)
- The **healthy organic shopper** is a distinct, identifiable customer segment
- BERT achieves **82.38% accuracy** on 3-class sentiment classification, with Positive reviews being the easiest to classify (F1 = 0.90)

---

## 💡 Actionable Recommendations

| Insight | Recommendation |
|---------|----------------|
| Organic Bananas ↔ Organic Avocado (lift 1.77) | Bundle as a promotional combo offer |
| Strawberries ↔ Banana (lift 1.58) | Surface together on app homepage |
| Organic Baby Spinach → Banana | Target spinach buyers with banana promotions |
| High PageRank products | Feature prominently on landing page |
| Negative sentiment reviews | Flag for customer service follow-up |

---

##  Output Files Summary

| File | Generated By | Used By |
|------|-------------|---------|
| `output/frequent_combinations.csv` | Person 2 | Person 4 |
| `output/meal_ranking.csv` | Person 3 | Person 4 |

---

*Project submitted as part of the Data Mining course.*
