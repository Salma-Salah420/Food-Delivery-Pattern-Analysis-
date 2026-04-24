📦 Food Delivery Pattern Analysis
---------------------------------------

🧠 Overview
--------------

This project analyzes food delivery transactional data using Data Mining techniques, specifically Association Rule Mining (FP-Growth algorithm), to discover frequent item combinations and customer purchasing behavior.

The goal is to extract meaningful patterns such as which products are often bought together.

📊 Dataset Description
-------------------------

The dataset contains transactional food delivery orders where each transaction represents a basket of items purchased together.

Example of output after processing:
Antecedents → items purchased first
Consequents → items frequently bought with them


⚙️ Techniques Used
----------------------
Data Cleaning & Preprocessing
Transaction Encoding
FP-Growth Algorithm
Association Rules Mining

📌 Metrics Used:
-----------------
Support
Confidence
Lift
Leverage
Conviction

📈 Key Insights
----------------
🍌 Banana is frequently bought with:
Organic Avocado
Strawberries
🥑 Organic Hass Avocado strongly correlates with:
Bag of Organic Bananas
🥬 Organic Strawberries often appear in the same basket as:
Organic Baby Spinach


📊 Visualizations
----------------------
1️⃣ Top Frequent Items

2️⃣ Support vs Confidence

3️⃣ Association Rules Network Graph

📉 Example of Generated Rules:
---------------------------------
Antecedent	Consequent	Support	Confidence	Lift
Banana	Organic Avocado	0.0259	0.125	1.54
Organic Hass Avocado	Bag of Organic Bananas	0.0289	0.29	1.77


🛠️ Tools & Libraries
--------------------
Python 🐍
Pandas
MLxtend
Matplotlib
Seaborn
Jupyter Notebook


📁 Project Structure:
--------------------------
Food-Delivery-Pattern-Analysis/
│
├── data/
│   └── frequent_combinations.csv
│
├── notebooks/
│   └── instacart-market-basket-analysis-association-rules.ipynb
│
├── images/
│   ├── frequent_items.png
│   ├── support_confidence.png
│   └── rules_network.png
│
└── README.md

🚀 Conclusion
----------------------
This analysis helps understand customer buying behavior and can be used for:

Recommendation systems
Marketing strategies
Product bundling in supermarkets and food delivery apps

## 👥 Team Members
-----------------------
 Alaa orabi-task 2
 Salma Salah>>task 1-
 Mariam Abdelfattah -
 Nada Walied -
 Ahmed Tarek
