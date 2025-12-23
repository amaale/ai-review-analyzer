# AI-Powered Review Auditor 🐍

An automated pipeline to extract technical insights from customer reviews using Python and LLMs (Google Gemini API).

## 🎯 Objective
To transform raw unstructured data (CSV reviews) into structured executive insights, identifying specific hardware/firmware bugs that standard sentiment analysis misses.

## 🛠 Tech Stack
- **Python 3.10+**
- **Pandas** (Data Manipulation)
- **Google Generative AI** (Gemini 1.5 Flash for NLP)
- **JSON** (Structured Output)

## 📊 Key Features
- **Smart Column Detection:** Automatically identifies review text columns in raw CSVs.
- **Technical Entity Extraction:** Isolates specific hardware faults (e.g., "WiFi module after 3 months").
- **Structured Reporting:** Outputs clean JSON ready for visualization.

## 🚀 Usage
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your CSV file (e.g., `reviews.csv`).
4. Set your API Key in the environment variables.
5. Run: `python analyzer.py`

## ⚠️ Note
This is a portfolio project demonstrating data engineering and API integration skills.
