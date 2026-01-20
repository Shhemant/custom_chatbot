# 🌍 Ecoinvent Search Agent

A smart AI-powered chat application that helps users search and explore the Ecoinvent database (environmental impact data). 

This tool combines **Semantic Search** (to find relevant data even if keywords don't match exactly) with a **Large Language Model** (Llama 3 via Groq) to provide natural language answers.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Features

* **Natural Language Search:** Ask questions like "What are the impacts of steel production?" instead of searching for exact codes.
* **Semantic Matching:** Uses `SentenceTransformers` (BERT) to find relevant database entries based on meaning, not just keywords.
* **AI Synthesis:** Uses the Groq API (Llama 3.3 model) to read the data and explain it to you in plain English.
* **Interactive Chat:** Built with Streamlit for a clean, chat-like interface.

## 🛠️ Prerequisites

Before running this project, ensure you have the following:

1.  **Python installed** (Version 3.8 or higher recommended).
2.  **Groq API Key:** You need an API key to run the LLM. Get one for free at [console.groq.com](https://console.groq.com/).


## 🚀 Setup & Installation

Follow these steps to get the app running on your computer.

### 1. Clone or download the Repository

### 2. Create a Virtual Environment
It is best practice to keep dependencies isolated.

Windows:


python -m venv venv
.\venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate
### 3. Install Dependencies
Install the required Python packages listed in requirements.txt:

pip install -r requirements.txt

### 4. Configure API Keys
Streamlit handles secrets (like API keys) using a specific file.

Create a folder named .streamlit in your project root.

Inside that folder, create a file named secrets.toml.

Add your Groq API key to the file:

# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
⚠️ Important: Never commit your secrets.toml file to GitHub!

## 🏃‍♂️ How to Run
Once everything is installed and configured, run the app with:

streamlit run app.py
A browser window should automatically open pointing to http://localhost:8501.

## 📂 Project Structure
app.py: The main application code containing the UI, the search logic, and the LLM integration.

data/: Directory to store the dataset.

.streamlit/secrets.toml: Configuration file for API keys (ignored by Git).

requirements.txt: List of Python libraries needed.

## 🧠 How it Works
Embedding: When the app starts, it loads the dataset and the BERT model.

User Query: When you ask a question (e.g., "electric car battery"), the app converts your text into numbers (vector embedding).

Vector Search: It compares your query numbers with the numbers in the database to find the most mathematically similar records.

LLM Response: The app sends the found data + your question to the Llama 3 model, which writes a human-readable answer.

## 🤝 Contributing
Pull requests are welcome