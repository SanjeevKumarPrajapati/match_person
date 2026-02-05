# 🔍 Name Matching System using LangChain & HuggingFace

A smart **name-matching system** that finds the most similar names from a predefined dataset based on user input.  
This project uses **semantic embeddings** and **vector similarity search** to handle spelling variations, typos, and name similarities efficiently.

---

## 🚀 Live Demo

👉 **Try it here:**  
[https://your-streamlit-app-link.streamlit.app](https://matchperson-name.streamlit.app/)


---

## 🧠 How It Works

1. A dataset of similar name variations is created.
2. Each name is converted into an **embedding** using a HuggingFace sentence transformer.
3. The embeddings are stored in a **Chroma vector database**.
4. When a user enters a name:
   - The input is embedded
   - Compared against stored vectors
   - The **top 5 most similar names** are returned with similarity scores

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – for the web interface
- **LangChain**
- **HuggingFace Embeddings**
  - `sentence-transformers/all-MiniLM-L6-v2`
- **Chroma Vector Store**

---

## 📂 Dataset Overview

The model is trained on variations of the following base names:

