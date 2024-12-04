# Virtual Human Project

This project is an AI chatbot that leverages the **Llama 2 7B Chat** model from HuggingFace and a Qdrant database for intelligent conversational capabilities. The project consists of a Google Colab notebook (`VirtualHumanProjectTeam3.ipynb`), a `requirements.txt` file for dependencies, and a `streamlit.py` script for hosting the chatbot via Streamlit.

---

## Project Structure

- **`VirtualHumanProjectTeam3.ipynb`**: A Google Colab notebook containing the core Python code to set up the chatbot, query the database, and integrate the AI model.
- **`requirements.txt`**: A list of required Python libraries and dependencies.
- **`streamlit.py`**: A Python script for hosting the chatbot using Streamlit.

---

## Features

- **AI Chatbot**: Powered by the Llama 2 7B Chat model from HuggingFace.
- **Database Integration**: Queries a stored Qdrant database for additional context and information.
- **Streamlit Hosting**: Deploy the chatbot via a user-friendly web interface.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or later.
- A [HuggingFace](https://huggingface.co/) account and personal token.
- Access to a GPU-enabled environment (recommended for running the chatbot efficiently).

---

### 1. Clone the Repository
Clone the project repository to your local machine: https://github.com/CaJacobsen1193/AI_Chatbot_Student_Advisor

---

### 2. Install Dependencies
Install all required Python libraries using the `requirements.txt` file: pip install -r requirement.txt

---

### 3. Set Up HuggingFace Token
- Sign in to your [HuggingFace account](https://huggingface.co/).
- Generate a personal token under **Account Settings > Access Tokens**.
- Save your token for use in the notebook and script.

---

## Running the Chatbot

### From Google Colab (`VirtualHumanProjectTeam3.ipynb`)
1. Open the notebook in Google Colab.
2. Follow these steps within the notebook:
   - Install necessary libraries using the provided `requirements.txt`.
   - Set up your HuggingFace token in the designated code block.
   - Load the **Llama 2 7B Chat** model from HuggingFace.
   - Read from the Qdrant database.
   - Query the chatbot by running the corresponding cells.

---

### Using Streamlit (`streamlit.py`)
1. **Download the Python Script**  
   Ensure you have `streamlit.py` in your working directory.

2. **Run Streamlit**  
   Open a terminal in your project directory and execute: streamlit run streamlit.py

3. **Access the Web Interface**  
Open the URL provided by Streamlit in your browser to interact with the chatbot.

---

## Notes
- **Model Loading**: Ensure you have the required permissions to access the Llama 2 7B Chat model on HuggingFace.
- **Database Integration**: The Qdrant database must be properly set up and accessible for querying.
- **Performance**: A GPU-enabled system is highly recommended for optimal performance.

---

## Technologies Used
- **Python**: Primary programming language.
- **HuggingFace Transformers**: For loading and interacting with the Llama 2 7B Chat model.
- **Qdrant**: Vector search database for efficient querying.
- **Streamlit**: Web framework for deploying the chatbot interface.

---

Enjoy using the Virtual Human Project! 🚀
