# TOS-Rag-Analyzer

TOS-Rag-Analyzer is a powerful tool that uses Retrieval-Augmented Generation (RAG) to analyze Terms of Service (TOS) documents. It leverages advanced language models and vector embeddings to provide detailed insights into the fairness and legal compliance of TOS agreements.

## Features

- Upload and analyze PDF or TXT Terms of Service documents
- Detailed clause-by-clause analysis
- Fairness rating for each clause (Clearly Fair, Potentially Unfair, Clearly Unfair)
- Overall summary of the TOS document
- Legal compliance assessment
- Interactive visualizations of fairness distribution
- Debug information for transparency and troubleshooting

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TOS-Rag-Analyzer.git
   cd TOS-Rag-Analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root directory or a `secrets.toml` file in the `.streamlit` folder.

2. Add your API keys to the file:

   For `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   For `secrets.toml`:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

   Replace `your_groq_api_key_here` and `your_google_api_key_here` with your actual API keys.

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload a Terms of Service document (PDF or TXT format) using the file uploader.

4. Wait for the analysis to complete. The app will display:
   - Summary statistics of clause fairness
   - A pie chart showing the distribution of fairness ratings
   - Detailed analysis of each clause
   - Overall summary and legal compliance assessment

5. Expand individual clause sections to view detailed explanations.

## Requirements

See `requirements.txt` for a full list of dependencies. Key requirements include:

- requests
- pandas
- faiss-cpu
- groq
- langchain-groq
- PyPDF2
- langchain_google_genai
- langchain
- streamlit
- langchain_community
- python-dotenv
- pypdf
- google-cloud-aiplatform>=1.38

## Contributing

Contributions to TOS-Rag-Analyzer are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for informational purposes only and should not be considered as legal advice. Always consult with a qualified legal professional for interpretation of Terms of Service agreements.