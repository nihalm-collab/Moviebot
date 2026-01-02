# Moviebot - AI-Powered Movie Recommendation System

Moviebot is an intelligent movie recommendation chatbot built with Streamlit and LangChain. It uses RAG (Retrieval-Augmented Generation) architecture to answer questions about movies from the IMDB Top 1000 dataset.

## Features

- 🎬 Natural language queries about movies
- 🔍 Semantic search using Google's embedding models
- 🤖 Powered by Google's Gemini 2.5 Flash Lite
- 📊 Recommendations based on IMDB Top 1000 movies
- ⚡ Fast vector similarity search with ChromaDB
- 💬 Context-aware responses with ratings and release years

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini and Embeddings)
- IMDB Top 1000 Movies Dataset (CSV file)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nihalm-collab/gemini-basic-example.git
cd moviebot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Ensure you have the dataset file `IMDB_Top_1000_Movies_Dataset.csv` in the project directory.

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser (usually at `http://localhost:8501`)

3. Enter your movie-related questions in the text input, such as:
   - "Recommend me some action movies?"
   - "What are the highest-rated sci-fi movies?"
   - "Show me comedy movies starring Meryl Streep"
   - "What are some good Christopher Nolan movies?"

## How It Works

1. **Data Loading**: The app loads the IMDB dataset using LangChain's CSVLoader
2. **Text Splitting**: Documents are split into 1000-character chunks for efficient processing
3. **Embeddings**: Text chunks are converted to vectors using Google's text-embedding-004 model
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
5. **Retrieval**: When you ask a question, the system retrieves the 10 most relevant movie entries
6. **Generation**: Gemini 2.5 Flash Lite generates a natural response based on retrieved context

## Configuration

You can customize the following parameters in the code:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 0)
- `k`: Number of documents to retrieve (default: 5)
- `temperature`: Model creativity (default: 0.3)
- `max_tokens`: Maximum response length (default: 500)

## Dataset

The app expects a CSV file named `IMDB_Top_1000_Movies_Dataset.csv` with movie information including:
- Movie titles
- Movie description
- Movie certificate
- Genres
- Movie cast
- Other relevant metadata

## Limitations

- Recommendations are limited to movies in the IMDB Top 1000 dataset
- Requires an active internet connection for Google API calls
- Response quality depends on the dataset structure and completeness

## Troubleshooting

**Error: "Could not find GOOGLE_API_KEY"**
- Make sure you have a `.env` file with your Google API key

**Error: "File not found"**
- Ensure `IMDB_Top_1000_Movies_Dataset.csv` is in the same directory as the script

**Slow performance**
- Vector store is rebuilt on every run. Consider implementing persistence for production use

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.