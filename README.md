# Product Recommendation Demo

A small full-stack-style AI project: pick a product a user just viewed, and get
3 similar products back — powered by real embeddings, with **no API key and no
cost**, using a free open-source model from Hugging Face.

This is a good portfolio piece because it demonstrates the same retrieval
pattern (embed → compare → rank) used in production recommendation systems
and RAG pipelines, without depending on a paid API.

## Setup

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The first run will download the embedding model (~90MB, one-time, fully
local afterwards — no internet needed on later runs).

Streamlit will open the app automatically in your browser at
`http://localhost:8501`.

## Files

- `app.py` — the Streamlit frontend and recommendation logic
- `products.py` — the sample product catalog (swap in your own data anytime)
- `requirements.txt` — Python dependencies

## Customizing

- **Add your own products**: edit the `PRODUCTS` list in `products.py`.
- **Use a different embedding model**: change the model name in
  `SentenceTransformer("all-MiniLM-L6-v2")` inside `app.py`. Other good
  free options: `all-mpnet-base-v2` (more accurate, slower).
- **Swap in the OpenAI API instead**: if you later get an OpenAI key, you
  can replace `load_model()`/`embed_products()` with calls to
  `client.embeddings.create(...)` — the rest of the app (similarity ranking,
  UI) stays the same. This is a good "v2" upgrade to show you can work with
  both open-source and commercial embedding APIs.

## Deploying it live (free)

Once it works locally, you can deploy it for free and get a public link to
put on your CV/portfolio:

- **Streamlit Community Cloud** (streamlit.io/cloud) — push this folder to
  a public GitHub repo, connect it, and it deploys automatically.
