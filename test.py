from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = embedding.embed_query("Hello world")
print(type(vec), len(vec), vec[:5])  # Should print <class 'list'> 384 [...]
