import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# File paths for data and outputs
NODES_DATA_FILE = 'treeoflife_nodes.csv'
EMBEDDING_FILE = 'embeddings.npy'
FAISS_INDEX_FILE = 'index.idx'

def generate_embeddings_with_sentence_transformers(node_names):
    """Generate embeddings for a list of node names."""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(node_names), batch_size), desc="Generating embeddings"):
        batch = node_names[i:i+batch_size]
        embeddings.extend(model.encode(batch, convert_to_numpy=True))
    return np.array(embeddings)

def build_faiss_index(embeddings):
    """Create and return a FAISS index using the given embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # Add each embedding to the index
    for i in tqdm(range(embeddings.shape[0]), desc="Building FAISS index"):
        index.add(embeddings[i].reshape(1, -1).astype('float32'))
    return index

def main():
    """Main pipeline to generate embeddings and build a FAISS index."""
    # Load node names from the CSV file
    node_names = pd.read_csv(NODES_DATA_FILE)['node_name'].tolist()
    # Generate embeddings for the node names
    embeddings = generate_embeddings_with_sentence_transformers(node_names)
    # Save the generated embeddings
    np.save(EMBEDDING_FILE, embeddings)
    # Build and save the FAISS index
    faiss_index = build_faiss_index(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)

if __name__ == "__main__":
    main()
