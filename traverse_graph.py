import pandas as pd
import faiss
import spacy
from sentence_transformers import SentenceTransformer

# File paths for data and inputs
NODES_DATA_FILE = 'treeoflife_nodes.csv'
LINKS_DATA_FILE = 'treeoflife_links.csv'
FAISS_INDEX_FILE = 'faiss_index.idx'

# Load the spaCy model for dependency parsing and Sentence Transformer model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_subject_from_query(query):
    """Extract the main subject from the query using spaCy's dependency parser."""
    doc = nlp(query)
    
    # Print the entire response for debugging purposes
    print(f"Response from nlp(query): {doc}")
    nouns = [chunk.text for chunk in doc.noun_chunks]
    
    # If subjects or noun phrases are detected, use them; otherwise, use the entire query
    noun_text = " ".join(nouns) if nouns else query
    
    print(f"Noun chunks extracted: {nouns}\nUsing: {noun_text}\n\n")
    
    return noun_text



def closest_node(query, index, node_names):
    """Find the node with the most similar embedding to a query using FAISS."""
    subject = extract_subject_from_query(query)
    query_embedding = model.encode([subject])
    _, indices = index.search(query_embedding.astype('float32'), k=1)
    return node_names[indices[0][0]]

def find_neighbors(node_name, nodes_df, links_df):
    """Find neighbors of a given node in a graph."""
    node_id = nodes_df[nodes_df['node_name'] == node_name]['node_id'].values[0]
    neighbor_ids = links_df[links_df['source_node_id'] == node_id]['target_node_id'].tolist()
    return nodes_df[nodes_df['node_id'].isin(neighbor_ids)]['node_name'].tolist()

def main():
    print("Initializing tools...")
    
    # Load FAISS index and node names
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    node_names = pd.read_csv(NODES_DATA_FILE)['node_name'].tolist()
    
    print("Loading data...")
    nodes_df = pd.read_csv(NODES_DATA_FILE)
    links_df = pd.read_csv(LINKS_DATA_FILE)

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        print("Processing your query...")
        main_node = closest_node(user_query, faiss_index, node_names)
        neighbors = find_neighbors(main_node, nodes_df, links_df)
        print(f"\nMain Node: {main_node}")
        print(f"Neighbors: {neighbors}")

if __name__ == "__main__":
    main()
