"""
Build vector embeddings for query understanding
Run this ONCE to create the index: python src/embeddings_builder.py
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Building RAG Index for Project Samarth...")
print("=" * 60)

# Load lightweight embedding model (420MB, runs on CPU)
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load Data to Build Knowledge Base
print("\nLoading datasets...")
crop_df = pd.read_csv(os.path.join(DATA_DIR, "crop_production.csv"))
rainfall_df = pd.read_csv(os.path.join(DATA_DIR, "rainfall.csv"))

# Normalize column names
crop_df.columns = [c.strip().lower().replace(' ', '_') for c in crop_df.columns]
rainfall_df.columns = [c.strip().upper() for c in rainfall_df.columns]

print(f"Crop data: {crop_df.shape}")
print(f"Rainfall data: {rainfall_df.shape}")

# Create Query Templates (Knowledge Base)
print("\nBuilding query pattern templates...")

# Extract entities from data
states = sorted(crop_df['state_name'].dropna().unique())
crops = sorted(crop_df['crop'].dropna().unique()[:100])  # Top 100 crops
subdivisions = sorted(rainfall_df['SUBDIVISION'].unique())

# Create comprehensive query templates
templates = []

# Rainfall + Crop Comparison
for state1 in states[:10]:
    for state2 in states[:10]:
        if state1 != state2:
            templates.append({
                "query": f"Compare average annual rainfall in {state1} and {state2} for the last 10 years and list top 5 crops",
                "intent": "rainfall_crop_comparison",
                "entities": {"states": [state1, state2], "crops": [], "years": "last 10"},
                "complexity": "high"
            })

# District Production Comparison
for crop in ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']:
    for state in states[:15]:
        templates.append({
            "query": f"Which district in {state} has highest {crop} production",
            "intent": "district_production",
            "entities": {"crop": crop, "state": state},
            "complexity": "medium"
        })

# Trend Correlation
for crop in ['Rice', 'Wheat', 'Cotton', 'Jowar', 'Bajra']:
    templates.append({
        "query": f"Analyze production trend of {crop} over the last decade and correlate with rainfall",
        "intent": "trend_correlation",
        "entities": {"crop": crop, "years": "decade"},
        "complexity": "high"
    })

# Policy Recommendations
policy_pairs = [
    ('Millets', 'Rice'),
    ('Pulses', 'Wheat'),
    ('Cotton', 'Sugarcane'),
]
for crop_a, crop_b in policy_pairs:
    for state in states[:10]:
        templates.append({
            "query": f"Provide data-backed arguments to promote {crop_a} over {crop_b} in {state}",
            "intent": "policy_recommendation",
            "entities": {"crops": [crop_a, crop_b], "state": state},
            "complexity": "high"
        })

# Simple Rankings
for state in states[:20]:
    templates.append({
        "query": f"Top 10 crops in {state}",
        "intent": "ranking",
        "entities": {"state": state},
        "complexity": "low"
    })

# Production by State
templates.append({
    "query": "Show production by state",
    "intent": "state_production",
    "entities": {},
    "complexity": "low"
})

# Rainfall Trends
for subdiv in subdivisions[:20]:
    templates.append({
        "query": f"Rainfall trend in {subdiv}",
        "intent": "rainfall_trend",
        "entities": {"subdivision": subdiv},
        "complexity": "low"
    })

print(f"Created {len(templates)} query templates")

# 3. Generate Embeddings
print("\nGenerating embeddings...")
query_texts = [t["query"] for t in templates]
embeddings = embedder.encode(query_texts, show_progress_bar=True)
print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# 4. Build FAISS Index
print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings.astype('float32'))
print(f"Index built with {index.ntotal} vectors")

# 5. Save Everything
print("\n💾 Saving models and metadata...")

# Save FAISS index
faiss.write_index(index, os.path.join(MODELS_DIR, "query_index.faiss"))
print("Saved: query_index.faiss")

# Save templates metadata
with open(os.path.join(MODELS_DIR, "templates.json"), "w") as f:
    json.dump(templates, f, indent=2)
print("Saved: templates.json")

# Save entity lists for fuzzy matching
entities = {
    "states": states,
    "crops": crops,
    "subdivisions": subdivisions
}
with open(os.path.join(MODELS_DIR, "entities.json"), "w") as f:
    json.dump(entities, f, indent=2)
print("Saved: entities.json")

metadata = {
    "model_name": "all-MiniLM-L6-v2",
    "embedding_dim": dimension,
    "num_templates": len(templates),
    "created_at": pd.Timestamp.now().isoformat()
}
with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved: metadata.json")

print("\n" + "=" * 60)
print("RAG Index built successfully!")
print(f"Models saved in: {MODELS_DIR}")
print(f"Ready to process {len(templates)} query patterns")
print("\nNow run: python src/app.py")
print("=" * 60)