"""
Intelligent query processing using RAG - FIXED VERSION
"""

import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class QueryProcessor:
    def __init__(self):
        print("Initializing Query Processor...")

        # Load embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load FAISS index safely
        index_path = os.path.join(MODELS_DIR, "query_index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            print("Warning: query_index.faiss not found, creating empty index.")
            self.index = faiss.IndexFlatL2(384)  # 384 dimensions for MiniLM-L6

        # Load templates and entities safely
        self.templates = self.safe_load_json(os.path.join(MODELS_DIR, "templates.json"), [])
        self.entities = self.safe_load_json(
            os.path.join(MODELS_DIR, "entities.json"),
            {"states": [], "crops": []}
        )

        # State to subdivision mapping
        self.state_to_subdivision_map = {
            "Maharashtra": ["Konkan & Goa", "Madhya Maharashtra", "Matathwada", "Vidarbha"],
            "Karnataka": ["Coastal Karnataka", "North Interior Karnataka", "South Interior Karnataka"],
            "West Bengal": ["Gangetic West Bengal", "Sub Himalayan West Bengal & Sikkim"],
            "Uttar Pradesh": ["East Uttar Pradesh", "West Uttar Pradesh"],
            "Madhya Pradesh": ["West Madhya Pradesh", "East Madhya Pradesh"],
            "Rajasthan": ["West Rajasthan", "East Rajasthan"],
            "Andhra Pradesh": ["Coastal Andhra Pradesh", "Rayalseema", "Telangana"],
            "Gujarat": ["Gujarat Region", "Saurashtra & Kutch"],
            "Assam": ["Assam & Meghalaya"],
            "Meghalaya": ["Assam & Meghalaya"],
            "Odisha": ["Orissa"],
            "Delhi": ["Haryana Delhi & Chandigarh"],
            "Haryana": ["Haryana Delhi & Chandigarh"],
            "Chandigarh": ["Haryana Delhi & Chandigarh"],
            "Jammu And Kashmir": ["Jammu & Kashmir"],
            "Goa": ["Konkan & Goa"],
            "Telangana": ["Telangana"],
            "Nagaland": ["Naga Mani Mizo Tripura"],
            "Manipur": ["Naga Mani Mizo Tripura"],
            "Mizoram": ["Naga Mani Mizo Tripura"],
            "Tripura": ["Naga Mani Mizo Tripura"],
            "Sikkim": ["Sub Himalayan West Bengal & Sikkim"],
            "Tamil Nadu": ["Tamil Nadu"],
            "Kerala": ["Kerala"],
            "Punjab": ["Punjab"],
            "Bihar": ["Bihar"],
            "Jharkhand": ["Jharkhand"],
            "Chhattisgarh": ["Chhattisgarh"],
            "Uttarakhand": ["Uttarakhand"],
            "Himachal Pradesh": ["Himachal Pradesh"],
            "Arunachal Pradesh": ["Arunachal Pradesh"],
        }

        print("Query Processor ready!")

    def safe_load_json(self, path, default):
        """Safely load JSON files with fallback"""
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            print(f"Missing file: {os.path.basename(path)} — using default.")
            return default

    def find_similar_queries(self, query: str, top_k: int = 3):
        """Find most similar template queries using embeddings"""
        if len(self.templates) == 0:
            return []

        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            similar.append(
                {
                    "template": self.templates[idx],
                    "distance": float(dist),
                    "similarity": float(1 / (1 + dist)),
                }
            )
        return similar

    def fuzzy_match_entity(self, text: str, entity_type: str, threshold: int = 1):
        """Enhanced fuzzy match for entities with multi-word and typo handling"""
        text_lower = text.lower().strip()
        entities_list = [e for e in self.entities.get(entity_type, []) if e]

        # Exact match
        for entity in entities_list:
            if entity.lower() in text_lower:
                return entity

        # Multi-word partial match
        for entity in entities_list:
            if text_lower in entity.lower():
                return entity

        # Fuzzy match (closest match)
        matches = get_close_matches(text_lower, [e.lower() for e in entities_list], n=1, cutoff=0.55)
        if matches:
            # Find original case version
            for entity in entities_list:
                if entity.lower() == matches[0]:
                    return entity
        
        return None

    def extract_entities_advanced(self, query: str, template_hint: dict = None):
        """Extract entities using rules + template hints + fuzzy matching"""
        query_lower = query.lower()

        entities = {
            "states": [],
            "crops": [],
            "subdivisions": [],
            "districts": [],
            "years": [],
            "intent": None,
            "aggregation": None,
            "comparison": False,
            "confidence": 0.0,
        }

        # Use template hint if available
        if template_hint:
            entities["intent"] = template_hint.get("intent")
            entities["confidence"] = template_hint.get("similarity", 0.0)
            template_entities = template_hint.get("entities", {})
            
            # Better handling of template states
            template_states = template_entities.get("states", [])
            if isinstance(template_states, list):
                entities["states"] = template_states
            elif isinstance(template_states, str):
                entities["states"] = [template_states]
            
            # Better handling of template crops
            crops = template_entities.get("crops", [])
            if isinstance(crops, list):
                entities["crops"] = crops
            elif isinstance(crops, str):
                entities["crops"] = [crops]

        # Extract states (fuzzy)
        words = query.split()
        for i, word in enumerate(words):
            for length in [3, 2, 1]:
                if i + length <= len(words):
                    phrase = " ".join(words[i:i + length])
                    matched_state = self.fuzzy_match_entity(phrase, "states")
                    if matched_state and matched_state not in entities["states"]:
                        entities["states"].append(matched_state)

        # Extract crops - improved logic
        # For policy/promotion queries, be more selective
        is_policy_query = any(k in query_lower for k in ["policy", "recommend", "promote", "argument"])
        
        if is_policy_query:
            # For policy queries, only extract crops explicitly mentioned
            policy_crops = []
            # Common policy keywords
            promote_match = re.search(r"promote\s+(\w+(?:\s+\w+)?)\s+over\s+(\w+(?:\s+\w+)?)", query_lower)
            if promote_match:
                crop1 = promote_match.group(1).strip()
                crop2 = promote_match.group(2).strip()
                matched1 = self.fuzzy_match_entity(crop1, "crops")
                matched2 = self.fuzzy_match_entity(crop2, "crops")
                if matched1:
                    policy_crops.append(matched1)
                if matched2:
                    policy_crops.append(matched2)
            
            if policy_crops:
                entities["crops"] = policy_crops
        else:
            # For non-policy queries, use broader matching
            for crop in self.entities.get("crops", [])[:200]:
                if crop.lower() in query_lower and crop not in entities["crops"]:
                    entities["crops"].append(crop)

        # Try fuzzy crop match if none found
        if not entities["crops"]:
            for word in words:
                if len(word) > 3:
                    matched_crop = self.fuzzy_match_entity(word, "crops")
                    if matched_crop and matched_crop not in entities["crops"]:
                        entities["crops"].append(matched_crop)

        # Correct year extraction
        year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        if year_matches:
            entities["years"] = [int(y) for y in year_matches]

        last_n = re.search(r"last (\d+) (year|decade)", query_lower)
        if last_n:
            n = int(last_n.group(1))
            if "decade" in last_n.group(2):
                n *= 10
            entities["years"] = f"last_{n}_years"
        elif "decade" in query_lower and not entities["years"]:
            entities["years"] = "last_10_years"

        # Detect intent with improved logic
        if not entities["intent"]:
            if "compare" in query_lower or "versus" in query_lower:
                entities["intent"] = "compare"
                entities["comparison"] = True
            elif "trend" in query_lower or "over time" in query_lower:
                entities["intent"] = "trend"
            elif any(k in query_lower for k in ["impact", "effect", "influence", "correlat"]):
                entities["intent"] = "trend_correlation"
            elif any(k in query_lower for k in ["highest", "lowest", "top"]):
                entities["intent"] = "ranking"
            elif any(
                k in query_lower
                for k in ["policy", "recommend", "promote", "scheme", "substitute", "replace", "encourage", "argument"]
            ):
                entities["intent"] = "policy"
            elif "district" in query_lower:
                entities["intent"] = "district"

        # Ensure compare intent is set
        if "compare" in query_lower or "versus" in query_lower:
            entities["comparison"] = True
            if not entities["intent"]:
                entities["intent"] = "compare"

        # Rainfall-specific detection
        if "rainfall" in query_lower and not entities["intent"]:
            entities["intent"] = "rainfall_analysis"

        # Aggregation
        if "average" in query_lower or "mean" in query_lower:
            entities["aggregation"] = "mean"
        elif "total" in query_lower or "sum" in query_lower:
            entities["aggregation"] = "sum"

        top_match = re.search(r"top (\d+)", query_lower)
        entities["top_n"] = int(top_match.group(1)) if top_match else None

        # Crop category
        for cat in ["cereal", "pulse", "oilseed", "fruit", "vegetable", "fibre", "spice"]:
            if cat in query_lower:
                entities["crop_category"] = cat
                break

        # Normalize + deduplicate
        entities["states"] = [
            s.strip().title().replace("  ", " ") for s in entities["states"]
        ]
        entities["crops"] = [
            c.strip().title().replace("  ", " ") for c in entities["crops"]
        ]
        entities["states"] = list({s.lower().title() for s in entities["states"]})
        entities["crops"] = list({c.lower().title() for c in entities["crops"]})

        # Filter states that are actually in the query
        query_words = query.lower()
        entities["states"] = [
            s for s in entities["states"] if s.lower() in query_words
        ]

        # Fallback for "in <state>"
        if not entities["states"] and entities["crops"]:
            match = re.search(r"in ([a-zA-Z\s]+)", query_lower)
            if match:
                maybe_state = self.fuzzy_match_entity(match.group(1).strip(), "states")
                if maybe_state:
                    entities["states"].append(maybe_state)

        return entities

    def process_query(self, query: str):
        """Main query processing pipeline"""
        similar = self.find_similar_queries(query, top_k=3)
        best_match = similar[0] if similar else None

        # Lower similarity threshold slightly
        if best_match and best_match["similarity"] > 0.55:
            entities = self.extract_entities_advanced(
                query,
                {
                    "intent": best_match["template"]["intent"],
                    "entities": best_match["template"]["entities"],
                    "similarity": best_match["similarity"],
                },
            )
        else:
            entities = self.extract_entities_advanced(query)

        # Safe key handling
        entities["best_template_match"] = (
            best_match.get("template", {}).get("query") if best_match else None
        )
        entities["match_confidence"] = (
            best_match.get("similarity", 0.0) if best_match else 0.0
        )

        entities["original_query"] = query

        # Intent mapping
        intent_map = {
            "compare": "rainfall_crop_comparison",
            "correlate": "trend_correlation",
            "trend": "trend_correlation",
            "district": "district_comparison",
            "policy": "policy_recommendation",
            "ranking": "district_comparison",
        }
        if entities["intent"] in intent_map:
            entities["intent"] = intent_map[entities["intent"]]
        
        # Ensure comparison queries have right intent
        if "compare" in query.lower() or "versus" in query.lower() or "vs" in query.lower():
            entities["comparison"] = True
            if not entities.get("intent"):
                entities["intent"] = "rainfall_crop_comparison"

        print(f"🧩 Detected intent: {entities['intent']} | States: {entities['states']} | Crops: {entities['crops']} | Years: {entities['years']}")
        print(json.dumps(entities, indent=2))  # Debug-friendly output

        return entities

    def get_subdivisions_for_state(self, state: str):
        """Get rainfall subdivisions for a state"""
        return self.state_to_subdivision_map.get(state, [])