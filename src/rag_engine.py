"""
Main RAG Engine - Orchestrates query processing and response generation
"""

import os
import pandas as pd
from query_processor import QueryProcessor
from data_analyzer import DataAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine...")

        # Load data
        print("Loading datasets...")
        self.crop_df = pd.read_csv(os.path.join(DATA_DIR, "crop_production.csv"))
        self.rainfall_df = pd.read_csv(os.path.join(DATA_DIR, "rainfall.csv"))

        # Clean column names
        self.crop_df.columns = [c.strip().lower() for c in self.crop_df.columns]
        self.rainfall_df.columns = [c.strip().upper() for c in self.rainfall_df.columns]

        print(f"Crop: {self.crop_df.shape}")
        print(f"Rainfall: {self.rainfall_df.shape}")

        # Initialize components
        self.query_processor = QueryProcessor()
        self.analyzer = DataAnalyzer(
            self.crop_df,
            self.rainfall_df,
            self.query_processor.state_to_subdivision_map,
        )

        print("RAG Engine ready!")

    def process_query(self, query: str):
        """Main entry point for query processing"""
        entities = self.query_processor.process_query(query)
        intent = entities.get("intent")

        try:
            if intent in ["rainfall_crop_comparison", "compare"]:
                if len(entities.get("states", [])) >= 2:
                    return self._handle_rainfall_crop_comparison(entities)

            elif intent in ["district_production", "district"]:
                return self._handle_district_comparison(entities)

            elif intent in ["trend_correlation", "correlate", "trend"]:
                return self._handle_trend_correlation(entities)

            elif intent in ["policy_recommendation", "policy"]:
                return self._handle_policy_recommendation(entities)

            elif intent == "ranking":
                return self._handle_ranking(entities)

            elif intent == "state_production":
                return self._handle_state_production(entities)

            else:
                return self._handle_fallback(entities)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "type": "error",
                "data": f"Error processing query: {str(e)}",
                "debug": {"intent": intent, "entities": entities},
            }

    def _handle_rainfall_crop_comparison(self, entities):
        states = entities["states"]
        years = entities.get("years")

        if len(states) < 2:
            return {
                "type": "error",
                "data": "Please specify two states to compare. Example: 'Compare Maharashtra and Punjab'",
            }

        result = self.analyzer.compare_rainfall_and_crops(states[0], states[1], years=years, top_n=5)

        if not result:
            return {
                "type": "error",
                "data": f"Insufficient data for {states[0]} or {states[1]}",
            }

        return {
            "type": "rainfall_crop_comparison",
            "data": result,
            "sources": [
                f"IMD Rainfall Data (Subdivisions: {', '.join(result['rainfall'][states[0]]['subdivisions'][:2])}...)",
                f"Crop Production Dataset ({result['records_analyzed']['crops']} records analyzed)",
            ],
            "query_understanding": {
                "matched_template": entities.get("best_template_match"),
                "confidence": entities.get("match_confidence", 0.0),
            },
        }
    def _handle_district_comparison(self, entities):
        """Compare top producing districts between one or two states for a crop."""
        states = entities.get("states", [])
        crops = entities.get("crops", [])
        crop = crops[0] if crops else None

        if not crop:
            return {
                "type": "error",
                "data": "Please specify a crop. Example: 'Which district has highest rice production?'",
            }

        # --- State setup ---
        if not states:
            return {"type": "error", "data": "Please specify at least one state."}

        state1 = states[0]
        state2 = states[1] if len(states) > 1 else None

        # --- Helper function: Get top district for a crop in a state ---
        def get_top_district(state, crop):
            df = self.crop_df[
                (self.crop_df["state_name"].str.lower() == state.lower())
                & (self.crop_df["crop"].str.lower() == crop.lower())
            ]
            if df.empty:
                return None

            top = (
                df.groupby("district_name")["production_"]
                .sum()
                .reset_index()
                .sort_values("production_", ascending=False)
                .iloc[0]
            )

            return {
                "district": top["district_name"],
                "state": state,
                "production": float(top["production_"]),
                "crop": crop,
            }

        # --- Collect results ---
        results = []
        top1 = get_top_district(state1, crop)
        if top1:
            results.append(top1)
        if state2:
            top2 = get_top_district(state2, crop)
            if top2:
                results.append(top2)

        if not results:
            return {
                "type": "district_comparison",
                "data": {
                    "error": f"No data available for {crop} in specified states.",
                    "crop": crop,
                },
            }

        # --- Compute structured comparison ---
        if len(results) == 1:
            result = {
                "highest": results[0],
                "second": None,
                "crop": crop,
                "ratio": None,
                "records_analyzed": len(self.crop_df),
            }
        else:
            # Sort by production
            results.sort(key=lambda x: x["production"], reverse=True)
            r1, r2 = results[0], results[1]
            ratio = (
                round(r1["production"] / r2["production"], 2)
                if r2["production"] > 0
                else None
            )
            result = {
                "highest": r1,
                "second": r2,
                "crop": crop,
                "ratio": ratio,
                "records_analyzed": len(self.crop_df),
            }

        # --- Final output ---
        return {
            "type": "district_comparison",
            "data": result,
            "sources": [
                f"Crop Production Dataset ({result['records_analyzed']} records analyzed)"
            ],
            "query_understanding": {
                "matched_template": entities.get("best_template_match"),
                "confidence": entities.get("match_confidence", 0.0),
            },
        }

    def _handle_trend_correlation(self, entities):
        """Handle trend correlation queries - IMPROVED"""
        crops = entities["crops"]
        states = entities["states"]
        years = entities.get("years")

        if not crops:
            return {"type": "error", "data": "Please specify a crop. Example: 'Show wheat production trend'"}

        # If asking about correlation, require a state
        query_lower = entities.get("original_query", "").lower()
        needs_correlation = "correlat" in query_lower or "rainfall" in query_lower
        
        if needs_correlation and not states:
            return {
                "type": "error",
                "data": "To correlate with rainfall, please specify a state. Example: 'Analyze wheat production trend in Punjab and correlate with rainfall'"
            }
        
        state = states[0] if states else None
        result = self.analyzer.analyze_trend_correlation(crops[0], state=state, years=years)

        if not result:
            return {"type": "error", "data": f"No data available for {crops[0]}" + (f" in {state}" if state else "")}

        sources = [f"Crop Production Dataset ({result['records_analyzed']} records)"]
        if "correlation" in result:
            sources.append("IMD Rainfall Data (correlated with production)")
        elif not state:
            # If no state specified, explain why no correlation
            result["note"] = "Specify a state to see rainfall correlation. Example: 'wheat trend in Punjab'"

        return {
            "type": "trend_correlation",
            "data": result,
            "sources": sources,
            "query_understanding": {
                "matched_template": entities.get("best_template_match"),
                "confidence": entities.get("match_confidence", 0.0),
            },
        }

    def _handle_policy_recommendation(self, entities):
        crops = entities["crops"]
        states = entities["states"]
        years = entities.get("years")

        if len(crops) < 2:
            query = entities.get("original_query", "").lower()
            if "millet" in query:
                crops.extend(["Bajra", "Jowar", "Ragi", "Small Millets"])
            if len(crops) < 2:
                return {"type": "error", "data": "Please specify two crops to compare."}

        state = states[0] if states else None
        result = self.analyzer.policy_recommendation(crops[0], crops[1], state=state, years=years)

        if not result:
            return {"type": "error", "data": f"Insufficient data to compare {crops[0]} and {crops[1]}"}

        return {
            "type": "policy_recommendation",
            "data": result,
            "sources": [
                f"Crop Production Dataset ({result['records_analyzed']} records analyzed)",
                f"Time Period: {result['time_period']}",
            ],
            "query_understanding": {
                "matched_template": entities.get("best_template_match"),
                "confidence": entities.get("match_confidence", 0.0),
            },
        }

    def _handle_ranking(self, entities):
        states = entities["states"]
        years = entities.get("years")
        state = states[0] if states else None

        result = self.analyzer.get_top_crops(state=state, years=years, top_n=10)
        if not result:
            return {"type": "error", "data": "No data available for the specified query"}

        return {
            "type": "ranking",
            "data": result,
            "sources": [f"Crop Production Dataset ({result['records_analyzed']} records)"],
        }
        
    def _handle_state_production(self, entities):
        result = (
            self.crop_df.groupby("state_name")["production_"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        return {
            "type": "table",
            "data": result.to_dict("records"),
            "sources": [f"Crop Production Dataset ({len(self.crop_df)} records)"],
        }

    def _handle_fallback(self, entities):
        suggestions = [
            "Compare rainfall in Maharashtra and Punjab for last 10 years",
            "Which district in Karnataka has highest rice production?",
            "Show wheat production trend and correlation with rainfall in Punjab.",
            "Why promote millets over rice in Maharashtra?",
        ]

        return {
            "type": "text",
            "data": f"""I couldn't fully understand that query.

Detected intent: {entities.get('intent', 'unclear')}
States: {', '.join(entities.get('states', [])) or 'None'}
Crops: {', '.join(entities.get('crops', [])) or 'None'}
Confidence: {entities.get('match_confidence', 0.0):.2f}

Try asking:
{chr(10).join('• ' + s for s in suggestions)}""",
            "debug_info": entities,
        }

    def get_stats(self):
        unique_states = self.crop_df["state_name"].nunique()
        return {
        "crop_records": len(self.crop_df),
        "rainfall_records": len(self.rainfall_df),
        "states": unique_states,  # Will show 33 (states + UTs in your data)
        "crops": int(self.crop_df["crop"].nunique()),
        "year_range_crop": f"{int(self.crop_df['crop_year'].min())}-{int(self.crop_df['crop_year'].max())}",
        "year_range_rainfall": f"{int(self.rainfall_df['YEAR'].min())}-{int(self.rainfall_df['YEAR'].max())}",
    }
