from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
from rag_engine import RAGEngine

# Path setup for Jinja2 template loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Initialize RAG Engine
print("\n Initializing Project Samarth...")
engine = RAGEngine()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats")
def stats():
    """Return dataset statistics"""
    try:
        stats_data = engine.get_stats()
        return jsonify(stats_data)
    except Exception as e:
        print(f" Error in /stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Handle natural language queries"""
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"type": "error", "data": "Empty query"}), 400

        print(f"\n{'='*60}")
        print(f" Query: {query}")
        
        response = engine.process_query(query)
        
        print(f" Response type: {response.get('type')}")
        print(f"{'='*60}\n")

        # Add timestamp
        response["timestamp"] = datetime.now().isoformat()
        
        # Return response (frontend will format it)
        return jsonify(response)

    except Exception as e:
        print(f" Error in /ask route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "type": "error", 
            "data": f"Processing error: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Starting Project Samarth Backend")
    print("=" * 60)
    
    try:
        stats = engine.get_stats()
        print(f"\n Dataset Statistics:")
        print(f"   Crop records: {stats['crop_records']:,}")
        print(f"   Rainfall records: {stats['rainfall_records']:,}")
        print(f"   States covered: {stats['states']}")
        print(f"   Crop types: {stats['crops']}")
        print(f"   Crop data period: {stats['year_range_crop']}")
        print(f"   Rainfall data period: {stats['year_range_rainfall']}")
        print("=" * 60)
        print("\nServer ready! Open http://localhost:5000\n")
    except Exception as e:
        print(f"\nWarning: Could not load stats: {e}")
        print("Server starting anyway...\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
