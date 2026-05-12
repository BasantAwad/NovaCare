from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import os

# Load environment variables from .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

# Add the current directory to the path (since we're now inside NovaCare)
sys.path.insert(0, current_dir)

# Import from LLMs module (after loading .env)
from LLMs.conversational_ai import ConversationalAI

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the AI
ai = ConversationalAI()

@app.route('/')
def index():
    """Serve the test HTML page"""
    return render_template('test_novabot.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NovaBot API'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - receives user message and returns AI response"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'No message provided'
            }), 400
        
        # Initialize AI if not already initialized
        if not ai._initialized:
            ai.initialize()
        
        # Get AI response
        ai_response = ai.chat(user_message)
        
        # Return response
        return jsonify({
            'response': ai_response,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"[API Error] {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'response': "I'm sorry, I encountered an error. Please try again.",
            'status': 'error'
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        ai.clear_history()
        return jsonify({
            'status': 'success',
            'message': 'History cleared'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("NovaBot API Server")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("API Endpoint: http://localhost:5000/api/chat")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
