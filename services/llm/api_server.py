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
from LLMs.conversational_ai import ConversationalAI, describe_llm_config

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the AI
ai = ConversationalAI()

# Lazy load emotion analyzer (to avoid slow startup)
emotion_analyzer = None

def get_emotion_analyzer():
    """Lazy load the emotion analyzer."""
    global emotion_analyzer
    if emotion_analyzer is None:
        try:
            from emotion_detection import get_analyzer
            emotion_analyzer = get_analyzer()
            print("✓ Emotion analyzer loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load emotion analyzer: {e}")
    return emotion_analyzer

@app.route('/')
def index():
    """Serve the test HTML page"""
    return render_template('test_novabot.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NovaBot API',
        'llm': describe_llm_config(),
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - receives user message and returns AI response"""
    try:
        print(f"\n📥 [API Route] POST /api/chat - Request received from {request.remote_addr}")
        data = request.json
        
        if not data:
            print("📤 [API Route] POST /api/chat - Error 400: No data provided")
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            print("📤 [API Route] POST /api/chat - Error 400: No message provided")
            return jsonify({
                'error': 'No message provided'
            }), 400

        raw_profile = (data.get('llm_profile') or '').strip().lower()
        if raw_profile in ('fast', 'quality'):
            llm_profile = raw_profile
        elif data.get('prefer_quality') is True:
            llm_profile = 'quality'
        else:
            llm_profile = None
        
        # Initialize AI if not already initialized
        if not ai._initialized:
            ai.initialize()
        
        # Get AI response
        chat_data = ai.chat(user_message, profile=llm_profile)
        
        print(f"📤 [API Route] POST /api/chat - Response returned. Route={ai.last_route}, Profile={ai.last_profile}")
        # Build response with mental-health metadata if available
        resp_data = {
            'response': chat_data['response'],
            'actions': chat_data.get('actions', []),
            'status': 'success',
            'llm_profile': ai.last_profile,
            'llm_route': ai.last_route,
        }
        # Attach mental health metadata if the pipeline triggered
        if ai.last_route and 'therapy' in str(ai.last_route):
            resp_data['mental_health'] = {
                'pipeline_triggered': True,
                'route': ai.last_route,
            }
        return jsonify(resp_data)
    
    except Exception as e:
        print(f"❌ [API Error] {str(e)}")
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


@app.route('/api/emotion/detect', methods=['POST'])
def detect_emotion():
    """
    Detect emotion from a base64 encoded image.
    
    Expects JSON: { "image": "base64_encoded_image_data" }
    Returns: { "emotion": str, "confidence": float, "all_scores": dict, "status": str }
    """
    try:
        print(f"\n📥 [API Route] POST /api/emotion/detect - Image payload received from {request.remote_addr}")
        data = request.json
        
        if not data:
            print("📤 [API Route] POST /api/emotion/detect - Error 400: No data provided")
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        image_data = data.get('image', '').strip()
        
        if not image_data:
            print("📤 [API Route] POST /api/emotion/detect - Error 400: No image provided")
            return jsonify({
                'error': 'No image provided',
                'status': 'error'
            }), 400
        
        # Get or initialize the emotion analyzer
        analyzer = get_emotion_analyzer()
        
        if analyzer is None:
            print("📤 [API Route] POST /api/emotion/detect - Error 500: Emotion analyzer not available")
            return jsonify({
                'error': 'Emotion analyzer not available. Please check server logs.',
                'status': 'error'
            }), 500
        
        # Run emotion detection
        result = analyzer.predict_from_base64(image_data, detect_face=True)
        
        if 'error' in result and result.get('emotion') == 'unknown':
            print(f"📤 [API Route] POST /api/emotion/detect - Face detection failed or error occurred: {result.get('error')}")
            return jsonify({
                'emotion': 'unknown',
                'confidence': 0.0,
                'face_detected': result.get('face_detected', False),
                'error': result.get('error', 'Unknown error'),
                'status': 'error'
            }), 400
        
        print(f"📤 [API Route] POST /api/emotion/detect - Success! Detected Emotion: '{result.get('emotion')}' (Confidence: {result.get('confidence', 0.0):.2%}, FaceDetected={result.get('face_detected', False)})")
        return jsonify({
            'emotion': result.get('emotion', 'unknown'),
            'confidence': result.get('confidence', 0.0),
            'face_detected': result.get('face_detected', False),
            'all_scores': result.get('all_scores', {}),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ [Emotion API Error] {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'emotion': 'unknown',
            'confidence': 0.0,
            'status': 'error'
        }), 500


@app.route('/api/emotion/health', methods=['GET'])
def emotion_health():
    """Check if emotion detection is available."""
    analyzer = get_emotion_analyzer()
    
    if analyzer is None or analyzer.model is None:
        return jsonify({
            'status': 'unavailable',
            'message': 'Emotion analyzer not loaded'
        }), 503
    
    return jsonify({
        'status': 'available',
        'message': 'Emotion analyzer ready',
        'device': analyzer.device,
        'labels': list(analyzer.id2label.values()) if hasattr(analyzer, 'id2label') else []
    })


@app.route('/api/medications', methods=['GET'])
def get_medications():
    """Fetch active medication schedules."""
    try:
        from utils.rag_helper import rag_manager
        meds = rag_manager.get_medications(rover_id="RV001")
        return jsonify({
            'status': 'success',
            'data': meds
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/medications/take', methods=['POST'])
def take_medication():
    """Mark a medication schedule as taken."""
    try:
        data = request.json or {}
        med_id = str(data.get('id', ''))
        if not med_id:
            return jsonify({'error': 'No medication ID provided', 'status': 'error'}), 400
        
        from utils.rag_helper import rag_manager
        if rag_manager.use_mock:
            # Update local mock database JSON file
            db = rag_manager._read_mock_db()
            updated = False
            for m in db.get("medications", []):
                if str(m.get("id")) == med_id:
                    from datetime import datetime
                    m["status"] = "taken"
                    m["taken_at"] = datetime.now().strftime("%H:%M:%S")
                    updated = True
                    break
            if updated:
                rag_manager._write_mock_db(db)
                print(f"[API Route] Mock medication '{med_id}' marked as taken.")
                return jsonify({'status': 'success', 'message': f'Medication {med_id} taken.'})
            else:
                return jsonify({'error': f'Medication {med_id} not found in mock database', 'status': 'error'}), 404
        else:
            # Update MySQL live database
            conn = None
            cursor = None
            try:
                conn = rag_manager.get_connection()
                cursor = conn.cursor()
                query = """
                    UPDATE medication_schedules 
                    SET status = 'taken', taken_at = NOW() 
                    WHERE id = %s AND rover_id = 'RV001'
                """
                cursor.execute(query, (med_id,))
                conn.commit()
                print(f"[API Route] Live SQL medication '{med_id}' marked as taken.")
                return jsonify({'status': 'success', 'message': f'Medication {med_id} marked taken in SQL.'})
            except Exception as sql_err:
                return jsonify({'error': f'SQL error: {sql_err}', 'status': 'error'}), 500
            finally:
                if cursor: cursor.close()
                if conn: conn.close()
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/navigation', methods=['GET', 'POST'])
def handle_navigation():
    """Get or update current navigation status."""
    from utils.rag_helper import rag_manager
    if request.method == 'POST':
        try:
            data = request.json or {}
            dest = data.get('destination')
            status = data.get('status', 'idle')
            follow_mode = data.get('follow_mode', False)
            
            if rag_manager.use_mock:
                db = rag_manager._read_mock_db()
                nav = db.get("navigation", {"destination": None, "status": "idle", "progress": 0, "follow_mode": False})
                
                nav["destination"] = dest
                nav["status"] = status
                nav["follow_mode"] = follow_mode
                if status == 'navigating':
                    nav["progress"] = 10  # Start with 10% progress
                elif status == 'idle':
                    nav["progress"] = 0
                
                db["navigation"] = nav
                rag_manager._write_mock_db(db)
                print(f"[API Route] Mock navigation updated: destination={dest}, status={status}, follow_mode={follow_mode}")
                return jsonify({'status': 'success', 'data': nav})
            else:
                # Live database navigation mode (could log audit entries)
                return jsonify({'status': 'success', 'message': 'MySQL live action logged.'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
    else:
        # GET request
        try:
            if rag_manager.use_mock:
                db = rag_manager._read_mock_db()
                nav = db.get("navigation", {"destination": None, "status": "idle", "progress": 0, "follow_mode": False})
                
                # Auto-advance progress to simulate robot walking!
                if nav.get("status") == "navigating" and nav.get("progress", 0) < 100:
                    nav["progress"] = min(100, nav.get("progress", 0) + 15)
                    if nav["progress"] == 100:
                        nav["status"] = "idle"  # Completed!
                    db["navigation"] = nav
                    rag_manager._write_mock_db(db)
                    print(f"[API Route] Simulating navigation progress: {nav['progress']}%")
                
                return jsonify({'status': 'success', 'data': nav})
            else:
                # Hardcoded or live state fallback
                return jsonify({
                    'status': 'success', 
                    'data': {"destination": None, "status": "idle", "progress": 0, "follow_mode": False}
                })
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/vitals', methods=['GET'])
def get_vitals():
    """Fetch patient vital signs."""
    try:
        from utils.rag_helper import rag_manager
        vitals = rag_manager.get_vitals(rover_id="RV001")
        return jsonify({
            'status': 'success',
            'data': vitals
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/battery', methods=['GET'])
def get_battery():
    """Fetch rover battery status."""
    try:
        from utils.rag_helper import rag_manager
        battery = rag_manager.get_battery(rover_id="RV001")
        return jsonify({
            'status': 'success',
            'data': battery
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ---------------------------------------------------------------------------
# Mental Health Pipeline endpoints
# ---------------------------------------------------------------------------

@app.route('/api/mental-health/analyze', methods=['POST'])
def mental_health_analyze():
    """
    Run the mental-health pipeline on a message WITHOUT going through the main chat.
    Useful for frontends that want to get pattern/risk data independently.
    Expects JSON: { "message": str, "emotion": str (opt), "emotion_confidence": float (opt) }
    """
    try:
        from mental_health_pipeline import get_pipeline
        pipeline = get_pipeline()

        if not pipeline.is_available:
            return jsonify({'error': 'Mental health pipeline not configured', 'status': 'unavailable'}), 503

        data = request.json or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        result = pipeline.process(
            user_message=message,
            emotion=data.get('emotion', 'neutral'),
            emotion_confidence=data.get('emotion_confidence', 0.0),
        )
        return jsonify({
            'triggered': result.triggered,
            'pattern': result.pattern,
            'risk_level': result.risk_level,
            'response': result.response,
            'route': result.route,
            'stages': result.stages_log,
            'status': 'success',
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/mental-health/session', methods=['GET'])
def mental_health_session():
    """Return a summary of mental-health patterns detected in this session."""
    try:
        from mental_health_pipeline import get_pipeline
        pipeline = get_pipeline()
        return jsonify(pipeline.get_session_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mental-health/health', methods=['GET'])
def mental_health_health():
    """Check if the mental-health pipeline is available."""
    try:
        from mental_health_pipeline import get_pipeline
        pipeline = get_pipeline()
        return jsonify({
            'status': 'available' if pipeline.is_available else 'unavailable',
            'groq_configured': bool(os.getenv('GROQ_API_KEY')),
            'cerebras_configured': bool(os.getenv('CEREBRAS_API_KEY')),
            'sambanova_configured': bool(os.getenv('SAMBANOVA_API_KEY')),
            'gemini_configured': bool(os.getenv('GEMINI_API_KEY')),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("NovaBot API Server")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("API Endpoints:")
    print("  - Chat: POST /api/chat")
    print("  - Emotion: POST /api/emotion/detect")
    print("  - Emotion Health: GET /api/emotion/health")
    print("  - Mental Health Analyze: POST /api/mental-health/analyze")
    print("  - Mental Health Session: GET /api/mental-health/session")
    print("  - Mental Health Health: GET /api/mental-health/health")
