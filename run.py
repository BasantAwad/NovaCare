"""
NovaCare - Application Entry Point (SOLID Refactored)
Simplified entry point that uses the app factory pattern.
"""
from app import create_app

# Create the application
app = create_app()


if __name__ == '__main__':
    # use_reloader=False prevents ERR_CONNECTION_RESET during AI inference
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

