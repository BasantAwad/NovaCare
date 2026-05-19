"""
NovaCare - Application Entry Point (SOLID Refactored)
Simplified entry point that uses the app factory pattern.
"""
import os
from dotenv import load_dotenv

# Load environment variables from the repository root .env file.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env_path = os.path.abspath(os.path.join(current_dir, '..', '.env'))
load_dotenv(root_env_path)

from backend import create_app

# Create the application
app = create_app()


if __name__ == '__main__':
    # use_reloader=False prevents ERR_CONNECTION_RESET during AI inference
    port = int(os.environ.get('PORT', 5001))  # Default to 5001 to avoid AirPlay conflict
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False, threaded=True)

