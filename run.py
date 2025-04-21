import os
from app import create_app

# Get configuration from environment variable, default to 'development'
config_name = os.environ.get('FLASK_CONFIG', 'development')
app = create_app(config_name)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 