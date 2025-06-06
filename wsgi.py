import os
from app import create_app

# Get configuration from environment variable
config_name = os.environ.get('FLASK_CONFIG', 'production')

print(f"🚀 Starting WSGI application with config: {config_name}")

# Create the Flask application instance
app = create_app(config_name)

if __name__ == "__main__":
    app.run()