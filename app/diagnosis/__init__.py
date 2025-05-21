from flask import Blueprint

diagnosis_bp = Blueprint('diagnosis', __name__)

from . import routes 