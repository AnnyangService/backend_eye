from marshmallow import Schema, fields

class DiagnosisRequestSchema(Schema):
    image_url = fields.Url(required=True)
    cat_id = fields.String(required=True)
