from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nickname = db.Column(db.String(128))
    timestamp = db.Column(db.String(64))
    image = db.Column(db.String(256))
    suicidal = db.Column(db.Float)
    anxiety = db.Column(db.Float)
    depression = db.Column(db.Float)
    keys = db.Column(db.Text)
