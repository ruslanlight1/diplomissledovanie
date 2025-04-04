import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")
    SQLALCHEMY_DATABASE_URI = os.environ.get("postgresql://testsite_1wod_user:O9HWM6TqN8S6gTucoOvqxmIDlUn1yREN@dpg-cvo0njripnbc73ecu1j0-a/testsite_1wod", "sqlite:///database.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
