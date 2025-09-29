from flask import Flask, request
from datetime import datetime

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def age_calculator():
  age = None
  if request.method == "POST":
    
