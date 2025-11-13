from flask import Flask, render_template, request
from health_matcher import HealthConsultationMatcher

app = Flask(__name__)
matcher = HealthConsultationMatcher()

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        user_input = request.form["symptom"]
        results = matcher.find_matching_drugs(user_input, top_k=3)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
