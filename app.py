from flask import Flask ,jsonify,render_template,request
from flask_cors import CORS
from airtable import save_rec
from chat import get_response

app = Flask( __name__) 
CORS(app)
#@app.get("/")
#def index_get():
 #   return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    save_rec(text,response)
    return jsonify(message)


if   __name__ == "__main__" : 
    app.run(debug=True)