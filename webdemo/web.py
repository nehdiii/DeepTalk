from flask import Flask,render_template,request
from flask_ngrok import run_with_ngrok
 
app = Flask(__name__)
run_with_ngrok(app) 

@app.route("/")
def home():
    return "<h1>GFG is great platform to learn</h1>"

@app.route('/form')
def form():
    return render_template('web.html')
 
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('data.html',form_data = form_data)
 
 
app.run()