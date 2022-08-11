from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import requests
import subprocess
import json

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb://db:27017")
db = client.ImageRecognition
users = db["Users"]

def verify_pw(username, password):
    if not userExists(username):
        return False

    hashed_pw = users.find_one({
        "Username": username
    })["Password"]

    if bcrypt.hashpw(password.encode('utf8'), hashed_pw):
        return True
    else:
        return False

def generateReturnDictionary(status, msg):
    retJson = {
        "status": status,
        "msg": msg
    }
    return retJson

def verifyCredentials(username, password):
    if not userExists(username):
        return generateReturnDictionary(301, "Invalid Username"), True

    pw_correct = verify_pw(username, password)
    if not pw_correct:
        return generateReturnDictionary(302, "Invalid Password")

    return None, False

def userExists(username):
    if users.find_one({"Username": username}):
        return True
    else:
        return False

class Register(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]

        if userExists(username):
            return jsonify( generateReturnDictionary(301, "This username already exists."))

        hashed_pw = bcrypt.hashpw(password.encode("utf8"), bcrypt.gensalt())

        users.insert_one({
            "Username": username,
            "Password": hashed_pw,
            "Tokens": 4
        })

        return jsonify( generateReturnDictionary(200, "You successfully signed up for this API.") )

class Classify(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]
        url = postedData["url"]

        retJson, error = verifyCredentials(username, password)
        if error:
            return jsonify(retJson)

        tokens = users.find_one({
            "Username": username
        })["Tokens"]

        if tokens <= 0:
            return jsonify( generateReturnDictionary(303, "Not enough tokens!") )

        r = requests.get(url)
        retJson = {}
        with open("temp.jpg", "wb") as f:
            f.write(r.content)
            proc = subprocess.Popen("python classifier.py --model_path=./mobilenet_v3_small.pth --image_path=./temp.jpg", shell=True)
            proc.communicate()[0]
            proc.wait()
            with open("text.txt") as g:
                retJson = json.load(g)
            users.update_one({
                "Username": username,
            },{
                "$set": {
                    "Tokens": tokens-1
                }
            })
            return retJson

class Refill(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["admin_pw"]
        amount = postedData["amount"]

        if not userExists(username):
            return jsonify( generateReturnDictionary(301, "This user does not exist.") )

        correct_pw = "abs123"

        if not password == correct_pw:
            return jsonify( generateReturnDictionary(304, "Invalid Admin Password."))

        users.update_one({
            "Username": username
        },{
            "$set":{
                "Tokens": amount
            }
        })

        return jsonify( generateReturnDictionary(200, "Refilled tokes successfully."))

api.add_resource(Register, '/register')
api.add_resource(Classify, '/classify')
api.add_resource(Refill, '/refill')

if __name__=="__main__":
    app.run(host='0.0.0.0')
