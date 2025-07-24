from flask import Flask, render_template, request
import joblib
from groq import Groq

import os
import sqlite3
import datetime


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    # db
    return(render_template("main.html"))

# spam checking using CountVectorizer
@app.route("/spam",methods=["GET","POST"])
def spam():
    return(render_template("spam.html"))

@app.route("/spam_check",methods=["GET","POST"])
def spam_check():
    
    text = request.form.get("q")

    if text is None or text.strip() == "":
        return "Error: No input text provided", 400

    message = [text.strip()]  # make sure whitespace is removed

    import joblib
    encoder = joblib.load("cv_encoder.pkl")
    X_countV = encoder.transform(message)  # ✅ fixed here

    model = joblib.load("lr_model.pkl")
    pred = model.predict(X_countV)

    return render_template("spam_check.html", r=pred)

# query using Llama
@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

@app.route("/deepseek",methods=["GET","POST"])
def deepseek():
    return(render_template("deepseek.html"))

# query using Deepseek
@app.route("/deepseek_reply",methods=["GET","POST"])
def deepseek_reply():
    q = request.form.get("q")
    # load model
    client = Groq()
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("deepseek_reply.html",r=completion_ds.choices[0].message.content))

# prediction using regression
@app.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    q = float(request.form.get("q"))
    # load model
    model = joblib.load("dbs.jl")
    # make prediction
    pred = model.predict([[q]])
    return(render_template("prediction.html",r=pred))

import requests

# telegram conversation using Webhook
@app.route("/telegram",methods=["GET","POST"])
def telegram():
    domain_url = 'https://dsat-ft1.onrender.com'
    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    # Set the webhook URL for the Telegram bot
    set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={domain_url}/webhook"
    webhook_response = requests.post(set_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot is running."
    else:
        status = "Failed to start the telegram bot."
    return(render_template("telegram.html", r=status))

@app.route("/stop_telegram",methods=["GET","POST"])
def stop_telegram():
    domain_url = 'https://dsat-ft1.onrender.com'
    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    webhook_response = requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    # Set the webhook URL for the Telegram bot
    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot has stop."
    else:
        status = "Failed to stop the telegram bot."
    return(render_template("stop_telegram.html", r=status))

@app.route("/webhook",methods=["GET","POST"])
def webhook():
    # This endpoint will be called by Telegram when a new message is received
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        # Extract the chat ID and message text from the update
        chat_id = update["message"]["chat"]["id"]
        query = update["message"]["text"]

        # Pass the query to the Groq model
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        response_message = completion_ds.choices[0].message.content

        # Send the response back to the Telegram chat
        send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(send_message_url, json={
            "chat_id": chat_id,
            "text": response_message
        })
    return('ok', 200)

# SQL database query, insert and delete
@app.route("/user_log",methods=["GET","POST"])
def user_log():
    conn = sqlite3.connect("user.db")
    c = conn.cursor()
    c.execute('''select * from user''')
    r=""
    for row in c:
      print(row)
      r = r + str(row)
    c.close()
    conn.close()
    return render_template("user_log.html", r=r)

@app.route("/add_log",methods=["GET","POST"])
def add_log():
    conn = sqlite3.connect("user.db")
    c = conn.cursor()
    q = request.form.get("q")
    t = datetime.datetime.now()
    c.execute('INSERT INTO user (name,timestamp) VALUES(?,?)',(q,t))
    conn.commit()
    c.close()
    conn.close()
    return render_template("add_log.html", message="User log added successfully.")


@app.route("/delete_log",methods=["GET","POST"])
def delete_log():
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute('DELETE FROM user')
    conn.commit()
    conn.close()
    return render_template("delete_log.html", message="User log deleted successfully.")

@app.route('/sepia', methods=['GET', 'POST'])
def sepia():
    return render_template("sepia_hf.html")

if __name__ == "__main__":
    app.run()

