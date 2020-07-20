from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import tensorflow as tf
from django.shortcuts import render, HttpResponse
from model.preprocess_data import conv_input
import numpy as np
import random as rnd

# Load json with replies
with open("model/replies.json") as file:
    replies = json.load(file)

# Load already created model
chatbot_model = tf.keras.models.load_model("model/model.hp5")

# Get all words
with open("model/all_words.txt", 'r') as file:
    all_words = file.readline().split()

# Get all tags
with open("model/tags.txt", 'r') as file:
    tags = file.readline().split()

# Create your views here.
@csrf_exempt
def get_response(request):
    response = {'status': None}

    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        message = data['message']

        # Get converted message
        conv_message = conv_input(message, all_words)

        # Get list with predictions
        predictions = chatbot_model.predict(np.array([conv_message]))
        # Find best suitable prediction
        answer_idx = np.argmax(predictions)
        # Read category
        cat = tags[answer_idx]
        for tg in replies["intents"]:
            if tg["tag"] == cat:
                responses = tg["responses"] # Get all possible responses

        chat_response = rnd.choice(responses)

        response['status'] = 'ok'
        response['message'] = {'text': chat_response, 'user': False, 'chat_bot': True}
    else:
        response['error'] = 'no post data found'

    return HttpResponse(
        json.dumps(response),
        content_type="application/json"
    )


def home(request):
    return render(request, 'chatbot/home.html', context={'title' : 'Chatbot by Kacper Garbaciński'})