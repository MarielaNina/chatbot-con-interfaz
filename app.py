from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load JSON data
with open("data.json", "r", encoding='utf-8') as file:
    data = json.load(file)

intents_data = data["intents"]

patterns = []
responses = []

# Extract patterns and responses from intents
for intent in intents_data:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(intent["responses"][0])  # Take only the first associated response

# Text preprocessing with spaCy
nlp = spacy.load("es_core_news_sm")
tokenizer = Tokenizer()

pattern_lemmas = []
for pattern in patterns:
    doc = nlp(pattern)
    lemmas = " ".join([token.lemma_ for token in doc])
    pattern_lemmas.append(lemmas)

tokenizer.fit_on_texts(pattern_lemmas)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(pattern_lemmas)
padded_sequences = pad_sequences(sequences)

labels = []
for response in responses:
    labels.append(responses.index(response))

# Load the model
model = tf.keras.models.load_model('ggg.keras')

# Function to generate response
def generate_response(user_input):
    # Process user input with spaCy
    doc = nlp(user_input)
    lemmatized_input = " ".join([token.lemma_ for token in doc])

    # Convert lemmatized input to token sequence and pad it
    input_sequence = tokenizer.texts_to_sequences([lemmatized_input])
    padded_input = pad_sequences(input_sequence, maxlen=len(padded_sequences[0]))

    # Predict response using the model
    result = model.predict(padded_input)
    predicted_response_index = np.argmax(result)
    predicted_response = responses[predicted_response_index]

    return predicted_response

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
        response = generate_response(msg)
        return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
