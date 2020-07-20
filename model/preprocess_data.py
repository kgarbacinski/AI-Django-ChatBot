import nltk
import json
import numpy as np
import random as rnd
import tensorflow as tf

STEMMER = nltk.stem.LancasterStemmer()


def load_json(json_name):
    with open(json_name) as file:
        return json.load(file)


# Distinguis between list of all words, diff tags, patterns(single quesiton)
def get_segments(data):
    all_words = []
    tags = []
    patterns = []
    patterns_cat = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            words = nltk.word_tokenize(pattern)
            all_words.extend(words)
            patterns.append(words)
            patterns_cat.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])

    return all_words, tags, patterns, patterns_cat


# Get basis from all words and make it unique
def stim_words(all_words):
    all_words = [STEMMER.stem(w.lower()) for w in all_words if w != "?"]
    all_words = sorted(list(set(all_words)))

    return all_words


def get_xy(all_words, tags, patterns, patterns_cat):
    X_train = []
    y_train = []

    for idx, pattern in enumerate(patterns):
        buff = []

        y_train_sample = [0 for _ in range(len(tags))]
        words = [STEMMER.stem(w.lower()) for w in pattern]

        for w in all_words:
            if w in words:
                buff.append(1)
            else:
                buff.append(0)

        y_train_sample[tags.index(patterns_cat[idx])] = 1

        y_train.append(y_train_sample)
        X_train.append(buff)

    return np.array(X_train), np.array(y_train)

# Create and save model
def create_model(X_train, y_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[None, 48]))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(len(y_train[0]), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=1000, batch_size=8)

    tf.keras.models.save_model(model, "model.hp5", save_format="h5")

    return model

# Convert input to binary
def conv_input(sent, all_words):
    buff = [0 for _ in range(len(all_words))]

    words = nltk.word_tokenize(sent)
    words = [STEMMER.stem(w.lower()) for w in words]

    for w in words:
        for idx, word in enumerate(all_words):
            if w == word:
                buff[idx] = 1

    return np.array(buff)


def main():
    # Deal with model
    data = load_json("replies.json")

    all_words, tags, patterns, patterns_cat = get_segments(data)
    all_words = stim_words(all_words)

    tags = sorted(tags)

    X_train, y_train = get_xy(all_words, tags, patterns, patterns_cat)

    model = create_model(X_train, y_train)

    # Save all words to file
    with open("all_words.txt", 'w') as file:
        for word in all_words:
            file.write(word + ' ')

    # Save all tags to file
    with open("tags.txt", 'w') as file:
        for tag in tags:
            file.write(tag + ' ')

    # Start program
    print("Hello! What's your question?")

    while True:
        sent = input("> ")
        sent_vec = conv_input(sent, all_words)

        predictions = model.predict(np.array([sent_vec]))

        answer_idx = np.argmax(predictions)

        cat = tags[answer_idx]

        for tg in data["intents"]:
            if tg["tag"] == cat:
                responses = tg["responses"]

        print(rnd.choice(responses))


if __name__ == "__main__":
    main()