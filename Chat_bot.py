import nltk
from tensorflow import keras
import numpy as np
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', ',', '.']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initializing training
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # init bag of words
    bag = []
    # list of tokenized words
    pattern_words = doc[0]
    # get base word to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create array with 1 if word match found current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = output_empty[:]
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

#create train and test list x patterns y intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = keras.models.load_model('chatbot_model.h5')

# Training loop
epochs = 200
batch_size = 5

for epoch in range(epochs):
    print("Epoch", epoch+1)
    history = model.fit(np.array(train_x), np.array(train_y), epochs=1, batch_size=batch_size, verbose=1)
    # Get user input during training
    user_input = input("Enter a user query: ")
    user_input_words = nltk.word_tokenize(user_input)
    user_input_words = [lemmatizer.lemmatize(word.lower()) for word in user_input_words]
    user_input_bag = []
    for word in words:
        user_input_bag.append(1) if word in user_input_words else user_input_bag.append(0)

    # Predict the intent of user input
    predictions = model.predict(np.array([user_input_bag]))
    predicted_index = np.argmax(predictions)
    tag = classes[predicted_index]

    print("ChatBot:", tag)

# Save the model after training
model.save('chatbot_model.h5')
print('Model saved.')



def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground='#442265', font=('Verdana', 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, 'Bot: ' + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

        # Collect user feedback or corrections
        user_intent = input("Enter the correct intent for the user query: ")
        training.append([msg, user_intent])  # Append user input and corrected intent

        # Update training data
        train_x = []
        train_y = []
        for data in training:
            train_x.append(bow(data[0], words, show_details=False))
            output_row = output_empty[:]
            output_row[classes.index(data[1])] = 1
            train_y.append(output_row)

        # Retrain the model
        model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)

        # Save the retrained model
        model.save('chatbot_model.h5')