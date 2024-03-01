import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\LEARNING AI\ques\word.json').read())
words = pickle.load(open('D:\LEARNING AI\ques\words.pkl', 'rb'))
classes = pickle.load(open('D:\LEARNING AI\ques\classes.pkl', 'rb'))
model = load_model('D:\LEARNING AI\ques\chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.85
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print(return_list)
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "Tôi không hiểu, bạn có thể giúp tôi giải đáp được không?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Chào bạn tôi là ITPTI ChatBot. Bấm E nếu bạn không còn thắc mắc gì.")

while True:
    message = input("")
    if message == "E":
        print("Hẹn gặp lại vào lần tới!")
        break
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)

        if res == "Tôi không hiểu, bạn có thể giúp tôi giải đáp được không?":
            print(res)
            new_tag = input("Hãy nhập tag/lĩnh vực của vấn đề này:")
            new_question = input("Hãy nhập câu hỏi: ")
            new_answer = input("Hãy nhập câu trả lời: ")

            new_intent = {
                "tag": [new_tag],
                "patterns": [new_question],
                "responses": [new_answer],
                "context": [""]
            }
            intents["intents"].append(new_intent)

            with open('D:\LEARNING AI\ques\word.json', 'w') as json_file:
                json.dump(intents, json_file, indent=4)

            print("Cảm ơn bạn vì cung cấp thông tin này cho tôi")
        else:
            print(res)