import os
import re
import json
import random
import logging
import numpy as np
import nltk
import torch
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from functools import wraps
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # Allow CORS for all domains on all routes

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication middleware
def authenticate_middleware(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            # Handle CORS preflight request
            response = app.make_default_options_response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            return response
        try:
            # app.logger.debug(f'Request Headers: {request.headers}')
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authorization header missing'}), 400

            # Split the Authorization header to get the token
            token_type, token = auth_header.split(None, 1)

            # Validate the token (e.g., compare with a hardcoded value)
            if token != 'TESTRR':  # Replace with your actual valid token
                return jsonify({'error': 'Invalid token'}), 401

            # If token is valid, proceed to the requested route function
            return func(*args, **kwargs)

        except ValueError:
            return jsonify({'error': 'Authorization header malformed'}), 403

    return wrapper

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Initialize NLP tools
stemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess intents data
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

# Load the trained model
def load_model(model_path, input_size, hidden_size, output_size):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model_path = 'model.pth'
input_size = len(words)
hidden_size = 8 
output_size = len(classes)
model = load_model(model_path, input_size, hidden_size, output_size)

# Preprocess sentence for model input
def preprocess_sentence(sentence, words):
    sentence_words = sentence.lower().split()
    sentence_words = [stemmer.lemmatize(word) for word in sentence_words if word in words]
    return sentence_words

def sentence_to_features(sentence_words, words):
    features = [1 if word in sentence_words else 0 for word in words]
    return torch.tensor(features).float().unsqueeze(0)

# Generate response from the model
def generate_response(sentence, model, words, classes):
    sentence_words = preprocess_sentence(sentence, words)
    if len(sentence_words) == 0:
        return "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"
    features = sentence_to_features(sentence_words, words)
    with torch.no_grad():
        outputs = model(features)
    probabilities, predicted_class = torch.max(outputs, dim=1)
    confidence = probabilities.item()
    predicted_tag = classes[predicted_class.item()]
    if confidence > 0.5:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])
    return "I'm sorry, but I'm not sure how to respond to that."


# Load data from a JSON file
try:
    with open('questions.json', 'r') as file:
        data = json.load(file)
except Exception as e:
    logger.error("Error loading JSON file: %s", e)
    raise

# Extract questions, answers, and URLs
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]
urls = [item['url'] for item in data]

# Download necessary NLTK data ##done again previously done above
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a set of pronouns to remove
pronouns = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'}

def preprocess_question(question):
    """Preprocess the question by removing punctuation, converting to lowercase,
    and removing stop words and pronouns."""
    try:
        question = re.sub(r'[^\w\s]', '', question)
        question = question.lower()
        question_words = question.split()
        filtered_words = [word for word in question_words if word not in stop_words and word not in pronouns]
        return ' '.join(filtered_words)
    except Exception as e:
        logger.error("Error preprocessing question: %s", e)
        raise

# Preprocess questions
try:
    preprocessed_questions = [preprocess_question(q) for q in questions]
except Exception as e:
    logger.error("Error preprocessing questions: %s", e)
    raise

# Load the vectorizer and matrix from files
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
except Exception as e:
    logger.error("Error loading model files: %s", e)
    raise

def get_best_match_answer(user_question):
    """Get the best match answer for a user's question."""
    try:
        preprocessed_user_question = preprocess_question(user_question)
        user_question_vector = vectorizer.transform([preprocessed_user_question])
        cos_similarities = cosine_similarity(user_question_vector, tfidf_matrix)
        best_match_index = np.argmax(cos_similarities)
        return answers[best_match_index], urls[best_match_index]
    except Exception as e:
        logger.error("Error finding best match: %s", e)
        raise
def update_tfidf_matrix(new_question):
    """Update the TF-IDF matrix with a new question."""
    try:
        global tfidf_matrix, vectorizer, preprocessed_questions
        
        # Preprocess the new question
        preprocessed_question = preprocess_question(new_question)
        
        # Update the vocabulary if there are new words
        new_words = set(preprocessed_question.split()) - set(vectorizer.get_feature_names_out())
        if new_words:
            all_questions = preprocessed_questions + [preprocessed_question]
            vectorizer = TfidfVectorizer().fit(all_questions)
            tfidf_matrix = vectorizer.transform(all_questions).toarray()
            
            # Save the updated vectorizer
            with open('tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
        else:
            # Vectorize the new question and append it to the existing matrix
            new_vector = vectorizer.transform([preprocessed_question])
            tfidf_matrix = np.vstack([tfidf_matrix, new_vector.toarray()])
        
        # Save the updated TF-IDF matrix
        with open('tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        
        # Update preprocessed questions
        preprocessed_questions.append(preprocessed_question)
        
    except Exception as e:
        logger.error("Error updating TF-IDF matrix: %s", e)
        raise




# Flask routes
@app.before_request
@authenticate_middleware
def authenticate():
    pass  # This function will only be called if authentication succeeds

@app.route('/api/testing', methods=['GET'])
def testing():
    return jsonify({'message': 'Token is valid, access granted'}), 200

@app.route('/api/tfidf_message', methods=['POST'])
def tfidf_message():
    """Endpoint to process user questions and return the best match answer and URL."""
    try:
        user_question = request.form.get('question')
        if not user_question:
            return jsonify({'error': 'No question provided'}), 400
        
        best_answer, best_url = get_best_match_answer(user_question)
        return jsonify({'answer': best_answer, 'link': best_url})
    
    except KeyError:
        logger.error("KeyError: 'question' not found in form data")
        return jsonify({'error': 'Question key missing in form data'}), 400
    
    except Exception as e:
        logger.error("Internal server error: %s", e)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/send_message', methods=['POST','OPTIONS'])
def send_input():
    
    try:
        user_input = request.form.get('user_input')  
        if user_input:
            response = generate_response(user_input, model, words, classes)
            return jsonify({'response': response}), 200 
        else:
            return jsonify({'error': 'Invalid input'}), 400  
    except Exception as e:
        app.logger.error("An error occurred while processing the request: %s", e)
        return jsonify({'error': 'Internal server error'}), 500  
    
# Path to the JSON file
json_file_path_user = 'user.json'

# Ensure the JSON file exists
if not os.path.exists(json_file_path_user):
    with open(json_file_path_user, 'w') as file:
        json.dump([], file)

@app.route('/api/relevant', methods=['POST'])
def relevant():
    """Endpoint to receive, store, and log relevant data sent by the user."""
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data or 'link' not in data:
            return jsonify({"error": "Invalid data"}), 400

        with open(json_file_path_user, 'r') as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        existing_data.append(data)

        with open(json_file_path_user, 'w') as file:
            json.dump(existing_data, file, indent=4)

        update_tfidf_matrix(data['question'])

        return jsonify({"message": "Data received and stored successfully!"}), 200
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({"error": "An error occurred"}), 500



# Custom error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404  

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405  

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500  


if __name__ == '__main__':
    app.run(port=5000,debug=True)





