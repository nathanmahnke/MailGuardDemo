from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer at startup
model = keras.models.load_model('HamSpamDropoutFinal.keras')

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_sequence_length.txt', 'r') as f:
    max_sequence_length = int(f.read())

# Download NLTK data (only if missing)
nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and stem
    filtered_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(filtered_words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email', '')

    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    processed_text = preprocess_text(email_text)
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)

    prediction = model.predict(padded_seq)[0][0]
    label = "Spam" if prediction >= 0.5 else "Ham"

    return jsonify({
        'label': label,
        'confidence': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
