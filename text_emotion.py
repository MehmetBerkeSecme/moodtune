# text_emotion_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import re
import string
import random
from collections import Counter


from nltk.sentiment.vader import VaderSentimentAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)  

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpus/wordnet')
except LookupError:
    nltk.download('wordnet')


analyzer = VaderSentimentAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


EMOTION_CATEGORIES = ['happy', 'sad', 'angry', 'neutral', 'energetic', 'calm']


EMOTION_LEXICON = {
    'happy': {
        'words': ['happy', 'joy', 'joyful', 'pleased', 'glad', 'delighted', 'thrilled', 'ecstatic', 'elated', 'cheerful', 'content', 'satisfied', 'optimistic', 'hopeful', 'excited', 'enthusiastic', 'wonderful', 'fantastic', 'amazing', 'great', 'good', 'lovely', 'beautiful', 'smile', 'laugh', 'fun', 'celebrate', 'sunshine', 'bright', 'blessed', 'grateful', 'proud', 'success', 'victory', 'win', 'excellent', 'superb', 'magnificent', 'joyous', 'blissful', 'jubilant', 'overjoyed', 'radiant'],
        'weight': 1.0
    },
    'sad': {
        'words': ['sad', 'sadness', 'unhappy', 'miserable', 'depressed', 'gloomy', 'melancholy', 'sorrow', 'grief', 'heartbroken', 'lonely', 'alone', 'cry', 'tears', 'weep', 'mourn', 'despair', 'hopeless', 'down', 'blue', 'hurt', 'pain', 'suffering', 'regret', 'disappointed', 'lost', 'empty', 'numb', 'somber', 'dreary', 'woeful', 'dejected', 'forlorn', 'wistful', 'downcast', 'despondent'],
        'weight': 1.2
    },
    'angry': {
        'words': ['angry', 'anger', 'furious', 'rage', 'mad', 'irritated', 'annoyed', 'frustrated', 'hostile', 'aggressive', 'outraged', 'livid', 'enraged', 'bitter', 'resentful', 'hate', 'hatred', 'disgust', 'contempt', 'fury', 'wrath', 'indignant', 'provoked', 'incensed', 'fuming', 'seething', 'irate', 'cross', 'vexed', 'grumpy'],
        'weight': 1.3
    },
    'energetic': {
        'words': ['energetic', 'energy', 'pumped', 'hyped', 'motivated', 'dynamic', 'vigorous', 'active', 'lively', 'vibrant', 'enthusiastic', 'excited', 'amped', 'electric', 'intense', 'powerful', 'unstoppable', 'driven', 'passionate', 'fired up', 'ready', 'charged', 'buzzing', 'thriving', 'zestful', 'spirited', 'buoyant'],
        'weight': 0.9
    },
    'calm': {
        'words': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'chill', 'quiet', 'still', 'composed', 'cool', 'collected', 'balanced', 'harmonious', 'meditative', 'mindful', 'easygoing', 'laid-back', 'soothing', 'gentle', 'placid', 'unruffled', 'untroubled', 'mellow', 'restful', 'at ease', 'carefree'],
        'weight': 0.9
    },
    'neutral': {
        'words': ['neutral', 'okay', 'fine', 'alright', 'normal', 'regular', 'average', 'standard', 'typical', 'common', 'ordinary', 'moderate', 'so-so', 'meh', 'whatever', 'indifferent', 'unbiased', 'detached', 'impartial', 'unemotional'],
        'weight': 0.7
    }
}


NEGATION_WORDS = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'without', 'hardly', 'scarcely', 'barely', 'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't"}

def preprocess_text(text):
    """Metni temizler ve normalize eder."""
    text = text.lower()
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # HTML etiketlerini kaldır
    text = re.sub(r'<.*?>', '', text)
    # Noktalama işaretlerini kaldır (ama kelime bütünlüğü için)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Fazla boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
   
    tokens = word_tokenize(text)
    
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

def analyze_vader_sentiment(text):
    
    scores = analyzer.polarity_scores(text)
    return scores

def analyze_emotion_lexicon(tokens):
    
    emotion_scores = {cat: 0 for cat in EMOTION_CATEGORIES}
    
    for token in tokens:
        for category, data in EMOTION_LEXICON.items():
            if token in data['words']:
                emotion_scores[category] += data['weight']
                break  
    
    
    total = sum(emotion_scores.values())
    if total > 0:
        for cat in emotion_scores:
            emotion_scores[cat] = emotion_scores[cat] / total
    return emotion_scores

def map_vader_to_emotion(vader_scores):

    compound = vader_scores['compound']
    pos = vader_scores['pos']
    neg = vader_scores['neg']
    
    # VADER sonucuna göre duygu belirleme
    if compound >= 0.7:
        return 'happy', compound
    elif compound >= 0.3:
        # Pozitif ama çok yüksek değil - energetic veya happy
        return 'happy', compound
    elif compound <= -0.7:
        return 'angry', abs(compound)
    elif compound <= -0.3:
        return 'sad', abs(compound)
    else:
        # Nötr bölge
        return 'neutral', abs(compound)

def detect_negation_impact(tokens, text):
   

    has_negation = any(neg in tokens for neg in NEGATION_WORDS)
    if has_negation:
        
        negation_pattern = re.compile(r'\b(?:' + '|'.join(NEGATION_WORDS) + r')\s+(\w+)', re.IGNORECASE)
        matches = negation_pattern.findall(text)
        if matches:
          
            for match in matches:
                match_lower = match.lower()
                for cat, data in EMOTION_LEXICON.items():
                    if match_lower in data['words']:
                        if cat in ['happy', 'energetic', 'calm']:
                            return 'sad', 0.7
                        elif cat in ['sad', 'angry']:
                            return 'neutral', 0.5
    return None, None

def combine_and_finalize(text, vader_scores, lexicon_scores):
    
    
    
    vader_emotion, vader_conf = map_vader_to_emotion(vader_scores)
    
   
    if lexicon_scores:
        lexicon_emotion = max(lexicon_scores, key=lexicon_scores.get)
        lexicon_conf = lexicon_scores[lexicon_emotion]
    else:
        lexicon_emotion = 'neutral'
        lexicon_conf = 0.3
    
    
    negation_emotion, negation_conf = detect_negation_impact(tokenize_and_lemmatize(text), text)
    

    if negation_emotion:
       
        final_conf = (vader_conf * 0.4 + lexicon_conf * 0.4 + negation_conf * 0.2)
       
        scores = {
            vader_emotion: vader_conf * 0.4,
            lexicon_emotion: lexicon_conf * 0.4,
            negation_emotion: negation_conf * 0.2
        }
        final_emotion = max(scores, key=scores.get)
    else:
        
        final_conf = (vader_conf * 0.5 + lexicon_conf * 0.5)
        
        if vader_emotion == lexicon_emotion:
            final_emotion = vader_emotion
        else:
            if vader_conf > lexicon_conf:
                final_emotion = vader_emotion
                final_conf = (vader_conf + lexicon_conf) / 2
            else:
                final_emotion = lexicon_emotion
                final_conf = (vader_conf + lexicon_conf) / 2
    
    
    final_conf = min(max(final_conf, 0.1), 0.98)
    
    return final_emotion, final_conf

def analyze_text_sentiment(text):
    
    if not text or len(text.strip()) < 3:
        return {'emotion': 'neutral', 'confidence': 0.5, 'error': 'Text too short'}
    
    # Metni ön işle
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        return {'emotion': 'neutral', 'confidence': 0.3, 'error': 'No valid text after preprocessing'}
    
    # Tokenize ve lemmatize et
    tokens = tokenize_and_lemmatize(cleaned_text)
    
    # VADER analizi
    vader_scores = analyze_vader_sentiment(cleaned_text)
    
    # Sözlük tabanlı analiz
    lexicon_scores = analyze_emotion_lexicon(tokens)
    
    # Birleştir ve finalize et
    emotion, confidence = combine_and_finalize(cleaned_text, vader_scores, lexicon_scores)
    
    # Özel durumlar: Energetic ve calm için VADER pozitifliği baz alınabilir
    if emotion in ['happy', 'neutral'] and vader_scores['pos'] > 0.5 and vader_scores['compound'] > 0.4:
        # Yüksek pozitiflik varsa energetic olabilir
        if any(word in cleaned_text for word in ['energy', 'pump', 'excite', 'run', 'go', 'power']):
            emotion = 'energetic'
            confidence = min(confidence + 0.1, 0.98)
    elif emotion in ['happy', 'neutral'] and vader_scores['compound'] < 0.2 and vader_scores['pos'] < 0.2:
        # Düşük uyarılma varsa calm olabilir
        if any(word in cleaned_text for word in ['calm', 'peace', 'quiet', 'relax', 'still', 'soft']):
            emotion = 'calm'
            confidence = min(confidence + 0.1, 0.98)
    
    return {
        'emotion': emotion,
        'confidence': round(confidence, 4),
        'vader_scores': vader_scores,  # Debug için, istenirse kaldırılabilir
        'lexicon_scores': lexicon_scores  # Debug için
    }

@app.route('/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    """
    API endpoint: POST /analyze-text
    Body: { "text": "Your text here" }
    Response: { "emotion": "happy", "confidence": 0.85 }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        result = analyze_text_sentiment(text)
        
        # Debug bilgilerini kaldır (opsiyonel)
        response = {
            'emotion': result['emotion'],
            'confidence': result['confidence']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü için endpoint."""
    return jsonify({'status': 'ok', 'service': 'text-sentiment-analysis'})

if __name__ == '__main__':
    print("Text Emotion Analysis API starting...")
    print("Endpoint: http://localhost:5001/analyze-text")
    print("Health: http://localhost:5001/health")
    app.run(host='0.0.0.0', port=5001, debug=True)