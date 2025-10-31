# --- Step 1: Import libraries ---
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Step 2: Download needed NLTK resources ---
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Step 3: Load dataset ---
data = pd.read_csv("data/words_pos.csv")
print("Dataset loaded ✅")
print(data.head())

# --- Step 4: Text cleaning & preprocessing ---
lemmatizer = WordNetLemmatizer()

def clean_word(word):
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)  # remove non-letter chars
    return lemmatizer.lemmatize(word)

# --- Step 5: Create contextual features ---
def word_features(i, sentence):
    word = sentence[i]
    features = {
        'word': clean_word(word),
        'suffix1': word[-1:],
        'suffix2': word[-2:],
        'suffix3': word[-3:],
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'length': len(word)
    }
    # previous and next words
    features['prev_word'] = clean_word(sentence[i - 1]) if i > 0 else '<START>'
    features['next_word'] = clean_word(sentence[i + 1]) if i < len(sentence) - 1 else '<END>'
    return features

# --- Step 6: Build sentences ---
sentences = []
current_sentence = []
prev_sent = data.iloc[0]['Sentence #']

for _, row in data.iterrows():
    if row['Sentence #'] != prev_sent:
        sentences.append(current_sentence)
        current_sentence = []
    current_sentence.append((row['Word'], row['POS']))
    prev_sent = row['Sentence #']
if current_sentence:
    sentences.append(current_sentence)

# --- Step 7: Flatten dataset for training ---
X, y = [], []
for sentence in sentences:
    words = [w for w, t in sentence]
    for i in range(len(words)):
        X.append(word_features(i, words))
        y.append(sentence[i][1])

# --- Step 8: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vec = DictVectorizer(sparse=True)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# --- Step 9: Train model ---
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=600, solver='saga', n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Step 10: Evaluate model ---
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- Step 11: Custom Test Sentence (User Input) ---
def predict_sentence(sentence):
    words = sentence.split()
    features = []
    for i in range(len(words)):
        word = words[i]
        feat = {
            'word': clean_word(word),
            'suffix1': word[-1:],
            'suffix2': word[-2:],
            'suffix3': word[-3:],
            'is_title': word.istitle(),
            'is_digit': word.isdigit(),
            'length': len(word),
            'prev_word': clean_word(words[i - 1]) if i > 0 else '<START>',
            'next_word': clean_word(words[i + 1]) if i < len(words) - 1 else '<END>'
        }
        features.append(feat)

    X_new = vec.transform(features)
    predicted_tags = model.predict(X_new)
    return list(zip(words, predicted_tags))

# ---- Example Test ----
sentence = "The cat sat on the mat"
result = predict_sentence(sentence)
print("\nCustom test sentence:")
for word, tag in result:
    print(f"{word:<10} → {tag}")

# --- Step 12: Compare with spaCy ---
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

doc = nlp(sentence)
print("\nspaCy tagging:")
for token in doc:
    print(f"{token.text:<10} → {token.pos_}")
