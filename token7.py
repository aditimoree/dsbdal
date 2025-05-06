import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample document
text = "Real Madrid is set to win the UCL for the season. Benzema might win Ballon d'Or. Salah might be the runner-up."

# 1. Tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Tokens:", words)

# 2. POS Tagging
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)

# 3. Stopwords Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
print("Filtered Words:", filtered_words)

# 4. Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_words]
print("Stemmed Words:", stemmed)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
print("Lemmatized Words:", lemmatized)

# 6. TF-IDF Vectorization
documents = [
    "Real Madrid is set to win the UCL for the season.",
    "Benzema might win Ballon d'Or.",
    "Salah might be the runner-up."
]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
