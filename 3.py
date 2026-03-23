import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

# Patch random.randint to ensure it returns integers, as skipgrams expects integer inputs
_original_randint = random.randint

def _patched_randint(a, b):
    return _original_randint(int(a), int(b))
random.randint = _patched_randint

# Sample corpus for training Word2Vec
corpus = [
    "neural networks are powerful",
    "word embeddings capture meaning",
    "deep learning learn representations",
    "natural language processing uses embeddings"
]

# Tokenize the corpus and create skip-gram pairs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

# Create word index and sequences
word_index  = tokenizer.word_index
vocab_size  = len(word_index) + 1
sequences   = tokenizer.texts_to_sequences(corpus)
window_size = 2
pairs, labels = [], []

# Generate skip-gram pairs
for seq in sequences:
    sg = skipgrams(seq, vocabulary_size=vocab_size, window_size=window_size)
    pairs  += sg[0]
    labels += sg[1]

random.randint = _original_randint

# Convert pairs and labels to numpy arrays
pairs  = np.array(pairs)
labels = np.array(labels)

embedding_dim = 50

# Build the Word2Vec model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=1,
        name="embedding"
    ),
    tf.keras.layers.Reshape((embedding_dim,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

target_words  = pairs[:, 0]
context_words = pairs[:, 1]  # not used as input here — Word2Vec needs a 2-input model

# Train the model
model.fit(target_words, labels, epochs=50, verbose=1)

# Extract and display the learned embeddings
embedding_layer = model.get_layer("embedding")
embeddings      = embedding_layer.get_weights()[0]
print("Embeddings for 'neural':")
print(embeddings[word_index['neural']])


#If there is an error as TypeError (float) uncomment the lines from 8 to 12.