import string
from symbol import tfpdef
from sre_parse import Tokenizer

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# reading dataset
df = pd.read_csv('topical_chat.csv')

#dropping duplicates
df.drop_duplicates(subset=['conversation_id', 'message'], inplace=True)

# preprocessing
def process(text):

    if isinstance(text, float) and np.isnan(text):
        return ''  # Return an empty string for NaN values

    text = str(text)  # Convert non-string values to string
    
    text = text.lower().replace('\n', ' ').replace('-', ' ').replace(':', ' ').replace(',', '').replace('"', '') \
          .replace("...", ".").replace("..", ".") \
          .replace("!", ".").replace("?", "") \
          .replace(";", ".").replace(":", " ")

    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')

    text = " ".join(text.split())
    return text


df.message = df.message.apply(process)

def process_messages(df):
    input_texts = []
    target_texts = []

    grouped = df.groupby('conversation_id')
    for _, group in grouped:
        conversation_texts = group['message'].tolist()
        conversation_texts = [text for text in conversation_texts if len(text.split()) < 50]

        for i in range(len(conversation_texts) - 1):
            input_text = conversation_texts[i]
            target_text = conversation_texts[i + 1]

            if len(target_text.split()) <= 10:
                input_texts.append(input_text)
                target_texts.append(target_text)

    return input_texts, target_texts

# Assuming you have a DataFrame named df with 'conversation_id' and 'message' columns
input_texts, target_texts = process_messages(df)

def generate_padded_sequences(sequences):
    max_sequence_len = max([len(x) for x in sequences])
    sequences = np.array(tfpdef.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
    return sequences, max_sequence_len
  
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts) 
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

target_total_words = len(target_tokenizer.word_index) + 1
target_sequences, target_max_sequence_len = generate_padded_sequences(target_sequences)

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts) 
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

input_total_words = len(input_tokenizer.word_index) + 1
input_sequences, input_max_sequence_len = generate_padded_sequences(input_sequences)

# Define Seq2Seq model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(input_max_sequence_len,))
encoder_embedding = Embedding(input_total_words, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(target_max_sequence_len,))
decoder_embedding = Embedding(target_total_words, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_total_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Train the model
model.fit([input_sequences, target_sequences], np.expand_dims(target_sequences, -1), batch_size=64, epochs=10)

# Define inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def reverse_target_char_index(target_token_index):
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    return reverse_target_char_index

# ! Encode the input as state vectors.
# Perform prediction using beam search
def predict_sequence(input_seq, beam_size):
    states_value = encoder_model.predict(input_seq)
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        # Exit condition: either hit max length or find stop eos token.
        if (sampled_char == 'eos' or len(decoded_sentence) > 50):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]