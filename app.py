import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
import os

# =============================================================================
# Helper functions for loading model and tokenizer with caching
# =============================================================================

@st.cache_resource(show_spinner=True, allow_output_mutation=True)
def load_model(model_path, custom_objects):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


@st.cache_resource(show_spinner=True, persist=True, allow_output_mutation=True)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    return tokenizer

def get_custom_objects():
    # Make sure these definitions match your model's code.
    # (You need to have imported or defined all these classes/functions.)
    custom_objects = {
        "Transformer": Transformer,
        "Encoder": Encoder,
        "Decoder": Decoder,
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
        "MultiHeadAttention": MultiHeadAttention,
        "point_wise_feed_forward_network": point_wise_feed_forward_network,
        "positional_encoding": positional_encoding,
        "create_padding_mask": create_padding_mask,
        "create_look_ahead_mask": create_look_ahead_mask,
    }
    return custom_objects

# =============================================================================
# Set paths and load model/tokenizer
# =============================================================================
# Set the model file path (update this if needed)
MODEL_PATH = "Eng-Arab_transformer_model.keras"

# Display the model path so the user knows where we're loading from.
st.sidebar.write("Model file path:", MODEL_PATH)

# Load custom objects and then the model and tokenizer.
custom_objects = get_custom_objects()
model = load_model(MODEL_PATH, custom_objects)
tokenizer = load_tokenizer()

# =============================================================================
# Option to Reload the Model (to update with an improved version)
# =============================================================================
if st.sidebar.button("Reload Model"):
    load_model.clear()  # Clears the cache for load_model
    model = load_model(MODEL_PATH, custom_objects)
    st.sidebar.success("Model reloaded!")

# =============================================================================
# Define the evaluation (greedy decoding) function
# =============================================================================
def evaluate(sentence, max_length=128):
    # Tokenize input sentence.
    inp = tokenizer(sentence, return_tensors="tf", padding="max_length",
                    truncation=True, max_length=max_length)["input_ids"]
    # Use BOS token if available; otherwise default to token id 2.
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2
    decoder_input = tf.expand_dims([start_token], 0)
    
    # Generate output tokens.
    for i in range(max_length):
        predictions = model({
            "input_ids": inp,
            "decoder_input": decoder_input,
            "attention_mask": inp
        }, training=False)
        predictions = predictions[:, -1:, :]  # shape: (1, 1, vocab_size)
        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
        if tokenizer.eos_token_id is not None and predicted_id[0][0] == tokenizer.eos_token_id:
            break
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    
    predicted_sentence = tokenizer.decode(tf.squeeze(decoder_input, axis=0), skip_special_tokens=True)
    return predicted_sentence

# =============================================================================
# Streamlit UI
# =============================================================================
st.title("Arabic-to-English Translator")
st.write("Enter Arabic text below to translate to English:")

user_input = st.text_area("Arabic Input", "")

if st.button("Translate"):
    if user_input:
        translation = evaluate(user_input)
        st.subheader("Translation:")
        st.write(translation)
    else:
        st.warning("Please enter some text.")
