"""
Image Caption Generator - Hugging Face Spaces Deployment Entry Point
ResNet50 + LSTM with Gradio GUI
"""

import numpy as np
import json
import pickle
import os
import requests
import tempfile
import threading
import warnings
from io import BytesIO
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input

# Suppress TF warnings for cleaner logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ===========================
# Configuration
# ===========================

MODEL_PATH = "model_19.h5"
PREPRO_FILE = "prepro_by_raj.txt"

# ===========================
# Global Variables
# ===========================

model = None
word2no = None
no2word = None
vocabsize = None
maxlen = None
newmodel = None


def load_vocabulary_from_prepro():
    """Load vocabulary from prepro_by_raj.txt file"""
    global word2no, no2word, vocabsize

    print("Loading vocabulary from prepro file...")
    try:
        with open(PREPRO_FILE, 'r') as f:
            discrib = f.read()

        jsonaccept = discrib.replace("'", '"')
        data = json.loads(jsonaccept)

        uni_words = set()
        for tup in data.items():
            [uni_words.update([word]) for sentence in tup[1] for word in sentence.split(" ")]

        wordfreq = {}
        for tup in data.items():
            for sen in tup[1]:
                for word in sen.split(" "):
                    if wordfreq.get(str(word)) is None:
                        wordfreq[str(word)] = 0
                    wordfreq[str(word)] += 1

        threshold = 10
        thresword = {k: v for k, v in wordfreq.items() if v > threshold}

        word2no = {}
        no2word = {}
        for ind, word in enumerate(thresword.keys()):
            word2no[word] = ind + 1
            no2word[ind + 1] = word

        word2no['<sos>'] = len(thresword) + 1
        word2no['<eos>'] = len(thresword) + 2
        no2word[len(thresword) + 1] = '<sos>'
        no2word[len(thresword) + 2] = '<eos>'

        vocabsize = len(word2no) + 1

        print(f"Vocabulary loaded: {len(word2no)} words")
        return True
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return False


def load_model_and_features():
    """Load pre-trained model and feature extractors"""
    global model, newmodel, maxlen

    print("Loading model and feature extractor...")
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
            print(f"Model loaded from {MODEL_PATH}")
            try:
                model_input_shape = model.input_shape
                if isinstance(model_input_shape, list):
                    seq_shape = model_input_shape[0]
                else:
                    seq_shape = model_input_shape
                if seq_shape is not None and len(seq_shape) > 1:
                    maxlen = int(seq_shape[1])
                else:
                    maxlen = 39
            except Exception:
                maxlen = 39
        else:
            print(f"Model file not found: {MODEL_PATH}")
            return False

        try:
            from tensorflow.keras.applications.resnet50 import ResNet50
            resnet_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
            newmodel = resnet_model.__class__(resnet_model.inputs, resnet_model.layers[-2].output)
            print("ResNet50 feature extractor loaded")
        except Exception as e:
            print(f"ResNet50 will be created on-the-fly: {e}")
            newmodel = None

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def preprocess_image(img_path):
    """Preprocess image for ResNet50"""
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224, 3))
        img_array = keras_image.img_to_array(img)
        img_array = img_array.reshape(1, 224, 224, 3)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def extract_image_features(img_path):
    """Extract features from image using ResNet50"""
    try:
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            return None

        if newmodel is not None:
            img_features = newmodel.predict(preprocessed, verbose=0)
        else:
            from tensorflow.keras.applications.resnet50 import ResNet50
            from tensorflow.keras.models import Model
            resnet = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
            feature_extractor = Model(resnet.inputs, resnet.layers[-2].output)
            img_features = feature_extractor.predict(preprocessed, verbose=0)

        img_features = img_features.reshape(-1)
        return img_features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def generate_caption(image_features):
    """Generate caption for given image features"""
    global model, word2no, no2word, maxlen

    if model is None or word2no is None:
        return "Error: Model not loaded properly"

    if maxlen is None:
        maxlen = 39

    try:
        cap = "<sos>"
        image_features = image_features.reshape((1, 2048))

        for i in range(maxlen):
            partial_cap = [word2no[x] for x in cap.split() if x in word2no]
            partial_cap = pad_sequences([partial_cap], maxlen, padding='post')

            predicted_array = model([partial_cap, image_features], training=False)
            predicted_array = predicted_array.numpy()

            predicted_no = int(predicted_array.argmax())
            if predicted_no == 0:
                sorted_indices = np.argsort(predicted_array[0, :])[::-1]
                for idx in sorted_indices:
                    if idx != 0 and idx in no2word:
                        predicted_no = idx
                        break

            if predicted_no not in no2word:
                break

            cap = cap + " " + no2word[predicted_no]

            if no2word[predicted_no] == "<eos>":
                break

        final_cap = cap.split()[1:-1]
        final_caption = " ".join(final_cap)

        if not final_caption.strip():
            return "(No caption generated — try a different image.)"

        return final_caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Error generating caption: {e}"


def process_image(image_input):
    """Main function to process image and generate caption"""
    try:
        if image_input is None:
            return "Please upload an image."

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name

        if isinstance(image_input, np.ndarray):
            Image.fromarray(image_input.astype('uint8')).save(temp_path)
        else:
            image_input.save(temp_path)

        image_features = extract_image_features(temp_path)
        if image_features is None:
            return "Error extracting image features."

        caption = generate_caption(image_features)
        return caption
    except Exception as e:
        return f"Error: {e}"


# ===========================
# Gradio Interface
# ===========================

custom_css = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif !important;
}
.header {
    text-align: center;
    margin-bottom: 1rem;
}
.header h1 {
    color: #00aaff;
    font-weight: 700;
}
"""


def create_interface():
    if not load_vocabulary_from_prepro():
        return None
    if not load_model_and_features():
        return None

    with gr.Blocks(title="Image Caption Generator", css=custom_css) as interface:
        gr.Markdown(
            "<div class='header'><h1>Image Caption Generator</h1>"
            "<p>Upload an image and get an AI-generated caption using ResNet50 + LSTM</p></div>"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                submit_btn = gr.Button("Generate Caption", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Column():
                caption_output = gr.Textbox(
                    label="Generated Caption",
                    lines=4,
                    interactive=False,
                    placeholder="Your AI-generated caption will appear here..."
                )

        submit_btn.click(fn=process_image, inputs=image_input, outputs=caption_output)
        clear_btn.click(fn=lambda: (None, ""), inputs=None, outputs=[image_input, caption_output])

    return interface


if __name__ == "__main__":
    print("=" * 60)
    print("Image Caption Generator - Starting Server")
    print("=" * 60)

    interface = create_interface()

    if interface is not None:
        print("\nLaunching Gradio server...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    else:
        print("\nFailed to initialize application")

