"""
Image Caption Generator - Hugging Face Spaces Deployment Entry Point
Full GUI matching app_gui.py with futuristic styling, URL input, TTS, and examples.
"""

import numpy as np
import json
import pickle
import os
import requests
import tempfile
import threading
from io import BytesIO
from PIL import Image
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "model_19.h5"
PREPRO_FILE = "prepro_by_raj.txt"
CONVOLVED_FEATURES_TRAIN = "convolved_train_features.pkl"
CONVOLVED_FEATURES_TEST = "convolved_test_features.pkl"

model = None
word2no = None
no2word = None
vocabsize = None
maxlen = None
trainconvolve = None
testconvolve = None
newmodel = None

TTS_AVAILABLE = False
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    TTS_AVAILABLE = True
except Exception:
    tts_engine = None


def load_vocabulary_from_prepro():
    global word2no, no2word, vocabsize
    try:
        with open(PREPRO_FILE, 'r') as f:
            discrib = f.read()
        jsonaccept = discrib.replace("'", '"')
        data = json.loads(jsonaccept)
        wordfreq = {}
        for tup in data.items():
            for sen in tup[1]:
                for word in sen.split(" "):
                    wordfreq[str(word)] = wordfreq.get(str(word), 0) + 1
        thresword = {k: v for k, v in wordfreq.items() if v > 10}
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
        return True
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return False


def load_model_and_features():
    global model, newmodel, trainconvolve, testconvolve, maxlen
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
            maxlen = 38
        else:
            return False
        from tensorflow.keras.applications.resnet50 import ResNet50
        resnet_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
        newmodel = resnet_model.__class__(resnet_model.inputs, resnet_model.layers[-2].output)
        if os.path.exists(CONVOLVED_FEATURES_TRAIN):
            with open(CONVOLVED_FEATURES_TRAIN, 'rb') as f:
                trainconvolve = pickle.load(f)
        if os.path.exists(CONVOLVED_FEATURES_TEST):
            with open(CONVOLVED_FEATURES_TEST, 'rb') as f:
                testconvolve = pickle.load(f)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def preprocess_image(img_path):
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
    try:
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            return None
        img_features = newmodel.predict(preprocessed, verbose=0)
        img_features = img_features.reshape(-1)
        return img_features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def generate_caption(image_features):
    global model, word2no, no2word, maxlen
    if model is None or word2no is None:
        return "Error: Model not loaded"
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
        final_caption = " ".join(cap.split()[1:-1])
        return final_caption if final_caption.strip() else "(No caption generated)"
    except Exception as e:
        return f"Error: {e}"


def text_to_speech(text):
    if not TTS_AVAILABLE:
        return None
    output_file = "caption_audio.wav"
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        return output_file if os.path.exists(output_file) else None
    except Exception:
        return None


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[ERROR] URL load: {e}")
        return None


def process_image(image_input, image_url):
    try:
        image_data = None
        if image_input is not None:
            if isinstance(image_input, np.ndarray):
                image_data = Image.fromarray(image_input.astype('uint8'))
            elif isinstance(image_input, Image.Image):
                image_data = image_input
        elif image_url and str(image_url).strip():
            image_data = load_image_from_url(image_url.strip())
        else:
            return "Upload an image or enter a URL", None
        if image_data is None:
            return "Error loading image", None
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
        image_data.save(temp_path, format='JPEG', quality=95)
        features = extract_image_features(temp_path)
        if features is None:
            return "Error extracting features", None
        caption = generate_caption(features)
        audio = text_to_speech(caption)
        return caption, audio
    except Exception as e:
        return f"Error: {e}", None


def clear_inputs():
    return None, ""


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Rajdhani', sans-serif !important;
    background: #000000 !important;
}

.header-bar {
    background: rgba(0,0,0,0.8);
    border-bottom: 2px solid #00ffff;
    padding: 1rem 2rem;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
}

.header-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    text-shadow: 0 0 15px rgba(0,255,255,0.5);
    margin: 0;
}

.header-subtitle {
    font-size: 0.9rem;
    color: #00cccc;
    margin-top: 0.2rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,255,255,0.1);
    color: #00ffff;
    padding: 0.4rem 1rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid rgba(0,255,255,0.4);
    font-family: 'Orbitron', sans-serif;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #00ffff;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(0,255,255,0.8);
    animation: dotPulse 1.5s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.content-area {
    padding: 1.5rem;
    display: flex;
    gap: 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
    align-items: flex-start;
}

.left-panel, .right-panel {
    background: rgba(10,10,20,0.9);
    border: 2px solid rgba(0,255,255,0.2);
    border-radius: 8px;
    padding: 1.5rem;
}

.left-panel { flex: 1.1; }
.right-panel { flex: 1; }

.panel-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #00ffff;
    margin-bottom: 1rem;
    text-transform: uppercase;
    font-family: 'Orbitron', sans-serif;
    border-left: 3px solid #00ffff;
    padding-left: 0.6rem;
}

.image-upload-wrap {
    border: 2px dashed rgba(0,255,255,0.3);
    border-radius: 8px;
    background: rgba(0,255,255,0.02);
}

.url-input input {
    background: rgba(0,0,0,0.6) !important;
    border: 2px solid rgba(0,255,255,0.2) !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    padding: 0.75rem !important;
}

.result-box {
    background: rgba(0,0,0,0.6);
    border-radius: 8px;
    padding: 1.25rem;
    border: 2px solid rgba(0,255,255,0.15);
    min-height: 100px;
}

.caption-text textarea {
    background: transparent !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

.action-bar {
    padding: 1rem 2rem;
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    background: rgba(0,0,0,0.8);
    border-top: 2px solid rgba(0,255,255,0.2);
}

.btn-clear {
    background: transparent !important;
    color: #00cccc !important;
    border: 2px solid rgba(0,255,255,0.4) !important;
    padding: 0.8rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
}

.btn-submit {
    background: #00ffff !important;
    color: #000000 !important;
    border: none !important;
    padding: 0.8rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 20px rgba(0,255,255,0.4);
}

.footer-bar {
    padding: 1rem;
    text-align: center;
    color: #336666;
    font-size: 0.8rem;
    background: rgba(0,0,0,0.8);
    border-top: 1px solid rgba(0,255,255,0.1);
    font-family: 'Orbitron', sans-serif;
}

@media (max-width: 1024px) {
    .content-area { flex-direction: column; padding: 1rem; }
}
"""


def create_interface():
    if not load_vocabulary_from_prepro():
        return None
    if not load_model_and_features():
        return None

    with gr.Blocks(title="Image Caption Generator", css=custom_css) as interface:
        with gr.Column():
            with gr.Row(elem_classes="header-bar"):
                with gr.Column(scale=3):
                    gr.Markdown("<h1 class='header-title'>Image Caption Generator</h1>")
                    gr.Markdown("<p class='header-subtitle'>AI-powered image understanding</p>")
                with gr.Column(scale=1):
                    gr.Markdown("<div style='text-align:right'><span class='status-badge'><span class='status-dot'></span>System Online</span></div>")

            with gr.Row(elem_classes="content-area"):
                with gr.Column(scale=1, elem_classes="left-panel"):
                    gr.Markdown("<div class='panel-title'>Upload Image</div>")
                    image_input = gr.Image(label="", type="pil", elem_classes="image-upload-wrap", height=280)
                    gr.Markdown("<div class='panel-title' style='margin-top:1rem'>Image URL</div>")
                    image_url = gr.Textbox(label="", placeholder="Paste image URL...", elem_classes="url-input")

                with gr.Column(scale=1, elem_classes="right-panel"):
                    gr.Markdown("<div class='panel-title'>Generated Caption</div>")
                    caption_output = gr.Textbox(label="", lines=3, interactive=False, placeholder="Caption appears here...", elem_classes="caption-text")
                    gr.Markdown("<div class='panel-title' style='margin-top:1rem'>Voice Output</div>")
                    audio_output = gr.Audio(label="", type="filepath", autoplay=False)

            with gr.Row(elem_classes="action-bar"):
                clear_btn = gr.Button("Clear", elem_classes="btn-clear")
                submit_btn = gr.Button("Generate Caption", elem_classes="btn-submit")

            with gr.Row(elem_classes="footer-bar"):
                gr.Markdown("ResNet50 + LSTM &middot; TensorFlow &middot; Gradio")

        clear_btn.click(fn=clear_inputs, inputs=None, outputs=[image_input, image_url], scroll_to_output=False)
        submit_btn.click(fn=process_image, inputs=[image_input, image_url], outputs=[caption_output, audio_output], scroll_to_output=False)

    return interface


if __name__ == "__main__":
    print("=" * 60)
    print("Image Caption Generator - Starting Server")
    print("=" * 60)
    interface = create_interface()
    if interface is not None:
        print("\nLaunching Gradio server...")
        interface.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False, theme=gr.themes.Base())
    else:
        print("\nFailed to initialize")
