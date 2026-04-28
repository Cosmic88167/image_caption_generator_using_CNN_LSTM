"""
Image Caption Generator GUI using Gradio
Supports image upload, caption generation, and text-to-speech
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
import pyttsx3

# ===========================
# Initialize Global Variables
# ===========================

MODEL_PATH = "model_19.h5"
PREPRO_FILE = "prepro_by_raj.txt"
CONVOLVED_FEATURES_TRAIN = "convolved_train_features.pkl"
CONVOLVED_FEATURES_TEST = "convolved_test_features.pkl"

# Global variables to store loaded data
model = None
word2no = None
no2word = None
vocabsize = None
maxlen = None
embedmatrix = None
trainconvolve = None
testconvolve = None
newmodel = None

# Text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)


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
        
        print(f"✓ Vocabulary loaded successfully!")
        print(f"  Vocabulary size: {len(word2no)}")
        print(f"  Total unique words: {len(uni_words)}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading vocabulary: {str(e)}")
        return False


def load_model_and_features():
    """Load pre-trained model and feature extractors"""
    global model, newmodel, trainconvolve, testconvolve, maxlen
    
    print("Loading model and feature extractor...")
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            try:
                model_input_shape = model.input_shape
                if isinstance(model_input_shape, list):
                    seq_shape = model_input_shape[0]
                else:
                    seq_shape = model_input_shape
                if seq_shape is not None and len(seq_shape) > 1:
                    maxlen = int(seq_shape[1])
                    print(f"✓ Inferred max sequence length from model: {maxlen}")
                else:
                    print("⚠️ Could not infer max sequence length from model input shape")
            except Exception as e:
                print(f"⚠️ Error inferring model maxlen: {e}")
                maxlen = 39
        else:
            print(f"✗ Model file not found: {MODEL_PATH}")
            return False
        
        try:
            from tensorflow.keras.applications.resnet50 import ResNet50
            resnet_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
            newmodel = resnet_model.__class__(resnet_model.inputs, resnet_model.layers[-2].output)
            print("✓ ResNet50 feature extractor loaded")
        except Exception as e:
            print(f"Note: ResNet50 will be created on-the-fly: {str(e)}")
            newmodel = None
        
        if os.path.exists(CONVOLVED_FEATURES_TRAIN):
            with open(CONVOLVED_FEATURES_TRAIN, 'rb') as f:
                trainconvolve = pickle.load(f)
                print(f"✓ Training features loaded ({len(trainconvolve)} images)")
        
        if os.path.exists(CONVOLVED_FEATURES_TEST):
            with open(CONVOLVED_FEATURES_TEST, 'rb') as f:
                testconvolve = pickle.load(f)
                print(f"✓ Test features loaded ({len(testconvolve)} images)")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
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
        print(f"Error preprocessing image: {str(e)}")
        return None


def extract_image_features(img_path):
    """Extract features from image using ResNet50"""
    try:
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            return None
        
        if newmodel is not None:
            img_features = newmodel.predict(preprocessed)
        else:
            from tensorflow.keras.applications.resnet50 import ResNet50
            from tensorflow.keras.models import Model
            resnet = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
            feature_extractor = Model(resnet.inputs, resnet.layers[-2].output)
            img_features = feature_extractor.predict(preprocessed)
        
        img_features = img_features.reshape(-1)
        return img_features
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None


def generate_caption(image_features, use_sampling=False):
    """Generate caption for given image features"""
    global model, word2no, no2word, vocabsize, maxlen
    
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
            
            if use_sampling:
                indices = np.arange(predicted_array.shape[1])
                predicted_no = np.random.choice(indices, p=predicted_array[0, :])
            else:
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
        print(f"Error generating caption: {str(e)}")
        return f"Error generating caption: {str(e)}"


def text_to_speech(text):
    """Convert text to speech with timeout"""
    output_file = "caption_audio.wav"
    tts_result = [None]

    def _do_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            tts_result[0] = output_file
        except Exception as e:
            print(f"TTS thread error: {e}")
            tts_result[0] = None

    try:
        t = threading.Thread(target=_do_tts)
        t.start()
        t.join(timeout=8)
        if t.is_alive():
            print("⚠️ TTS timed out, skipping audio")
        if os.path.exists(output_file):
            return output_file
        return None
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None


def load_image_from_url(url):
    """Load image from a URL into PIL format"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Loading image from URL: {e}")
        return None


def process_image(image_input, image_url):
    """Main function to process image and generate caption"""
    try:
        image_data = None
        if image_url:
            image_data = load_image_from_url(image_url.strip())
            if image_data is None:
                return "Error loading image from URL", None
        elif isinstance(image_input, np.ndarray) or isinstance(image_input, Image.Image):
            image_data = image_input
        else:
            return "Upload an image or enter a valid image URL", None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
        if isinstance(image_data, np.ndarray):
            Image.fromarray(image_data.astype('uint8')).save(temp_path)
        else:
            image_data.save(temp_path)

        print("Extracting image features...")
        image_features = extract_image_features(temp_path)
        if image_features is None:
            return "Error extracting image features", None

        print("Generating caption...")
        caption = generate_caption(image_features)

        print("Converting to speech...")
        audio_file = text_to_speech(caption)
        return caption, audio_file
    except Exception as e:
        return f"Error: {str(e)}", None


def clear_inputs():
    return None, ""


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    background: #000000 !important;
}

.main-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: #000000;
    color: #ffffff;
    position: relative;
    overflow-x: hidden;
}

/* Animated grid background */
.main-wrap::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridScroll 30s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes gridScroll {
    0% { transform: perspective(500px) rotateX(60deg) translateY(0); }
    100% { transform: perspective(500px) rotateX(60deg) translateY(50px); }
}

/* Glowing orbs */
.main-wrap::after {
    content: '';
    position: fixed;
    top: -20%; left: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0, 255, 255, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: orbFloat 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes orbFloat {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(50px, 30px); }
}

/* Header */
.header-bar {
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(20px);
    border-bottom: 2px solid #00ffff;
    padding: 1.5rem 3rem;
    position: relative;
    z-index: 10;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.1);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.05em;
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    text-shadow: 0 0 20px rgba(0, 255, 255, 0.5), 0 0 40px rgba(0, 255, 255, 0.2);
}

.header-subtitle {
    font-size: 1rem;
    color: #00cccc;
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 255, 255, 0.1);
    color: #00ffff;
    padding: 0.5rem 1.2rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 700;
    border: 1px solid rgba(0, 255, 255, 0.4);
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    animation: neonFlicker 4s ease-in-out infinite;
}

@keyframes neonFlicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
    52% { opacity: 1; }
    54% { opacity: 0.9; }
    56% { opacity: 1; }
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #00ffff;
    border-radius: 50%;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.4);
    animation: dotPulse 1.5s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.5); }
}

/* Content */
.content-area {
    flex: 1;
    padding: 3rem;
    display: flex;
    gap: 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
    align-items: flex-start;
    position: relative;
    z-index: 5;
}

/* Cards */
.left-panel, .right-panel {
    background: rgba(10, 10, 20, 0.9);
    border: 2px solid rgba(0, 255, 255, 0.2);
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.05), inset 0 0 20px rgba(0, 255, 255, 0.02);
    transition: all 0.3s ease;
}

.left-panel:hover, .right-panel:hover {
    border-color: rgba(0, 255, 255, 0.5);
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.1), inset 0 0 30px rgba(0, 255, 255, 0.03);
    transform: translateY(-3px);
}

.left-panel { flex: 1.1; }
.right-panel { flex: 1; }

.panel-title {
    font-size: 1rem;
    font-weight: 700;
    color: #00ffff;
    margin-bottom: 1.25rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-family: 'Orbitron', sans-serif;
    border-left: 3px solid #00ffff;
    padding-left: 0.75rem;
}

/* Image upload */
.image-upload-wrap {
    border: 2px dashed rgba(0, 255, 255, 0.3);
    border-radius: 8px;
    transition: all 0.3s ease;
    min-height: 320px;
    background: rgba(0, 255, 255, 0.02);
}

.image-upload-wrap:hover {
    border-color: rgba(0, 255, 255, 0.6);
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.1), inset 0 0 20px rgba(0, 255, 255, 0.03);
    background: rgba(0, 255, 255, 0.04);
}

.image-upload-wrap .image-container {
    border-radius: 6px !important;
}

.image-upload-wrap button {
    background: transparent !important;
    color: #00cccc !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* URL input */
.url-input input {
    background: rgba(0, 0, 0, 0.6) !important;
    border: 2px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    padding: 1rem !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.url-input input:focus {
    border-color: #00ffff !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2) !important;
    outline: none !important;
}

.url-input input::placeholder {
    color: #336666 !important;
}

/* Examples */
.examples-grid img {
    border-radius: 6px !important;
    border: 2px solid rgba(0, 255, 255, 0.2) !important;
    transition: all 0.3s ease !important;
    opacity: 0.8;
}

.examples-grid img:hover {
    border-color: #00ffff !important;
    transform: translateY(-5px) scale(1.05);
    opacity: 1;
    box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
}

/* Result */
.result-box {
    background: rgba(0, 0, 0, 0.6);
    border-radius: 8px;
    padding: 1.5rem;
    border: 2px solid rgba(0, 255, 255, 0.15);
    min-height: 120px;
    position: relative;
    overflow: hidden;
}

.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00ffff, #0088ff, transparent);
    animation: scanline 2s linear infinite;
}

@keyframes scanline {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.caption-text textarea {
    background: transparent !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 1.2rem !important;
    line-height: 1.7 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    padding: 0 !important;
    resize: none !important;
}

.caption-text textarea::placeholder {
    color: #336666 !important;
    font-weight: 400 !important;
}

/* Audio */
audio {
    border-radius: 8px;
    width: 100%;
    filter: invert(1) hue-rotate(180deg) brightness(0.9) contrast(1.2);
}

/* Action bar */
.action-bar {
    padding: 1.5rem 3rem 3rem;
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(20px);
    border-top: 2px solid rgba(0, 255, 255, 0.2);
    position: relative;
    z-index: 10;
    box-shadow: 0 -10px 40px rgba(0, 255, 255, 0.05);
}

/* Buttons */
.btn-clear {
    background: transparent !important;
    color: #00cccc !important;
    border: 2px solid rgba(0, 255, 255, 0.4) !important;
    padding: 1rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
}

.btn-clear:hover {
    background: rgba(0, 255, 255, 0.1) !important;
    border-color: #00ffff !important;
    color: #ffffff !important;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    transform: translateY(-2px);
}

.btn-submit {
    background: #00ffff !important;
    color: #000000 !important;
    border: none !important;
    padding: 1rem 3rem !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.4), 0 0 40px rgba(0, 255, 255, 0.1);
}

.btn-submit:hover {
    background: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.6), 0 0 60px rgba(0, 255, 255, 0.2);
}

.btn-submit:active {
    transform: translateY(0);
}

/* Footer */
.footer-bar {
    padding: 1.5rem 3rem;
    text-align: center;
    color: #336666;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: rgba(0, 0, 0, 0.7);
    border-top: 1px solid rgba(0, 255, 255, 0.1);
    position: relative;
    z-index: 10;
    font-family: 'Orbitron', sans-serif;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #000000;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 255, 255, 0.3);
    border-radius: 4px;
    border: 1px solid rgba(0, 255, 255, 0.1);
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 255, 255, 0.6);
}

/* Responsive */
@media (max-width: 1024px) {
    .content-area {
        flex-direction: column;
        padding: 1.5rem;
        gap: 1.5rem;
    }
    .header-bar, .action-bar, .footer-bar {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .header-title {
        font-size: 1.5rem;
    }
}
"""


def create_interface():
    """Create and configure a bold futuristic Gradio interface"""

    if not load_vocabulary_from_prepro():
        return None
    if not load_model_and_features():
        return None

    with gr.Blocks(title="Image Caption Generator") as interface:
        with gr.Column(elem_classes="main-wrap"):

            with gr.Row(elem_classes="header-bar"):
                with gr.Column(scale=3):
                    gr.Markdown("<h1 class='header-title'>Image Caption Generator</h1>")
                    gr.Markdown("<p class='header-subtitle'>AI-powered image understanding with deep learning</p>")
                with gr.Column(scale=1):
                    gr.Markdown("<div style='text-align:right'><span class='status-badge'><span class='status-dot'></span>System Online</span></div>")

            with gr.Row(elem_classes="content-area"):
                with gr.Column(scale=1, elem_classes="left-panel"):
                    gr.Markdown("<div class='panel-title'>Upload Image</div>")

                    image_input = gr.Image(
                        label="",
                        type="pil",
                        elem_classes="image-upload-wrap",
                        height=340
                    )

                    gr.Markdown("<div class='panel-title' style='margin-top:1.25rem'>Image URL</div>")
                    image_url = gr.Textbox(
                        label="",
                        placeholder="Paste image URL...",
                        elem_classes="url-input"
                    )

                    gr.Markdown("<div class='panel-title' style='margin-top:1.25rem'>Quick Examples</div>")
                    example_images = [
                        os.path.join("flicker 8k dataset/Images", img)
                        for img in [
                            "1000268201_693b08cb0e.jpg",
                            "1001773457_577c3a7d70.jpg",
                            "1002674143_1b742ab4b8.jpg",
                            "1003163366_44323f5815.jpg"
                        ]
                        if os.path.exists(os.path.join("flicker 8k dataset/Images", img))
                    ]
                    if example_images:
                        gr.Examples(examples=example_images, inputs=image_input, label="")

                with gr.Column(scale=1, elem_classes="right-panel"):
                    gr.Markdown("<div class='panel-title'>Generated Caption</div>")

                    with gr.Column(elem_classes="result-box"):
                        caption_output = gr.Textbox(
                            label="",
                            lines=4,
                            interactive=False,
                            placeholder="Your AI-generated caption will appear here...",
                            elem_classes="caption-text"
                        )

                    gr.Markdown("<div class='panel-title' style='margin-top:1.25rem'>Voice Output</div>")
                    audio_output = gr.Audio(label="", type="filepath", autoplay=False)
                    flag_output = gr.Textbox(value="", visible=False)

            with gr.Row(elem_classes="action-bar"):
                clear_btn = gr.Button("Clear", elem_classes="btn-clear")
                submit_btn = gr.Button("Generate Caption", elem_classes="btn-submit")

            with gr.Row(elem_classes="footer-bar"):
                gr.Markdown("ResNet50 + LSTM  &middot;  TensorFlow  &middot;  Gradio")

        clear_btn.click(fn=clear_inputs, inputs=None, outputs=[image_input, image_url])
        submit_btn.click(fn=process_image, inputs=[image_input, image_url], outputs=[caption_output, audio_output])

    return interface

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("Image Caption Generator - Gradio GUI")
    print("=" * 60)
    
    interface = create_interface()
    
    if interface is not None:
        print("\n" + "=" * 60)
        print("✓ Application initialized successfully!")
        print("Starting Gradio server...")
        print("=" * 60 + "\n")
        
        starting_port = 7860
        last_port = starting_port + 4
        for port in range(starting_port, last_port + 1):
            try:
                interface.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=True,
                    debug=True,
                    css=custom_css,
                    theme=gr.themes.Base()
                )
                break
            except OSError as e:
                print(f"Port {port} unavailable: {e}")
                if port == last_port:
                    raise
                print(f"Trying next port: {port + 1}")
    else:
        print("\n✗ Failed to initialize application")
        print("Please check:")
        print("  - model_19.h5 exists in current directory")
        print("  - prepro_by_raj.txt exists in current directory")
        print("  - All required packages are installed")
