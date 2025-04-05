import os
import json
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import re

# Constants and defaults
DEFAULT_TEMPERATURE = 0.5               # Controls randomness in generation (higher = more random)
DEFAULT_TOP_P = 0.9                    # Cumulative probability threshold for nucleus sampling
DEFAULT_TOP_K = 20                     # Number of highest probability tokens to consider
DEFAULT_MAX_LENGTH = 256               # Max number of tokens to generate
DEFAULT_REPETITION_PENALTY = 1.2       # Penalizes repetition of tokens (1.0 = no penalty)
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3       # Prevents repetition of n-grams of this size
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STORAGE_DIR = "./locally_stored_model"
SETTINGS_FILE = os.path.join(STORAGE_DIR, "settings.json")
MODEL_FILE = os.path.join(STORAGE_DIR, "selected_model.txt")
DEFAULT_PROMPT = "I saw him walking into the principal's office with his parents..."
DEFAULT_SYSTEM_PROMPT = (
    "You are a creative and skilled storyteller. Your task is to continue the given story prompt in a coherent, engaging, and detailed manner. "
    "Maintain the tone and context of the original prompt, introducing interesting developments, vivid descriptions, and natural dialogue where appropriate. "
    "Focus on advancing the narrative logically while keeping the reader intrigued."
)

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)

# Cache for models to avoid reloading
model_cache = {}

def is_valid_huggingface_model(model_name):
    """Check if a model exists on Hugging Face Hub."""
    try:
        api_url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(api_url)
        return response.status_code == 200
    except Exception:
        return False

def load_model(model_name):
    """Load model and tokenizer from Hugging Face."""
    if model_name in model_cache:
        return model_cache[model_name]
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

def save_settings(model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt):
    """Save current settings to file with additional parameters."""
    settings = {
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_length": max_length,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "system_prompt": system_prompt
    }
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)
    
    # Also save model name to separate file
    with open(MODEL_FILE, "w") as f:
        f.write(model_name)
    
    return f"Settings saved successfully to {SETTINGS_FILE}"

def load_saved_settings():
    """Load settings from file including the new parameters."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        return settings
    return {
        "model_name": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "max_length": DEFAULT_MAX_LENGTH,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "system_prompt": DEFAULT_SYSTEM_PROMPT
    }

def export_settings(model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt):
    """Export settings to a downloadable file."""
    settings = {
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_length": max_length,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "system_prompt": system_prompt
    }
    
    export_path = os.path.join(STORAGE_DIR, "exported_settings.json")
    with open(export_path, "w") as f:
        json.dump(settings, f, indent=2)
    
    return export_path

def import_settings(file_obj):
    """Import settings from uploaded file."""
    if file_obj is None:
        return DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_MAX_LENGTH, DEFAULT_REPETITION_PENALTY, DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_SYSTEM_PROMPT
    
    try:
        settings = json.load(file_obj)
        return (
            settings.get("model_name", DEFAULT_MODEL),
            settings.get("temperature", DEFAULT_TEMPERATURE),
            settings.get("top_p", DEFAULT_TOP_P),
            settings.get("top_k", DEFAULT_TOP_K),
            settings.get("max_length", DEFAULT_MAX_LENGTH),
            settings.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
            settings.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE),
            settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
    except Exception as e:
        raise ValueError(f"Failed to import settings: {e}")

def generate_text(model_name, temperature, top_p, top_k, prompt, max_length=100, repetition_penalty=1.2, no_repeat_ngram_size=3, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Generate text using the specified model and parameters with repetition control and system prompt."""
    try:
        if not is_valid_huggingface_model(model_name):
            return f"Error: '{model_name}' is not a valid model on Hugging Face Hub."
        
        model, tokenizer = load_model(model_name)
        
        # Get device to ensure consistency
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\nUser Prompt: {prompt}"
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Set random seed for reproducible results when parameters are the same
        torch.manual_seed(42)
        
        # Generate text with specified parameters including repetition controls
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=None
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format the result to highlight the user prompt vs. the generated text
        prompt_escaped = re.escape(prompt)
        highlighted_text = re.sub(
            f".*User Prompt: ({prompt_escaped})(.*)",  # Match after "User Prompt:" to exclude system prompt in output
            r"<b>Prompt:</b> \1<br><br><b>Generated:</b> \2", 
            generated_text,
            flags=re.DOTALL
        )
        
        return highlighted_text
    
    except Exception as e:
        return f"Error generating text: {str(e)}"

def analyze_generation(original_text):
    """Analyze the quality and diversity of generated text."""
    if not original_text or isinstance(original_text, str) and "Error" in original_text:
        return "No text to analyze"
    
    # Extract only the generated part from the HTML
    match = re.search(r'<b>Generated:</b>\s+(.*?)$', original_text, re.DOTALL)
    if not match:
        return "Could not extract generated text for analysis"
    
    generated_text = match.group(1).strip()
    
    # Count tokens (rough estimate)
    words = generated_text.split()
    token_count = len(words)
    
    # Detect repetition
    repetition_score = 0
    repeated_phrases = []
    
    # Check for repeated phrases of 3-5 words
    for n in range(3, 6):
        if len(words) < n*2:
            continue
            
        phrases = {}
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            phrases[phrase] = phrases.get(phrase, 0) + 1
        
        for phrase, count in phrases.items():
            if count > 1:
                repetition_score += count
                if len(repeated_phrases) < 3:  # Limit to 3 examples
                    repeated_phrases.append(f'"{phrase}" (x{count})')
    
    # Calculate lexical diversity (unique words / total words)
    unique_words = len(set(word.lower() for word in words))
    diversity = unique_words / max(1, len(words))
    
    # Build analysis report
    analysis = f"<h4>Generation Analysis</h4>"
    analysis += f"<p><b>Token count:</b> ~{token_count}</p>"
    analysis += f"<p><b>Lexical diversity:</b> {diversity:.2f} ({unique_words} unique words out of {len(words)})</p>"
    
    if repetition_score > 0:
        analysis += f"<p><b>Repetition detected:</b> Score of {repetition_score}</p>"
        if repeated_phrases:
            analysis += f"<p><b>Examples:</b> {', '.join(repeated_phrases)}</p>"
        analysis += "<p><i>Try increasing the repetition penalty or no-repeat n-gram size</i></p>"
    else:
        analysis += "<p><b>No significant repetition detected</b></p>"
    
    return analysis

def reset_parameters():
    """Reset parameters to default values."""
    return (
        DEFAULT_MODEL, 
        DEFAULT_TEMPERATURE, 
        DEFAULT_TOP_P, 
        DEFAULT_TOP_K, 
        DEFAULT_MAX_LENGTH, 
        DEFAULT_REPETITION_PENALTY, 
        DEFAULT_NO_REPEAT_NGRAM_SIZE,
        DEFAULT_SYSTEM_PROMPT
    )

def update_model_status(model_name):
    """Check if the model exists and update status."""
    if not model_name:
        return "Please enter a model name."
    
    valid = is_valid_huggingface_model(model_name)
    if valid:
        return f"✅ Model '{model_name}' is available on Hugging Face."
    else:
        return f"❌ Model '{model_name}' not found on Hugging Face."

# Create the Gradio interface
with gr.Blocks(title="LLM Parameter Visualization") as app:
    gr.Markdown("""
    # 🤗 Language Model Parameter Visualization
    
    This app helps you understand how different parameters affect text generation with Hugging Face language models.
    Adjust the sliders to see how temperature, top-p, and top-k change the generated text in real-time.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Selection")
            
            model_name = gr.Textbox(
                label="Model Name", 
                value=DEFAULT_MODEL,
                info="Enter a model name from Hugging Face (e.g., gpt2, distilgpt2)"
            )
            
            model_status = gr.Textbox(
                label="Model Status",
                value="Enter a model name and press Check",
                interactive=False
            )
            
            check_model_btn = gr.Button("Check Model Availability")
            
            prompt_input = gr.Textbox(
                label="Prompt", 
                value=DEFAULT_PROMPT,
                info="Enter text prompt for generation"
            )
            
            system_prompt_input = gr.Textbox(
                label="System Prompt", 
                value=DEFAULT_SYSTEM_PROMPT,
                info="Enter system prompt to guide the model's behavior",
                lines=4
            )
            
            gr.Markdown("### Model Parameters")
            
            with gr.Accordion("Parameter Explanations", open=False):
                gr.Markdown("""
                - **Temperature**: Controls randomness. Higher values (e.g., 1.5) make output more random, lower values (e.g., 0.2) make it more deterministic.
                - **Top-p (Nucleus Sampling)**: Cumulative probability threshold. Only tokens whose cumulative probability exceeds top-p are considered. Lower values make output more focused.
                - **Top-k**: Limits the set of tokens to consider to the k highest probability tokens. Lower values restrict creativity but increase coherence.
                - **Max Length**: Maximum number of tokens to generate. Higher values allow for longer responses.
                - **Repetition Penalty**: Discourages the model from repeating the same phrases. Values above 1.0 penalize repetitions, with higher values imposing stronger penalties.
                - **No Repeat N-gram Size**: Prevents the model from generating the same sequence of N tokens. Setting to 3 prevents repeating any 3-token sequence.
                - **System Prompt**: Instructions to guide the model's behavior and style.
                """)
            
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=DEFAULT_TEMPERATURE, 
                step=0.1, 
                label="Temperature",
                info="Controls randomness (higher = more random)"
            )
            
            top_p = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=DEFAULT_TOP_P, 
                step=0.05, 
                label="Top-p (Nucleus Sampling)",
                info="Cumulative probability threshold"
            )
            
            top_k = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=DEFAULT_TOP_K, 
                step=1, 
                label="Top-k",
                info="Number of highest probability tokens to consider"
            )
            
            with gr.Accordion("Advanced Parameters", open=False):
                max_length = gr.Slider(
                    minimum=10, 
                    maximum=500, 
                    value=100, 
                    step=10, 
                    label="Max Length",
                    info="Maximum number of tokens to generate"
                )

                repetition_penalty = gr.Slider(
                    minimum=1.0, 
                    maximum=2.0, 
                    value=1.2, 
                    step=0.05, 
                    label="Repetition Penalty",
                    info="Higher values penalize repeated tokens more (1.0 = no penalty)"
                )

                no_repeat_ngram_size = gr.Slider(
                    minimum=0, 
                    maximum=10, 
                    value=3, 
                    step=1, 
                    label="No Repeat N-gram Size",
                    info="Prevent repetition of n-grams of this size (0 = disabled)"
                )
            
            with gr.Row():
                reset_btn = gr.Button("Reset to Defaults")
                generate_btn = gr.Button("Generate Text", variant="primary")
            
            with gr.Accordion("Save & Load Settings", open=True):
                save_btn = gr.Button("Save Current Settings")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                
                export_btn = gr.Button("Export Settings")
                export_file = gr.File(label="Download Settings", interactive=False)
                
                upload_file = gr.File(label="Import Settings")
                import_btn = gr.Button("Load Imported Settings")
        
        with gr.Column(scale=1):
            gr.Markdown("### Generated Text Output")
            output_text = gr.HTML(
                label="Generated Text", 
                value="Adjust parameters and click 'Generate Text' to see results."
            )

        with gr.Column(scale=1):
            gr.Markdown("### Output Analysis")
            output_analysis = gr.HTML(label="Analysis", value="Generate text to see analysis")
    
    # Setup event handlers
    check_model_btn.click(update_model_status, inputs=[model_name], outputs=[model_status])
    
    generate_btn.click(
        lambda *args: [generate_text(*args), analyze_generation(generate_text(*args))],
        inputs=[model_name, temperature, top_p, top_k, prompt_input, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt_input],
        outputs=[output_text, output_analysis]
    )

    
    reset_btn.click(
        reset_parameters,
        inputs=[],
        outputs=[model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt_input]
    )
    
    save_btn.click(
        save_settings,
        inputs=[model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt_input],
        outputs=[save_status]
    )
    
    export_btn.click(
        export_settings,
        inputs=[model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt_input],
        outputs=[export_file]
    )
    
    import_btn.click(
        import_settings,
        inputs=[upload_file],
        outputs=[model_name, temperature, top_p, top_k, max_length, repetition_penalty, no_repeat_ngram_size, system_prompt_input]
    )
    
    # Load settings on startup
    saved_settings = load_saved_settings()
    model_name.value = saved_settings["model_name"]
    temperature.value = saved_settings["temperature"]
    top_p.value = saved_settings["top_p"]
    top_k.value = saved_settings["top_k"]
    max_length.value = saved_settings.get("max_length", DEFAULT_MAX_LENGTH)
    repetition_penalty.value = saved_settings.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
    no_repeat_ngram_size.value = saved_settings.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)
    system_prompt_input.value = saved_settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

# Launch the app
if __name__ == "__main__":
    app.launch()