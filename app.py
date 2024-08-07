import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Ensure the use of 'accelerate' for better performance
try:
    import accelerate
except ImportError:
    st.warning("accelerate is not installed. Install it for better performance.")

# Load text generation model from Hugging Face
try:
    text_generator = pipeline('text-generation', model='gpt2')
except EnvironmentError as e:
    st.error(f"Failed to load the text generation model: {e}")

# Load Stable Diffusion model for image generation
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"

try:
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32 if device == "cpu" else torch.float16
    ).to(device)
except EnvironmentError as e:
    st.error(f"Failed to load the image generation model: {e}")

def generate_image(prompt):
    with torch.no_grad():
        image = sd_pipeline(prompt).images[0]
    return image

st.title("Illustrated Short Storybook Generator")

# Option to provide your own text or generate text using GPT-2
st.header("Step 1: Choose Your Text Source")
text_option = st.radio("Select the source of the text:", ("Generate text with GPT-2", "Provide your own text"))

if text_option == "Generate text with GPT-2":
    st.header("Provide a story idea for GPT-2")
    story_idea = st.text_area("Enter your story idea:", height=100)
else:
    st.header("Provide your own text")
    user_text = st.text_area("Enter your text:", height=300)

# Option to select a theme
st.header("Step 2: Choose a theme")
theme = st.selectbox("Select a theme for your story:", ["Fantasy", "Adventure", "Mystery", "Sci-Fi"])

# Option to describe the main character for consistency
st.header("Step 3: Describe the main character")
character_description = st.text_input("Enter a description for the main character:", "A brave knight with a red cape and a shining sword")

if st.button("Generate Story"):
    if text_option == "Generate text with GPT-2":
        if story_idea:
            if 'text_generator' in globals():
                # Generate the story based on the provided idea or text
                generated_story = text_generator(story_idea, max_length=300, num_return_sequences=1)[0]['generated_text']
                st.subheader("Generated Short Story")
                st.write(generated_story)
            else:
                st.error("Text generation model is not available.")
        else:
            st.error("Please provide a story idea for GPT-2.")
        story_text = generated_story
    else:
        if user_text:
            story_text = user_text
            st.subheader("Provided Text")
            st.write(story_text)
        else:
            st.error("Please provide your own text.")
    
    # Generate illustrations for the story
    if story_text:
        st.subheader("Illustrations")
        story_parts = story_text.split('. ')  # Split the story into sentences or short segments
        for i, part in enumerate(story_parts):
            if part.strip():
                st.write(part.strip())
                detailed_prompt = f"{theme} {part.strip()}. The main character is {character_description}."
                if 'sd_pipeline' in globals():
                    illustration = generate_image(detailed_prompt)
                    st.image(illustration, caption=f"Illustration for segment {i+1}")
                else:
                    st.error("Image generation model is not available.")
