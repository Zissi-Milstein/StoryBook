import streamlit as st
from transformers import pipeline, set_seed
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import accelerate

# Load text generation model from Hugging Face
try:
    text_generator = pipeline('text-generation', model='gpt2')
except EnvironmentError as e:
    st.error(f"Failed to load the text generation model: {e}")

# Load Stable Diffusion model for image generation
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16", use_auth_token=True
    ).to(device)
except EnvironmentError as e:
    st.error(f"Failed to load the image generation model: {e}")

def generate_image(prompt):
    with torch.no_grad():
        image = sd_pipeline(prompt).images[0]
    return image

st.title("Illustrated Storybook Generator")

# Option to provide your own text or a story idea
st.header("Step 1: Provide a story idea or your own text")
story_idea = st.text_area("Enter your story idea or text:", height=200)

# Option to select a theme
st.header("Step 2: Choose a theme")
theme = st.selectbox("Select a theme for your story:", ["Fantasy", "Adventure", "Mystery", "Sci-Fi"])

# Option to describe the main character for consistency
st.header("Step 3: Describe the main character")
character_description = st.text_input("Enter a description for the main character:", "A brave knight with a red cape and a shining sword")

if st.button("Generate Story"):
    if story_idea:
        if 'text_generator' in globals():
            # Generate the story based on the provided idea or text
            generated_story = text_generator(story_idea, max_length=512, num_return_sequences=1)[0]['generated_text']
            
            st.subheader("Generated Story")
            st.write(generated_story)
            
            # Generate illustrations for each part of the story
            st.subheader("Illustrations")
            story_parts = generated_story.split('\n\n')  # Split the story into parts
            for part in story_parts:
                st.write(part)
                detailed_prompt = f"{theme} {part}. The main character is {character_description}."
                if 'sd_pipeline' in globals():
                    illustration = generate_image(detailed_prompt)
                    st.image(illustration)
                else:
                    st.error("Image generation model is not available.")
        else:
            st.error("Text generation model is not available.")
    else:
        st.error("Please provide a story idea or text.")
