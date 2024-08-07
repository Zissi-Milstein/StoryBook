import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load text generation model from Hugging Face
text_generator = pipeline('text-generation', model='gpt-2')

# Load Stable Diffusion model for image generation
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

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
            illustration = generate_image(detailed_prompt)
            st.image(illustration)
    else:
        st.error("Please provide a story idea or text.")
