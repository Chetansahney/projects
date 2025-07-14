
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

# 🔑 Hugging Face token (hardcoded for testing)
HF_TOKEN = "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"  # Replace with your token

client = InferenceClient(api_key=HF_TOKEN)

st.set_page_config(page_title="AI Image Generator")
st.title("🎨 AI Image Generator")

prompt = st.text_input("Enter your prompt:", "a soft wallpaper")

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.write("Sending prompt to Hugging Face model...")
        with st.spinner("Generating image..."):
            try:
                image = client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-dev",
                )
                st.success("Image generated successfully!")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
