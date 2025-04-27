# app.py
import streamlit as st
import requests # Still needed for the non-streaming reasoning capture
import json
import os
from openai import OpenAI # Use OpenAI client for xAI API
from dotenv import load_dotenv
import time # Optional: for adding a small delay

# Load environment variables from .env file
load_dotenv()
# Use XAI_API_KEY for xAI API
XAI_API_KEY = os.getenv("XAI_API_KEY")

# --- Configuration ---
# Use the xAI API base URL
XAI_BASE_URL = "https://api.x.ai/v1"

MINI_MODEL = "grok-3-mini-latest" # Model for its own output and reasoning capture
LARGE_MODEL = "grok-3-latest"   # Larger model for baseline and injected reasoning answers

# --- Helper Functions ---

def get_reasoning_content_non_streaming(question_text):
    """
    Calls grok-3-mini-latest non-streamingly to reliably capture reasoning_content
    using reasoning_effort='high'. Uses requests library.
    """
    if not XAI_API_KEY:
        st.error("XAI_API_KEY not found.")
        return None

    try:
        # Use requests directly for this one non-streaming call
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
            # You might need a referer header for xAI API too, check their docs
            # "HTTP-Referer": "https://streamlit-reasoning-compare.streamlit.app/"
        }

        payload = {
            "model": MINI_MODEL,
            "messages": [
                {"role": "user", "content": question_text}
            ],
            "reasoning_effort": "low" # Request high reasoning effort
            # Do NOT set stream=True here
        }

        st.info(f"Requesting reasoning content from {MINI_MODEL} (non-streaming)...")
        response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses

        response_data = response.json()

        if response_data.get('choices') and response_data['choices'][0].get('message'):
             message = response_data['choices'][0]['message']
             # This is where we get the explicit reasoning_content field
             reasoning_text = message.get('reasoning_content', '')

             if reasoning_text:
                 st.success("Reasoning content captured!")
                 return reasoning_text.strip()
             else:
                 st.warning(f"No explicit 'reasoning_content' field found in {MINI_MODEL} response.")
                 # Optionally show response for debugging if needed: st.json(response_data)
                 return "" # Return empty string if not found, but call was successful

        else:
            st.warning(f"No choices or message found in {MINI_MODEL} response for reasoning capture.")
            # Optionally show response for debugging if needed: st.json(response_data)
            return None # Indicate failure to get a valid response structure

    except requests.exceptions.RequestException as e:
        st.error(f"Error capturing reasoning from {MINI_MODEL}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during reasoning capture: {e}")
        return None


def stream_model_output(model, content, placeholder):
    """
    Streams the response (content chunks) from the specified model using the OpenAI client.
    Writes delta.content to a Streamlit placeholder.
    """
    if not XAI_API_KEY:
        placeholder.error("API key missing.")
        return None

    try:
        # Use the OpenAI client configured for the xAI API
        # THIS IS WHERE THE ERROR IS OCCURRING based on your screenshot.
        # The error "Client.init() got an unexpected keyword argument 'proxies'"
        # means something is passing 'proxies' to this line:
        client = OpenAI(
            base_url=XAI_BASE_URL,
            api_key=XAI_API_KEY,
            # There is no 'proxies' argument here in the code I provided.
            # If the error persists, it's likely due to environment variables
            # like HTTP_PROXY or HTTPS_PROXY being set, which the openai library
            # is trying to use in a way incompatible with its current version's __init__.
        )

        # The streaming call primarily outputs delta.content
        # For the mini model, setting reasoning_effort might influence
        # the content stream itself, but we won't rely on separate reasoning chunks here.
        # For the large model, we just stream the answer based on the prompt.
        # Only include reasoning_effort='high' if we want the model to *think* more
        # even when streaming its final answer. Let's omit it for the answer streams
        # and rely on prompt injection for the 'smart' case.
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            stream=True,
            # reasoning_effort="high" # Omit for standard answer streaming
            # max_tokens=... # Consider adding a max_tokens limit
        )

        full_response = ""
        # The stream object from openai client iterates over completion_create_chunk objects
        for chunk in stream:
             # Check if the chunk has choices and a delta
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                # Extract content chunks
                delta_content = delta.content
                if delta_content:
                    full_response += delta_content
                    # Update the placeholder with accumulated text + cursor
                    placeholder.markdown(full_response + "▌")
                    # time.sleep(0.005) # Optional delay

            # Check for potential refusal or error in the chunk (less common in delta but possible)
            # if hasattr(chunk.choices[0], 'refusal') and chunk.choices[0].refusal:
            #     st.warning(f"Refusal detected: {chunk.choices[0].refusal}")
            #     break # Or handle as needed

        # After streaming, remove cursor and return the full response
        placeholder.markdown(full_response)
        return full_response.strip()

    except Exception as e:
        st.error(f"Error streaming response from {model}: {e}")
        placeholder.error(f"Error: {e}")
        return None


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="LLM Reasoning & Answer Comparison (xAI Grok)")

st.title("LLM Reasoning & Answer Comparison (xAI Grok Models)")
st.markdown(f"""
This app compares how different xAI Grok models answer a question:
1.  The smaller model (`{MINI_MODEL}`) acting alone (its natural streamed output).
2.  The larger model (`{LARGE_MODEL}`) acting alone (baseline streaming).
3.  The larger model (`{LARGE_MODEL}`) with explicit `reasoning_content` captured from `{MINI_MODEL}` injected into the prompt (streaming).

The `reasoning_content` from `{MINI_MODEL}` is captured via a separate, non-streaming call to ensure reliability based on xAI API examples.
""")

# User Input for the Question
user_question = st.text_area(
    "Enter your question here:",
    height=100,
    placeholder="e.g., A is taller than B, and B is shorter than C. Is A definitely taller than C? Explain.",
    key="user_question_input"
)

# --- Trigger the process ---
if st.button("Run Comparison"):
    if not XAI_API_KEY:
         st.error("xAI API key not found. Please set the XAI_API_KEY environment variable or add it to a `.env` file.")
    elif not user_question or user_question.strip() == "":
        st.warning("Please enter a question before running the comparison.")
    else:
        question_text = user_question.strip()
        st.info(f"Processing the following question:\n\n**{question_text}**")

        # Step 1: Reliably capture reasoning content from the mini model (non-streaming)
        st.subheader(f"Step 1: Capture Reasoning Content from {MINI_MODEL}")
        with st.spinner(f"Attempting to capture explicit reasoning content from {MINI_MODEL}..."):
             # Use the non-streaming function to get the specific reasoning field
             captured_reasoning = get_reasoning_content_non_streaming(question_text)

        # --- Step 2: Run and stream the three comparison answers ---
        # This step runs REGARDLESS of whether reasoning was captured,
        # but the 3rd column will note if reasoning wasn't injected.

        st.subheader("Step 2: Compare Answers (Streaming)")

        # Create three columns for side-by-side display
        col1, col2, col3 = st.columns(3)

        # --- Column 1: Grok-3-mini's natural streamed output ---
        with col1:
            st.write(f"### {MINI_MODEL} (Direct Output)")
            # Note the prompt description reflects that reasoning_effort is used for the *request*,
            # but the streamed output is delta.content
            st.write("Prompt: Just the question (request includes `reasoning_effort='high'`).")
            st.write("Generating...")
            mini_output_placeholder = st.empty() # Placeholder for streamed output
            st.write("---")

        # --- Column 2: Grok-3 Baseline answer ---
        with col2:
            st.write(f"### {LARGE_MODEL} (Baseline)")
            st.write("Prompt: Just the question.")
            st.write("Generating...")
            large_baseline_placeholder = st.empty() # Placeholder for streamed output
            st.write("---")

        # --- Column 3: Grok-3 answer with injected reasoning ---
        with col3:
            st.write(f"### {LARGE_MODEL} (With Injected Reasoning)")
            if captured_reasoning:
                 st.write(f"Prompt: Question + Reasoning from {MINI_MODEL}.")
            else:
                 st.write(f"Prompt: Just the question (Reasoning capture failed).") # Indicate failure in prompt description
            st.write("Generating...")
            large_smart_placeholder = st.empty() # Placeholder for streamed output
            st.write("---")

        # Start streaming answers in each column (These calls block their columns)

        # Grok-3-mini direct answer (streamed output)
        with col1:
             # We don't need the return value here, just stream the output
             # Note: The reasoning_effort param for the API call itself is handled in stream_model_output
             # if you uncommented it there, but I removed it to focus stream on content.
             # The non-streaming call was solely for capturing the *specific* field.
             # So this call uses the standard stream_model_output focusing on content.
             stream_model_output(MINI_MODEL, question_text, mini_output_placeholder)
             st.write("Generation Complete.")

        # Grok-3 baseline answer
        with col2:
             # We don't need the return value here, just stream the output
             stream_model_output(LARGE_MODEL, question_text, large_baseline_placeholder)
             st.write("Generation Complete.")

        # Grok-3 answer with injected reasoning
        with col3:
             # Create the combined prompt for the 'smart' response
             injected_prompt_content = question_text
             if captured_reasoning: # Only inject if reasoning was successfully captured (not None and not empty string)
                 injected_prompt_content = f"""You are answering the following question:

{question_text}

Helpful Thinking Process:

{captured_reasoning}

Instructions:
- Carefully review the Helpful Thinking Process provided above.
- Thoughtfully incorporate useful insights from it into your final answer when appropriate.
- If the Helpful Thinking Process is incomplete or partially incorrect, intelligently improve upon it before answering.
- Always prioritize logical rigor, clarity, and accuracy.
- Keep your response self-contained, addressing the question fully without relying on unstated assumptions.
- Do not fabricate facts or steps; reason from what is given and what logically follows.
- Adjust your tone to match the question's subject (e.g., formal for technical topics, conversational for open-ended ones).
- Produce the best, most complete answer you can based on both the question and the Helpful Thinking Process

Now, carefully produce your answer."""
                  # If no explicit reasoning was captured, the prompt content remains just the original question.
                  st.warning(f"No explicit reasoning content was captured from {MINI_MODEL}. The injected prompt for {LARGE_MODEL} contains only the original question.")


             # We don't need the return value here, just stream the output
             stream_model_output(LARGE_MODEL, injected_prompt_content, large_smart_placeholder)
             st.write("Generation Complete.")


st.markdown("---")
st.write("Note: The quality of generated reasoning and final answers depends on the specific models and prompts used.")
st.write("Make sure `XAI_API_KEY` is set in your environment or a `.env` file.")
st.write(f"Reasoning content captured from: `{MINI_MODEL}` (non-streaming API call with `reasoning_effort='high'`). Answers compared for: `{MINI_MODEL}` (natural streamed output), `{LARGE_MODEL}` (baseline streamed), and `{LARGE_MODEL}` (with captured reasoning injected, streamed).")