import csv
import os
import json
import google.generativeai as genai
import argparse
from collections import Counter
import re
from dotenv import load_dotenv
import time

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Ensure your Google API Key is set as an environment variable:
# export GOOGLE_API_KEY="YOUR_API_KEY"
# Or configure it directly:
# genai.configure(api_key="YOUR_API_KEY")
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("🚨 Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    # You might want to exit here or provide a way to input the key manually
    # For this example, we'll proceed assuming it might be configured elsewhere
    # or the user will handle it.
    pass # Allow script to continue, genai calls will fail if not configured

# Configure the Gemini model
# Update model_name if needed, e.g., 'gemini-1.5-flash'
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 256, # Adjust as needed for snippet length
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest", # Or another suitable model
    generation_config=generation_config,
    # safety_settings=... # Optional: Add safety settings if needed
)

# --- Helper Functions ---

def clean_text(text):
    """Removes punctuation and converts to lowercase."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def get_keywords(text, num_keywords=10):
    """Extracts simple keywords (most frequent words, excluding common stop words)."""
    # Very basic English stop words list - consider using a library like NLTK for better results
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
        "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
        "weren", "won", "wouldn", "also", "like", "get", "make", "use"
    ])
    words = clean_text(text).split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(num_keywords)]

# --- Core Functions ---

def read_customer_segments(csv_filepath):
    """
    Reads customer segment data from a CSV file.

    Args:
        csv_filepath (str): Path to the input CSV file.
                            Expected columns: 'segment_name', 'key_interests',
                                              'preferred_language', 'content_examples_they_like'

    Returns:
        list: A list of dictionaries, where each dictionary represents a customer segment.
              Returns an empty list if the file cannot be read or is improperly formatted.
    """
    segments = []
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            # Check for required columns (case-insensitive check)
            required_cols = {'segment_name', 'key_interests', 'preferred_language', 'content_examples_they_like'}
            if not required_cols.issubset(set(map(str.lower, reader.fieldnames))):
                print(f"🚨 Error: CSV file '{csv_filepath}' is missing one or more required columns.")
                print(f"Required columns: {', '.join(required_cols)}")
                return []
            for row in reader:
                # Normalize keys to lowercase for consistency
                normalized_row = {k.lower(): v for k, v in row.items()}
                segments.append(normalized_row)
        print(f"✅ Successfully read {len(segments)} segments from '{csv_filepath}'.")
        return segments
    except FileNotFoundError:
        print(f"🚨 Error: Input CSV file not found at '{csv_filepath}'.")
        return []
    except Exception as e:
        print(f"🚨 Error reading CSV file '{csv_filepath}': {e}")
        return []

def analyze_content_directory(content_dir):
    """
    Analyzes text files in a directory to extract basic information.

    Args:
        content_dir (str): Path to the directory containing .txt content files.

    Returns:
        dict: A dictionary where keys are filenames and values are dictionaries
              containing 'filepath', 'keywords', and potentially 'summary' (future).
              Returns an empty dict if the directory doesn't exist or no .txt files are found.

    Note:
        This currently extracts simple keywords. For deeper understanding (themes,
        sentiment, target audience), consider using the LLM itself to analyze each
        document in a separate step or enhancing the keyword extraction.
    """
    analyzed_content = {}
    if not os.path.isdir(content_dir):
        print(f"🚨 Error: Content directory not found at '{content_dir}'.")
        return {}

    print(f"Analyzing content in directory: '{content_dir}'...")
    found_files = False
    for filename in os.listdir(content_dir):
        if filename.lower().endswith(".txt"):
            found_files = True
            filepath = os.path.join(content_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    keywords = get_keywords(content)
                    analyzed_content[filename] = {
                        "filepath": filepath,
                        "keywords": keywords,
                        # Placeholder for future enhancement:
                        # "summary": generate_llm_summary(content)
                    }
                    print(f"  - Analyzed '{filename}' (Keywords: {keywords[:3]}...)")
            except Exception as e:
                print(f"🚨 Error processing file '{filename}': {e}")

    if not found_files:
        print(f"⚠️ Warning: No .txt files found in '{content_dir}'.")
    elif analyzed_content:
        print(f"✅ Content analysis complete. Processed {len(analyzed_content)} files.")
    else:
         print(f"⚠️ Warning: No .txt files could be successfully processed in '{content_dir}'.")

    return analyzed_content

def generate_few_shot_prompt(segment, analyzed_content, content_type):
    """
    Constructs a few-shot prompt for the LLM.

    Args:
        segment (dict): A dictionary representing a customer segment.
        analyzed_content (dict): The dictionary of analyzed content information.
        content_type (str): The specific type of content to generate
                            (e.g., 'email subject line', 'short social media post').

    Returns:
        str: The formatted prompt string for the LLM.
    """
    prompt = f"**Task:** Generate a personalized '{content_type}' for a specific customer segment.\n\n"

    prompt += "**Customer Segment Information:**\n"
    prompt += f"- Segment Name: {segment.get('segment_name', 'N/A')}\n"
    prompt += f"- Key Interests: {segment.get('key_interests', 'N/A')}\n"
    prompt += f"- Preferred Language Style/Tone: {segment.get('preferred_language', 'neutral')}\n"
    prompt += f"- Examples of Content They Like: {segment.get('content_examples_they_like', 'N/A')}\n\n"

    prompt += "**Available Content Insights:**\n"
    if analyzed_content:
        prompt += "We have existing content pieces. Here are some keywords from them:\n"
        # Select a few relevant content pieces based on keyword overlap (simple example)
        relevant_keywords = []
        segment_interests = set(segment.get('key_interests', '').lower().split(','))
        count = 0
        for filename, data in analyzed_content.items():
            if count < 3 and segment_interests.intersection(set(data.get('keywords', []))):
                 prompt += f"- From '{filename}': {', '.join(data.get('keywords', []))}\n"
                 count += 1
        if count == 0: # Fallback if no overlap found
             prompt += "- General content keywords: " + ", ".join(
                 list(analyzed_content.values())[0].get('keywords', []))[:100] + "...\n" # Show some keywords anyway
    else:
        prompt += "No existing content analysis available.\n"

    prompt += "\n**Instruction:**\n"
    prompt += f"Based on the customer segment's interests and preferences, and considering the style of content they like, generate **one** engaging and personalized '{content_type}'.\n"
    prompt += f"Use a {segment.get('preferred_language', 'neutral')} tone.\n"
    prompt += "The output should be **only the generated snippet string**, without any extra labels or explanations.\n\n"

    # --- Few-Shot Examples (Crucial for better results) ---
    # You should replace these with *actual* good examples relevant to your domain.
    # These examples teach the model the desired output format and style.
    prompt += "**Examples:**\n\n"
    prompt += "*(Example 1 - Input like above)*\n"
    prompt += "*(Output)*\n"
    prompt += "Subject: ✨ Unlock Exclusive Insights for Tech Leaders!\n\n"

    prompt += "*(Example 2 - Input like above)*\n"
    prompt += "*(Output)*\n"
    prompt += "🚀 Sneak Peek: Our Latest Gadget is Here!\n\n"

    prompt += "*(Example 3 - Input for a different segment/type)*\n"
    prompt += "*(Output)*\n"
    prompt += "Find your zen with our new meditation guide. 🧘‍♀️ #mindfulness #wellness\n\n"

    prompt += "**Generate Content Now:**\n"
    prompt += "*(Output)*\n" # Signal to the model where its output should go

    # print(f"\n--- Generated Prompt for Segment: {segment.get('segment_name')} ---\n{prompt}\n-----------------------------\n") # Uncomment for debugging
    return prompt


def call_gemini_api(prompt):
    """
    Sends the prompt to the Gemini API and gets the generated content.

    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        str: The generated text snippet, or None if an error occurs.
    """
    try:
        response = model.generate_content(prompt)
        # Basic check if the response has the expected structure
        if response.parts:
             generated_text = response.text.strip()
             # Simple post-processing: remove potential markdown/quotes if model adds them
             generated_text = generated_text.replace("```", "").replace('"', '').strip()
             return generated_text
        else:
             print("🚨 Error: Received an unexpected response structure from the API.")
             # Log or inspect response.prompt_feedback or other attributes if needed
             # print(f"API Response Feedback: {response.prompt_feedback}")
             return None
    except Exception as e:
        print(f"🚨 Error calling Gemini API: {e}")
        # Consider more specific error handling based on potential API errors
        return None

def save_results_to_json(results, json_filepath):
    """
    Saves the list of personalized snippets to a JSON file.

    Args:
        results (list): A list of dictionaries, each containing personalized content info.
        json_filepath (str): Path to the output JSON file.
    """
    try:
        with open(json_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved {len(results)} personalized snippets to '{json_filepath}'.")
    except Exception as e:
        print(f"🚨 Error writing JSON file '{json_filepath}': {e}")

# --- Main Execution ---

def main():
    """Main function to orchestrate the personalization process."""
    parser = argparse.ArgumentParser(description="AI Content Personalization Assistant")
    parser.add_argument("csv_file", help="Path to the input CSV file with customer segments.")
    parser.add_argument("content_dir", help="Path to the directory containing .txt marketing content files.")
    parser.add_argument("output_json", help="Path for the output JSON file.")
    parser.add_argument("--content_type", default="email subject line",
                        help="Type of content to generate (e.g., 'email subject line', 'short social media post').")

    args = parser.parse_args()

    print("🚀 Starting AI Content Personalization Assistant...")

    # 1. Read Customer Segments
    print("\n--- Step 1: Reading Customer Segments ---")
    customer_segments = read_customer_segments(args.csv_file)
    if not customer_segments:
        print("❌ Aborting due to issues reading customer segments.")
        return

    # 2. Analyze Marketing Content
    print("\n--- Step 2: Analyzing Content Directory ---")
    analyzed_content = analyze_content_directory(args.content_dir)
    # Proceed even if analysis fails or finds nothing, but LLM results might be weaker.

    # 3. Generate Personalized Content
    print(f"\n--- Step 3: Generating Personalized Content ({args.content_type}) ---")
    personalized_results = []
    if not customer_segments:
        print("No customer segments to process.")
    else:
        for i, segment in enumerate(customer_segments):
            segment_name = segment.get('segment_name', f'Segment_{i+1}')
            print(f"  Generating content for segment: '{segment_name}'...")

            # 3a. Construct Prompt
            prompt = generate_few_shot_prompt(segment, analyzed_content, args.content_type)

            # Add a delay to avoid exceeding API quota
            time.sleep(2)

            # 3b. Call LLM API
            snippet = call_gemini_api(prompt)

            if snippet:
                print(f"    ✅ Generated Snippet: {snippet}")
                personalized_results.append({
                    "segment_name": segment_name,
                    "content_type": args.content_type,
                    "personalized_snippet": snippet
                })
            else:
                print(f"    ⚠️ Failed to generate snippet for segment: '{segment_name}'.")

    # 4. Save Results
    print("\n--- Step 4: Saving Results ---")
    if personalized_results:
        save_results_to_json(personalized_results, args.output_json)
    else:
        print("⚠️ No personalized snippets were generated. Nothing to save.")

    print("\n✨ Personalization process finished! ✨")

if __name__ == "__main__":
    main()
