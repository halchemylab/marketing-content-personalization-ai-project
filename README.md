# AI Content Personalization Assistant

This project is an AI-powered tool designed to generate personalized marketing content for different customer segments using Google's Gemini AI model. It analyzes existing marketing content, processes customer segment data, and creates tailored content snippets such as email subject lines or social media posts.

This tool is particularly helpful in marketing as it enables businesses to create highly targeted and engaging content that resonates with specific customer segments. By leveraging AI to analyze customer preferences and existing content, marketers can save time, improve campaign effectiveness, and foster stronger connections with their audience.

## Features
- **Customer Segmentation**: Reads customer segment data from a CSV file.
- **Content Analysis**: Extracts keywords from existing marketing content.
- **AI-Powered Personalization**: Uses Google's Gemini AI to generate personalized content snippets.
- **Customizable Content Types**: Supports generating various content types (e.g., email subject lines, social media posts).
- **Output Management**: Saves generated snippets to a JSON file for easy access.

---

## Prerequisites
1. **Python**: Ensure Python 3.7 or later is installed.
2. **Google API Key**: Obtain an API key for Google's Gemini AI and set it as an environment variable:
   ```bash
   export GOOGLE_API_KEY="YOUR_API_KEY"
   ```
3. **Dependencies**: Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd marketing-content-personalization-ai-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Google API key:
   - Add your API key to an `.env` file in the project root:
     ```
     GOOGLE_API_KEY=YOUR_API_KEY
     ```

---

## Usage
Run the script using the following command:
```bash
python marketing_content_generation_agent.py segments.csv marketing_content/ output_snippets.json
```

### Arguments:
- `segments.csv`: Path to the CSV file containing customer segment data.
- `marketing_content/`: Directory containing `.txt` files with existing marketing content.
- `output_snippets.json`: Path to save the generated personalized snippets.

### Optional Argument:
- `--content_type`: Specify the type of content to generate (default: "email subject line").

---

## Example
### Input:
- **CSV File**: `segments.csv`
  ```csv
  segment_name,key_interests,preferred_language,content_examples_they_like
  Small Business Owners,local hiring,professional,"Hire local talent, Save costs"
  Tech Startups,innovation,casual,"Disruptive ideas, Growth hacks"
  ```
- **Content Directory**: `marketing_content/`
  - `Small Business Owners.txt`
  - `Tech Startups.txt`

### Command:
```bash
python marketing_content_generation_agent.py segments.csv marketing_content/ output_snippets.json
```

### Output:
- **JSON File**: `output_snippets.json`
  ```json
  [
      {
          "segment_name": "Small Business Owners",
          "content_type": "email subject line",
          "personalized_snippet": "Hire Local Talent Faster & Cheaper: See How"
      },
      {
          "segment_name": "Tech Startups",
          "content_type": "email subject line",
          "personalized_snippet": "Level Up Your Startup's Talent Game 🚀"
      }
  ]
  ```

---

## Troubleshooting
1. **Missing API Key**: Ensure the `GOOGLE_API_KEY` environment variable is set.
2. **File Not Found**: Verify the paths to the CSV file and content directory.
3. **Dependency Issues**: Reinstall dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

---

## Future Enhancements
- Add support for more content types.
- Improve keyword extraction using advanced NLP libraries.
- Implement parallel processing for large datasets.

---

Feel free to customize this `README.md` further based on your specific needs!