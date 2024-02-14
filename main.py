import PyPDF2
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_number in range(4, 8):  # Adjusted to match page numbers starting from 1
            page = reader.pages[page_number]
            text += page.extract_text()
    return text

# Function to summarize text using a pre-trained model
def summarize_text(text):
    model_name = "t5-small"  # You can choose other models like 't5-base' for better performance
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to search for playlists on YouTube based on the summarized syllabus
def search_youtube_playlist(keywords):
    youtube_search_url = f"https://www.youtube.com/results?search_query={'+'.join(keywords.split())}+playlist"
    return youtube_search_url

# Example usage
pdf_path = 'coursehandouts/AI2204_course handout_Automata.pdf'
start_page = 4
end_page = 8

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path, start_page, end_page)

# Summarize the syllabus into a single line
summarized_text = summarize_text(pdf_text)

# Search for playlists related to the summarized syllabus on YouTube
playlist_url = search_youtube_playlist(summarized_text)

print("Summarized Text:", summarized_text)
print("YouTube Playlist URL:", playlist_url)
