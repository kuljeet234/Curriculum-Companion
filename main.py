import PyPDF2
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_number in range(4, 8):  # edit page numbers accordingly
            page = reader.pages[page_number]
            text += page.extract_text()
    return text


def summarize_text(text):
    model_name = "t5-small"  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def search_youtube_playlist(keywords):
    youtube_search_url = f"https://www.youtube.com/results?search_query={'+'.join(keywords.split())}+playlist"
    return youtube_search_url


pdf_path = 'coursehandouts/AI2204_course handout_Automata.pdf' # add path
start_page = 4
end_page = 8


pdf_text = extract_text_from_pdf(pdf_path, start_page, end_page)


#summarized_text = summarize_text(pdf_text)   to get summarised text


playlist_url = search_youtube_playlist(summarized_text)

print("Summarized Text:", summarized_text)
print("YouTube Playlist URL:", playlist_url)
