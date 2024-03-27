from accelerate import Accelerator
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from youtube_search import YoutubeSearch
from PyPDF2 import PdfReader

model_path = 'psmathur/orca_mini_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)

accelerator = Accelerator()
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model = model.to(torch.device("cuda"))

def generate_text(system, instruction, input=None):
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to(model.device)

    instance = {'input_ids': tokens, 'top_p': 1.0, 'temperature': 0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens,
            max_length=length + instance['generate_len'],
            use_cache=True,
            do_sample=True,
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'

def extract_text_from_pdf(pdf_path, start_page, end_page):
    pdf_text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(start_page, end_page + 1):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

pdf_path = '/AI2204_course handout_Automata.pdf'
start_page = 2
end_page = -2  

course_info = extract_text_from_pdf(pdf_path, start_page, end_page)

system = 'You are an AI expert in the field of education that follows instruction extremely well. Help as much as you can.'
instruction = '''
i am giving you is a course handout and i want you to pick 10 words and these 10 words should be the most important topics from this course handout these 10 words should cover all the topics and these 10 words should be completely covering the topic you are only supposed to give the 10 words nothing else  you should also avoid putting anything else by yourself afterwards search these words on youtube and give me the links- 
''' + course_info

generated_text = generate_text(system, instruction)  

lines = generated_text.split('\n')
important_topics = [line for line in lines if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.'))]

for topic in important_topics:
    print(topic)

for topic in important_topics:
    results = YoutubeSearch(topic, max_results=1).to_dict()
    if results:
        video_title = results[0]['title']
        video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
        print(f"Topic: {topic}\nVideo Title: {video_title}\nVideo URL: {video_url}\n")
    else:
        print(f"No results found for topic: {topic}\n")
