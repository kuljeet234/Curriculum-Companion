import argparse
import torch
from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer
from youtube_search import YoutubeSearch
from PyPDF2 import PdfReader


DEFAULT_MODEL = "psmathur/orca_mini_3b"
TOPIC_PREFIXES = tuple(f"{i}." for i in range(1, 11))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    device, dtype = pick_device()
    Accelerator()  # initialise; harmless on CPU
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device)
    return tokenizer, model, device


def generate_text(tokenizer, model, system, instruction, input_text=None, max_new_tokens=1024):
    if input_text:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=tokens,
            max_length=tokens.shape[1] + max_new_tokens,
            use_cache=True,
            do_sample=True,
            top_p=1.0,
            temperature=0.7,
            top_k=50,
        )
    return tokenizer.decode(out[0][tokens.shape[1]:], skip_special_tokens=True)


def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        total = len(reader.pages)
        last = end_page if end_page >= 0 else total + end_page
        last = max(0, min(last, total - 1))
        first = max(0, min(start_page, total - 1))
        for i in range(first, last + 1):
            text += reader.pages[i].extract_text() or ""
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Extract topics from a course PDF and find a YouTube video for each."
    )
    parser.add_argument("pdf_path", help="Path to the course handout PDF")
    parser.add_argument("--start-page", type=int, default=2)
    parser.add_argument("--end-page", type=int, default=-2,
                        help="Negative values count from the end (e.g. -2 = second-to-last)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    print(f"Model loaded on {device}")

    course_info = extract_text_from_pdf(args.pdf_path, args.start_page, args.end_page)

    system = "You are an AI expert in the field of education that follows instruction extremely well. Help as much as you can."
    instruction = (
        "i am giving you is a course handout and i want you to pick 10 words and these 10 words "
        "should be the most important topics from this course handout these 10 words should cover "
        "all the topics and these 10 words should be completely covering the topic you are only "
        "supposed to give the 10 words nothing else  you should also avoid putting anything else "
        "by yourself afterwards search these words on youtube and give me the links- \n" + course_info
    )

    raw = generate_text(tokenizer, model, system, instruction)
    topics = [line.strip() for line in raw.split("\n") if line.strip().startswith(TOPIC_PREFIXES)]
    if not topics:
        print("Model did not return numbered topics. Raw output:\n", raw)
        return

    for topic in topics:
        print(topic)

    for topic in topics:
        results = YoutubeSearch(topic, max_results=1).to_dict()
        if results:
            video_title = results[0]["title"]
            video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
            print(f"Topic: {topic}\nVideo Title: {video_title}\nVideo URL: {video_url}\n")
        else:
            print(f"No results found for topic: {topic}\n")


if __name__ == "__main__":
    main()
