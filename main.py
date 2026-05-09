import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from youtube_search import YoutubeSearch
from PyPDF2 import PdfReader


# Phi-3-mini-4k-instruct (3.8B, 4k context, ungated) replaces the
# original orca_mini_3b (Llama-1-based, 2023). It uses standard
# Auto* classes and a chat template, so swapping to another current
# model — Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, etc. — is a
# one-line override via --model.
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
TOPIC_PREFIXES = tuple(f"{i}." for i in range(1, 11))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device, dtype = pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    return tokenizer, model, device


def generate_text(tokenizer, model, system, instruction, max_new_tokens=512):
    """Run an instruction through the tokenizer's chat template.

    Modern instruct models ship a `chat_template` that knows how to
    wrap system/user messages correctly — using it directly avoids
    the orca-specific `### System:` prompt format the old script
    relied on. Decoding is greedy (do_sample=False) because the prompt
    asks for a fixed-format numbered list — sampling produced variable
    topic counts run-to-run.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(inputs)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


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

    course_info = extract_text_from_pdf(args.pdf_path, args.start_page, args.end_page).strip()
    if not course_info:
        sys.exit(
            f"\nNo text was extracted from {args.pdf_path} between pages "
            f"{args.start_page} and {args.end_page}. Adjust --start-page / "
            "--end-page or pass a PDF that actually contains selectable text."
        )

    tokenizer, model, device = load_model(args.model)
    print(f"Model {args.model} loaded on {device}")

    system = (
        "You are an education expert. Given a course handout, extract the "
        "10 most important topics. Output exactly 10 lines, each starting "
        "with a number and a dot (e.g. '1. Topic name'). Output nothing else."
    )
    instruction = (
        "Extract the 10 most important topics from this course handout:\n\n"
        + course_info
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
