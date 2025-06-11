from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Set the model path
model_path = "C:/Users/Zeke/Pretrained models/eng-je-llm"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to separate Katakana and casual Japanese
def separate_katakana_and_casual(text):
    katakana = "".join(re.findall(r'[ァ-ンーｧ-ﾝﾞﾟ]+', text))
    casual = re.sub(r'[ァ-ンーｧ-ﾝﾞﾟ]+', '', text)
    casual = casual.lstrip("、。・「」『』？！：； 」」」　 ")  # Strip Japanese punctuation and full-width spaces
    casual = casual.strip()
    return katakana, casual

print("🧠 English to Japanese Translator (type 'exit' to quit)\n")

# Main loop
while True:
    raw_text = input("Enter text (or 'exit'): ").strip()
    if raw_text.lower() == "exit":
        break
    if not raw_text:
        continue

    print(f"🗣️ You said: {raw_text}")

    # No instruction — just raw input
    text = raw_text

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output
    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id.get("ja_XX", None)
    )

    # Decode the output
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Remove "日本語" prefix if present
    if translated_text.startswith("日本語"):
        translated_text = re.sub(r'^日本語[\s、:：]*', '', translated_text).strip()

    # Separate Katakana and Casual
    katakana, casual = separate_katakana_and_casual(translated_text)

    # Inject Katakana into casual if it's missing
    if katakana and katakana not in casual:
        casual = f"{katakana}、{casual}"

    # Suppress katakana line if it's already inside casual
    if katakana and katakana in casual:
        katakana = ""

    # Display output
    print("🧠 Model says:")
    if katakana:
        print(f"Katakana: {katakana}")
        print(f"Casual  : {casual}\n")
    else:
        print(f"{casual}\n")
