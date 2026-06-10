import httpx
import os
from openai import OpenAI
from dotenv import load_dotenv
from chutes_e2ee import ChutesE2EETransport
import json
import re
from predict import predict_age

load_dotenv()

API_KEY = os.getenv("CHUTES_API_KEY")
API_BASE = os.getenv("CHUTES_API_BASE")
LLM = os.getenv("CHUTES_LLM")

# (Unused)
# MODEL_SLUG = "chutes-google-gemma-3-31b-turbo-tee"
SYSTEM_PROMPT_PATH = "./prompts/system_prompt.txt"


def check_file(file_path):
    # Check if path exists
    if not os.path.exists(file_path):
        print(f"Path '{file_path}' does not exist")
        return False

    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        print(f"'{file_path}' is not a file")
        return False

    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        print(f"File '{file_path}' is empty")
        return False

    print(f"File '{file_path}' exists and is not empty")
    return True


def extract_json_data(mixed_string: str | None) -> dict:
    if not mixed_string:
        return {}

    mixed_string = re.sub(r"^```json\s*", "", mixed_string.strip())
    mixed_string = re.sub(r"\s*```$", "", mixed_string)

    return json.loads(mixed_string)


def main():
    if not API_KEY:
        raise ValueError("API error: API_KEY not set")
    if not API_BASE:
        raise ValueError("API error: API_BASE not set")

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )

    system_prompt = ""
    messages = []

    if check_file(SYSTEM_PROMPT_PATH):
        print("Loading system prompt...")
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    if system_prompt != "":
        messages.append(
            {"role": "system", "content": system_prompt}
        )

    print("|----------------Message Input----------------|")
    user_input = input("> ")
    print(">")

    if user_input != "":
        pred_results_dict = predict_age(user_input)
        pred_results_str = json.dumps(pred_results_dict)

        for k, v in pred_results_dict.items():
            if k != "text":
                print(f"> {k}: {v}")

        messages.append(
            {"role": "user", "content": pred_results_str}
        )

    else:
        print("Empty input. Cannot work without the user text")
        print("Terminating.....")
        return


    try:
        response = client.chat.completions.create(
            model=LLM,
            messages=messages,
        )
        llm_output_str = response.choices[0].message.content
        llm_output_dict = extract_json_data(llm_output_str)

        print("|----------------Reasoning----------------|")
        print(f"> {llm_output_dict['reasoning']}")

    except Exception as e:
        print(type(e))
        print(e)


if __name__ == "__main__":
    main()