import os
from openai import OpenAI
import json
import re
from predict import predict_age
from config import load_config, Config

def check_file(path):
    return (
        os.path.exists(path)
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def extract_json_data(mixed_string: str | None) -> dict:
    if not mixed_string:
        return {}

    mixed_string = re.sub(r"^```json\s*", "", mixed_string.strip())
    mixed_string = re.sub(r"\s*```$", "", mixed_string)

    try:
        return json.loads(mixed_string)
    except json.JSONDecodeError:
        return {}


def create_client(conf: Config) -> OpenAI:
    return OpenAI(
        api_key=conf.api_key,
        base_url=conf.api_base
    )


def main():

    print("|----------------Message Input----------------|")
    user_input = input("> ")
    print("> ")

    if user_input != "":
        pred_results_dict = predict_age(user_input)
        pred_results_str = json.dumps(
            pred_results_dict,
            ensure_ascii=False,
            indent=2
        )

        for k, v in pred_results_dict.items():
            if k != "text":
                print(f"> {k}: {v}")

    else:
        print("Empty input. Cannot work without the user text")
        print("Terminating.....")
        return

    conf = load_config()

    # If LLM is connected
    if conf.llm_enabled():
        client = create_client(conf)

        system_prompt = ""
        messages = []

        if check_file(conf.system_prompt_path):
            print("Loading system prompt...")
            with open(conf.system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()

        else:
            print(f"Path '{conf.system_prompt_path}' does not exist or is empty")

        if system_prompt != "":
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        if pred_results_str != "":
            messages.append(
                {"role": "user", "content": pred_results_str}
            )

        try:
            response = client.chat.completions.create(
                model=conf.llm,
                messages=messages,
            )
            llm_output_str = response.choices[0].message.content
            llm_output_dict = extract_json_data(llm_output_str)

            print("|----------------Reasoning----------------|")
            print(f"> {llm_output_dict.get(
                        "reasoning",
                        "Reasoning unavailable"
                        )}")

        except Exception as e:
            print(type(e))
            print(e)

    else:
        print("Reasoning unavailable. No LLM configuration set")


if __name__ == "__main__":
    main()