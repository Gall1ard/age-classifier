from transformers import pipeline

id2label = {
    "LABEL_0": "13-17",
    "LABEL_1": "18-29",
    "LABEL_2": "30-48"
}


def predict_age(text: str) -> dict:
    classifier = pipeline(
        "text-classification",
        model="./models/age_classifier"
    )

    result = classifier(text)[0]

    return {
        "age_group": id2label[result["label"]],
        "confidence": f"{int(result["score"] * 100)}%",
        "text": text
    }