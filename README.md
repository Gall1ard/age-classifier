# Age Prediction from Text using BERT

A machine learning project that predicts the author's age group from English text using a fine-tuned BERT model and lets a chosen external LLM to explain how the choice was made.

## Overview

This project fine-tunes BERT on a dataset of English texts labeled by age group.

The model classifies a text into one of the following categories:

* 13–17
* 18–29
* 30–48

The goal is to investigate whether writing style, vocabulary, and discussed topics can be used to estimate the author's age.

## Dataset

Dataset fields:

| Column    | Description                                    |
| --------- | ---------------------------------------------- |
| text      | Original text                                  |
| age_group | Target age category                            |

Class distribution:

| Age group | Samples |
| --------- | ------: |
| 13–17     |  17,471 |
| 18–29     |  55,377 |
| 30–48     |  23,351 |

## Model

Base model:

* bert-base-uncased

Training setup:

* Epochs: 3
* Batch size: 16
* Learning rate: 2e-5
* Max sequence length: 128

## Results

Test metrics:

| Metric   | Score |
| -------- | ----: |
| Accuracy |  0.74 |
| Macro F1 |  0.70 |

Classification report:

| Age group | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| 13–17     |      0.80 |   0.68 | 0.74 |
| 18–29     |      0.75 |   0.84 | 0.79 |
| 30–48     |      0.65 |   0.54 | 0.59 |

Confusion matrix:

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/007afe0a-97cf-4a64-b743-022ccb779443" />

## Project Structure

```text
project/
│
├── data/
│   └── prepared.csv
│
├── models/
│   └── age_classifier/
│
├── prompts/
│   └── system_prompt.txt
│
├── src/
│   ├── evaluate.py
│   ├── fine_tuning.py
│   ├── main.py
│   ├── predict.py
│   └── prepare_data.py
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd project

pip install -r requirements.txt
```

## Training

```bash
python src/fine_tuning.py
```

## Prediction

```bash
python src/main.py
```

## Optional LLM Reasoning

The project supports optional LLM-powered explanations for age predictions.

When enabled, an external LLM analyzes the input text and explains why the classifier assigned a particular age group.

### Configuration

Create a `.env` file in the project root:

```env
CHUTES_API=your_api_key
CHUTES_BASE_URL=https://your-chutes-endpoint
CHUTES_LLM=chosen_model_name (example: google/gemma-4-31B-turbo-TEE)
```

### Environment Variables

| Variable          | Description                         |
| ----------------- | ----------------------------------- |
| `CHUTES_API`      | Chutes API key                      |
| `CHUTES_BASE_URL` | Chutes API base URL                 |
| `CHUTES_LLM`      | Model identifier used for reasoning |

### Example

Input:

```text
I am preparing for my university exams and applying for internships.
```

Prediction:

```text
18-29
```

LLM Explanation:

```text
The text references university studies and internships, which are commonly associated with young adults. The writing style and discussed topics are more typical of the 18–29 age group than teenagers or older adults.
```

## Future Improvements

* Compare BERT and RoBERTa
* Use industry as an additional feature
* Experiment with larger context lengths
* Deploy as a FastAPI service
* Create a web interface

```
```
