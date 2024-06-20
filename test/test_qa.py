from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def answer_question(question, context):

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # print(answer_start_scores)
   #  print(answer_end_scores)

    start_probs = F.softmax(answer_start_scores, dim=-1)
    end_probs = F.softmax(answer_end_scores, dim=-1)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    start_prob = start_probs[0, answer_start].item()
    end_prob = end_probs[0, answer_end - 1].item()
    certainty_score = start_prob * end_prob

    print(certainty_score)

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer


def main():
    print('Hello World')

    context = "Jane Doe, who works for IBM, married John Smith, who works for Google, on March 13th, 2010."
    question = "When did Jane Doe get married?"

    context = "The net loss of Microsoft in the second quarter of 2022 was $1 billion dollars."

    question = "Who has the loss?"

    answer = answer_question(question, context)

    print("Answer: " + answer)

    question = "When was the loss?"

    answer = answer_question(question, context)

    print("Answer: " + answer)

    question = "How much was the loss?"

    answer = answer_question(question, context)

    print("Answer: " + answer)

    question = "How much was the profit?"

    answer = answer_question(question, context)

    print("Answer: " + answer)


if __name__ == "__main__":
    main()
