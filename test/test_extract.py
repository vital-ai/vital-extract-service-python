from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def main():
    print('Hello World')

    tokenizer = AutoTokenizer.from_pretrained("ibm/knowgl-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("ibm/knowgl-large")

    # sentence = "John Smith married Jane Doe, who works for IBM, on March 13th, 2010."

    sentence = "Jane Doe, who works for IBM, married John Smith, who works for Google, on March 13th, 2010."

    inputs = tokenizer(sentence, return_tensors="pt")

    # outputs = model(**inputs, labels=inputs["input_ids"])
    # output_sentence = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # loss = outputs.loss
    # perplexity = torch.exp(loss).item()
    # print(f"Perplexity: {perplexity}")

    # model.eval()

    inputs['labels'] = inputs['input_ids']

    output_tokens = model.generate(inputs["input_ids"])

    output_sentence = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # with torch.no_grad():
        # Perform a forward pass to get logits and loss for perplexity calculation
        # outputs = model(**inputs)

        # Extract logits (unnormalized scores) and loss
        # logits = outputs.logits
        # loss = outputs.loss

        # Generate tokens for output sequence
        # generated_tokens = model.generate(inputs["input_ids"])

    # Decode generated tokens to text
    # decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Calculate perplexity
    # perplexity = torch.exp(loss).item()

    # print(f"Generated Output Sequence: {decoded_output}")
    # print(f"Perplexity: {perplexity}")

    # [(subject mention # subject label # subject type) | relation label | (object mention # object label # object type)]

    print(f"Generated Output Sequence: {output_sentence}")

    relationships = output_sentence.split('$')

    def parse_part(part):
        part = part.strip('[] ')
        mention, label, type_ = part.split('#')
        return {'mention': mention.strip('('), 'label': label, 'type': type_.strip(')')}

    parsed_relationships = []
    for relationship in relationships:
        if relationship:
            subject, relation, object_ = relationship.split('|')
            parsed_relationship = {
                'subject': parse_part(subject),
                'relation': relation.strip(),
                'object': parse_part(object_),
            }
            parsed_relationships.append(parsed_relationship)

    for rel in parsed_relationships:
        print(rel)


if __name__ == "__main__":
    main()


