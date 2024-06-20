from frame_semantic_transformer import FrameSemanticTransformer


def main():
    print('Hello World')

    # sentence = "On November 17, 2023, the board removed Altman as CEO, while Brockman was removed as chairman and then resigned as president."

    # sentence = "The CEO of ABC Company hired Jane Doe for the position of Marketing Manager to handle the company's promotional activities."

    # sentence = "the book cover is 5 inches wide."

    # sentence = '"Hello" is a song by band Joy, released in November 1983 as the second single from its fifth album.'

    sentence = "The net loss of Microsoft in the second quarter of 2022 was $1 billion dollars."

    # sentence = "John works for OpenAI."
    # sentence = "He started in 2020."

    frame_transformer = FrameSemanticTransformer("base")

    frame_transformer.setup()

    result = frame_transformer.detect_frames(sentence)

    print(f"Results found in: {result.sentence}")

    for frame in result.frames:
        print(f"FRAME: {frame.name}")
        for element in frame.frame_elements:
            print(f"{element.name}: {element.text}")


if __name__ == "__main__":
    main()

# use LLM to take request and see what frames would be a match
# so we just need to index the output of frameparser?
