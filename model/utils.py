def load_sentences(file_path):
    """
    Reads a text file and returns a list of sentence strings.
    """
    with open(file_path, "r") as file:
        sentences = file.readlines()

    for i in range(len(sentences)):
        sentence = sentences[i].strip()  # Strip whitespace characters
        if sentence:  # Add check to ignore empty lines
            sentences[i] = sentence

    return sentences


def tokenize_sentences(sentences, tokenizer, max_length=512):
    """
    Tokenizes sentences with the passed in tokenizer.
    Returns input_ids, attention_mask as required by the HuggingFace transformers library.
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    return input_ids, attention_mask
