import argparse
from transformers import BertTokenizer
from model.sentence_transformer import SentenceTransformer, MultiTaskModel
from model.utils import load_sentences, tokenize_sentences


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, help="Specify task # to run (1 or 2)")
    parser.add_argument("--sentence", type=str, help="Specify a sentence to run the model with (optional)")
    return parser


def main():
    args = make_parser().parse_args()

    # For task 2 - multitask model
    num_classes = 3
    num_sentiments = 2

    # Set up the tokenizer, load and tokenize the sentences dataset
    max_length = 128  # Max sentence length to the transformer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")  # Using BERT model based tokenizer
    sentences = args.sentence if args.sentence else load_sentences("data/sentences.txt")
    input_ids, attention_mask = tokenize_sentences(sentences, tokenizer, max_length=max_length)

    if args.task == 1:
        model = SentenceTransformer()

        # Forward pass input tokenized sentences to get embeddings
        embeddings = model(input_ids, attention_mask)

        print("")
        print("Number of sentences passed as input: ", len(sentences))
        print("Embeddings shape: ", embeddings.shape)
        print("Embeddings Output: ", embeddings)
        print("")

        for i in range(len(sentences)):
            print("Sentence:", sentences[i])
            print("Embedding (truncated to first 5 values):", embeddings[i, :5])
            print("")
    elif args.task == 2:
        model = MultiTaskModel(num_classes, num_sentiments)

        # Forward pass input tokenized sentences to get outputs of the two task heads
        classification_probs, sentiment_probs = model(input_ids, attention_mask)

        print("")
        print("Number of sentences passed as input: ", len(sentences))
        print(f"Classification Probabilities ({num_classes} classes): ", classification_probs)
        print(f"Sentiment Probabilities: ({num_sentiments} classes):", sentiment_probs)
        print("")

        for i in range(len(sentences)):
            print("Sentence:", sentences[i])
            print("Sentence Classification Probabilities:", classification_probs[i])
            print("Sentiment Analysis Probabilities:", sentiment_probs[i])
            print("")
    else:
        print(f"Unknown task specified: {args.task}")


if __name__ == "__main__":
    main()
