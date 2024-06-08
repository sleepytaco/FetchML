# Fetch ML Apprentice Take Home

## Project Overview

The goal is first to build a sentence transformer that produces sentence embeddings. I utilized a pre-trained BERT model to build this sentence transformer. I chose BERT due to it being a general-purpose model for most NLP tasks. Then, this sentence transformer is extended to support multitask learning to do sentence classification and sentiment analysis.

I used the `transformers` library from HuggingFace to build the transformer backbone to produce sentence embeddings based on a pre-trained BERT model. I use `poetry` to manage my Python dependencies. I dockerized this project and have provided instructions below to run my code locally.

### Task 1: Sentence Transformer Implementation

The `model/sentence_transformer.py` file contains the `SentenceTransformer` class which is a wrapper over a pre-trained BERT model. The `forward` method of the `SentenceTransformer` takes in tokenized sentences and produces sentence embeddings by taking the mean of the token embeddings from the last layer of the pre-trained BERT model.

As the `SentenceTransformer` model is based on the HuggingFace's `transformers` library, it expects sentence inputs in the form of `input_ids` and `attention_mask`. I use the `BertTokenizer` to tokenize the input sentences into a form that is expected by the HuggingFace transformer.

With such an approach it is possible to get fixed-sized embeddings of dimension 768. Here, 768 is the hidden dim of the last hidden state of the pre-trained BERT model. 

It is possible to get custom-sized sentence embedding by attaching a fully connected layer at the end of the transformer backbone. Its purpose would be to project embeddings from 768 to any size of our choice. However, there is an additional cost incurred as some fine-tuning on downstream tasks (e.g. classification) is required for this newly added fully-connected layer to produce good embeddings. As such, I did not make this design choice.

There is no model training required for this task as we rely on a pre-trained BERT model to help us produce sentence embeddings.

### Task 2: Multi-Task Learning Expansion

The `model/sentence_transformer.py` file also contains a `MultiTaskModel` class which extends the `SentenceTransformer` class to support multitask learning. Specifically, I added two fully-connected layer heads: one to do sentence classification and another to do sentiment analysis. I made each head contain two fully connected layers to allow for some flexibility during learning.

Training this multitask model involves first combining the losses from the two task heads and then optimizing it the usual way using backpropagation. This allows the model's heads to learn weights such that they perform well on their specific tasks. This multitask model will be trained on a dataset of sentences, classification labels, and sentiment labels.

### Task 3: Training Considerations

Different scenarios for training the `MultiTaskModel` with a pre-trained BERT backbone and two task heads:
- **If the entire network should be frozen.** In this case, no training is required. None of the weights of the model are updated. The implication of this is that the task heads will be randomly initialized initially and will not be updated at all. Hence, the overall network will not be producing accurate outputs.
- **If only the transformer backbone should be frozen.** In this case, training involves updating the weights of both the task heads and keeping the weights of the transformer backbone fixed. The task-specific heads can be fine-tuned on a dataset of sentences, classification labels, and sentiment labels. With such a training scheme, the task heads will learn the appropriate weights to perform well on their individual tasks while leveraging the features of the pre-trained transformer backbone.
- **If only one of the task-specific heads (either for Task A or Task B) should be frozen.**  In this case, training involves updating the weights of the transformer model and one of the task-specific heads. With training the model using this approach, you can risk losing the beneficial features that were learned by the earlier transformer layers.

Transfer-learning based scenarios:
- **The choice of a pre-trained model.** I used a pre-trained BERT model as a starting point to produce sentence embeddings. BERT is well known to be a general-purpose model for NLP tasks. There are other pre-trained models (like RoBERTa, GPT, etc.) that might be well suited for different tasks. You want to choose a pre-trained model that is as close as possible to the task you are trying to achieve. This way you can take advantage of the features it learned which can produce sentence embeddings more suitable to your specific task. 
- **The layers you would freeze/unfreeze.** The earlier layers of the neural network typically contain low-level features of sentences and the later layers build on these low-level features. To take advantage of the features learned by a pre-trained model, I would freeze its earlier layers and fine-tune the later layers to my specific task.
- **The rationale behind these choices.** To do transfer learning, it is wise to pick a pre-trained model close to your task domain and find a good balance between the number of layers to keep frozen and unfrozen. This way you can leverage the beneficial features of the pre-trained model from the earlier layers and fine-tune the rest of the layers to perform your specific task.

## Running the Code

I have dockerized my code. Please ensure to have Docker installed and open before running the commands below.

Run the following command in your terminal to build the docker container:

```shell
docker build -t fetch-ml .
```

Once the container is built, you can specify which task to run as an argument to the `docker run` command. For example, run the following to run task 1:

```shell
docker run --rm fetch-ml --task=1 --sentence="Hello world!"
```

The `--sentence` argument is optional. If none is specified, the sentences from the `data/sentences.txt` file are passed as input to the transformer model.


