from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset


VALID_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14]


def load_and_clean_dataset(dataset_name):
    # Loads data, only keeps english, removes language column
    return (
        load_dataset(dataset_name)
        .filter(lambda x: x["lang"] == "en")
        .remove_columns("lang")
    )


# Taken from HF, to align the ner labels with tokenized words
def align_labels_with_tokens(labels, word_ids, max_length):
    new_labels = []
    current_word = None

    # Word ids are mappings between original words and tokens, ie [hej jag heter] => [hej jag het #er] => [0,1,2,2]
    for word_id in word_ids:
        if word_id != current_word:
            # New word
            current_word = word_id
            label = (
                -100 if word_id is None else labels[word_id]
            )  # -100 to not take into account during loss function
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)  # -100 for special tokens
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels[:max_length]  # Truncates labels to match max length


def to_tensor_dataset(data, labels):
    # Effectively zips the data and labels
    inp_ids = data["input_ids"]
    atmask = data["attention_mask"]
    return TensorDataset(inp_ids, atmask, labels)


def label_fix(labels):
    # Input is a nested list of labels
    return [[x if x in VALID_LABELS else 0 for x in sublist] for sublist in labels]


def prepare_data(dataset, tokenizer, splitname, system, max_length=70):
    # Dataset, split, all_labels or subset => Returns torch tensors

    sents = dataset[splitname]["tokens"]
    labels = dataset[splitname]["ner_tags"]

    # Change labels if B
    # A bit slow - would be faster with tensor operations
    if system == "B":
        labels = label_fix(labels)

    assert len(sents) == len(
        labels
    ), "The number of sentences and labels dont match!"  # Sanity check

    # Tokenize and align labels, currently pad everything to the same length
    tokenized_sents = tokenizer(
        sents,
        is_split_into_words=True,  # Input came as lists
        add_special_tokens=True,  # CLS and SEP
        padding="max_length",  # Pad shorter sequences
        truncation=True,  # Truncates longer sequences
        max_length=max_length,  # Set to 70 by default
        return_tensors="pt",
    )  # Pytorch tensors

    # Stack to turn list of torch tensors into one tensor
    aligned_labels = torch.stack(
        [
            torch.tensor(
                align_labels_with_tokens(
                    labels[i], tokenized_sents[i].word_ids, max_length
                )
            )
            for i in range(len(labels))
        ]
    )

    dataset = to_tensor_dataset(tokenized_sents, aligned_labels)

    return dataset


def ret_mapping():
    # Mapping from integer labels to strings (from HF dataset repo) and vice versa
    stoi = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    itos = {value: key for key, value in stoi.items()}
    return stoi, itos
