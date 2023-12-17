import preprocess
import training
import evaluation
import argparse
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-cased"
DATASET_NAME = "Babelscape/multinerd"


def main():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["A", "B"], required=True)
    parser.add_argument("--search", choices=["Yes", "No"], required=True)
    args = parser.parse_args()

    # Dataset and mappings
    dataset_eng = preprocess.load_and_clean_dataset(DATASET_NAME)

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Preprocess data
    train_data = preprocess.prepare_data(dataset_eng, tokenizer, "train", args.system)
    val_data = preprocess.prepare_data(
        dataset_eng, tokenizer, "validation", args.system
    )
    test_data = preprocess.prepare_data(dataset_eng, tokenizer, "test", args.system)

    # Hyperparameter search & Training
    if args.search == "Yes":
        best_hyperparameters = training.hyperparameter_search(MODEL_NAME, train_data, val_data, tokenizer) # Runs for 3000 steps
    else:
        best_configs =  {"A": {'lr': 2e-05, 'batch_size': 16, 'warmup_steps': 0, 'epochs': 1},
                        "B": {'lr': 3e-05, 'batch_size': 32, 'warmup_steps': 0, 'epochs': 1}}
        best_hyperparameters = best_configs[args.system]
    
    model = training.init_model(MODEL_NAME, tokenizer, training_data)
    model, t_loss, e_loss = train_model(best_hyperparameters, train_data, val_data, hyperparameter_search=False, args.system) # Trains a model and saves it

    # Evaluation
    preds, labels = evaluation.test(model, test_data, args.system)


if __name__ == "__main__":
    main()
