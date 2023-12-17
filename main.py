import preprocess
import training
import evaluation
import argparse
from transformers import AutoTokenizer, BertForTokenClassification


MODEL_NAME = "bert-base-cased"
DATASET_NAME = "Babelscape/multinerd"


def main():
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["A", "B"], required=True)
    parser.add_argument("--search", choices=["Yes", "No"], help="Perform hyperparameter search or not")
    parser.add_argument("--model_path", type=str, help="Path to already trained model")
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

    
    if not args.model_path:

        # Hyperparameter search & Training
        if args.search == "Yes":
            best_hyperparameters = training.hyperparameter_search(MODEL_NAME, train_data, val_data, tokenizer) # Runs for 3000 steps
        else:
            best_configs =  {"A": {'lr': 2e-05, 'batch_size': 16, 'warmup_steps': 0, 'epochs': 1},
                            "B": {'lr': 3e-05, 'batch_size': 32, 'warmup_steps': 0, 'epochs': 1}}
            best_hyperparameters = best_configs[args.system]

        # Initializes the model based on model name, tokenizer and labels present in the training data
        model = training.init_model(MODEL_NAME, tokenizer, train_data) 

        # Trains a model and saves it in /models/
        model, t_loss, e_loss = training.train_model(best_hyperparameters, model, train_data, val_data, hyperparameter_search=False, system=args.system) 
    
    else:
        try:
            print("Loading pre-trained model..")
            model = training.init_pretrained_model(args.model_path)
            print("Model loaded!")
        except Exception as e:
            print(e)

    # Evaluation, saves result as a .csv in ./system_A_results
    print("Evaluating results...")
    evaluation.test(model, test_data, args.system)
    print("Evaluation finished!")


if __name__ == "__main__":
    main()
