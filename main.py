import preprocess
import training
import evaluation
import argparse
from transformers import AutoTokenizer


def main():
    
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["A", "B"], required=True)
    args = parser.parse_args()


    # Dataset and mappings
    dataset_eng = preprocess.load_and_clean_dataset("Babelscape/multinerd")
    stoi, itos = preprocess.ret_mapping()


    # A or B system
    if args.system == "A":
        num_labels = len(itos)
    else:
        num_labels = 11


    # Init model_name and tokenizer
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Datasets
    train_data = preprocess.prepare_data(dataset_eng, tokenizer, "train", args.system)
    val_data = preprocess.prepare_data(dataset_eng, tokenizer, "validation", args.system)
    test_data = preprocess.prepare_data(dataset_eng, tokenizer, "test", args.system)


    # Training
    device, gpu_per_trial, cpu_per_trial = training.device_check()
    model = training.load_model(model_name, num_labels, tokenizer, device)
    best_config = training.random_hyperparameter_search(model, trai_data, val_data, gpu_per_trial, cpu_per_trial, num_trials=10)
    trained_model, e_loss, t_loss = training.train_model(best_config, model, train_data, val_data, hyperparameter_search=False)


    # Evaluation
    preds, labels = evaluation.test(model, test_data)




if __name__=="__main__":
    main()