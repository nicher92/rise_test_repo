
import evaluate
from torch.utils.data import DataLoader
import torch
import pandas as pd
import preprocess
import os


def mapper(predictions, inverse_label_mapping):
    # In system B we went integer => integer mapping, we need to reverse this before integer to string to map to the original dict in order get correct classes
    return [[inverse_label_mapping.get(label_id, 0) for label_id in pred_list] for pred_list in predictions]


def dfs_save(evals, system):
    # Saves two dataframes, one for overall metrics and one for class-wise metrics

    output_dir = f"./system_{system}_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Separate DFs for entities and overall metrics, rounded to 3 decimals

    entity_df = pd.DataFrame({k:v for k,v in evals.items() if not k.startswith("overall")}).T.round(3)
    overall_df = pd.DataFrame({k:v for k,v in evals.items() if k.startswith("overall")}, index=[0]).round(3)

    entity_df.to_csv(os.path.join(output_dir, "entity_metrics.csv"))
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"))


def compute_metrics(preds, labels, system):
    # Computes metrics using seqeval

    stoi, itos, LABEL_MAPPING, INVERSE_LABEL_MAPPING = preprocess.ret_mapping()

    seqeval = evaluate.load('seqeval')

    if system == "B":
        preds = mapper(preds, INVERSE_LABEL_MAPPING)
        labels = mapper(labels, INVERSE_LABEL_MAPPING)



    evals = seqeval.compute(
                        predictions=[[itos[x] for x in sublist] for sublist in preds], 
                        references=[[itos[x] for x in sublist] for sublist in labels],
                        mode="strict"
        )


    for k,v in evals.items():
        print(k, v)
        print("------------------")

    dfs_save(evals, system)


def test(model, test_dataset, system):
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _ids, at, lab = batch
            
            # Batch, padded tokenized sentence, classes => B * 70 * 31/11
            out = model(input_ids=_ids.to(model.device), attention_mask=at.to(model.device))
            batch_preds = torch.argmax(out.logits, dim=2)
            
            
            for b in range(_ids.shape[0]):
                mask = lab[b] != -100

                prediction = batch_preds[b][mask].tolist()
                label = lab[b][mask].tolist()

                # Tolist for comparison
                predictions.append(prediction)
                labels.append(label)
    


    compute_metrics(predictions, labels, system)






    

