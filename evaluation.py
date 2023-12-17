
import evaluate


def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _ids, at, lab = batch
            out = model(
                input_ids=_ids.to(model.device), attention_mask=at.to(model.device)
            )

            # print(out.logits.shape)  #dim 2 are predictions for each input
            preds = torch.argmax(out.logits, dim=2)

            # Only compare indicies where there is not padding or special tokens
            mask = lab != -100 

            # Retrieve correct indices
            preds = preds[mask]
            labs = lab[mask]

            # Tolist for comparison
            predictions.extend(preds.tolist())
            labels.extend(labs.tolist())

    return predictions, labels


def dfs_save(evals, system):
    output_dir = f"./system_{system}_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Separate DFs for entities and overall metrics, rounded to 3 decimals
    entity_df = pd.DataFrame({k:v for k,v in evals.items() if not k.startswith("overall")}).T.round(3)
    overall_df = pd.DataFrame({k:v for k,v in evals.items() if k.startswith("overall")}, index=[0]).round(3)

    entity_df.to_csv(os.path.join(output_dir, "entity_metrics.csv"))
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"))


def compute_metrics(preds, labels, system):
    stoi, itos = ret_mapping()

    seqeval = evaluate.load('seqeval')
    evals = seqeval.compute(
                    predictions=[[itos[x] for x in sublist] for sublist in preds], 
                    references=[[itos[x] for x in sublist] for sublist in labels],
                    mode="strict"
    )

    print(evals)

    dfs_save(evals, system)




    

