



def metrics(preds, labels):
    f1 = f1_score(preds, labels)          #
    acc = accuracy_score(preds, labels)   # 
    rec = recall_score(preds, labels)     # % of entites found
    prec = precision_score(preds, labels) # % of entities found that are correct
    
    return f1, acc, rec, prec


def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _ids, at, lab = batch
            out = model(input_ids=_ids.to(model.device), attention_mask=at.to(model.device)) 

            #print(out.logits.shape)  #dim 2 are predictions for each input
            preds = torch.argmax(out.logits, dim=2)


            # Only compare indicies where there is not padding or special tokens
            mask = (lab != -100) #& (lab != 0)

            # Also test with mask = lab != 0, since the class is so large

            # Retrieve correct indices
            preds = preds[mask]
            labs = lab[mask]

            # Tolist for comparison
            predictions.extend(preds.tolist())
            labels.extend(labs.tolist())
    
    return predictions, labels


#accuracy is high, because most are just 0
#sum([1 for x,y in zip(predictions, labels) if x==y]) / len(labels)



