from torch.utils.data import DataLoader
import torch
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
    get_scheduler,
    BertForTokenClassification,
)
from preprocess import load_dataset, prepare_data
import os


def ret_dataloaders(training_dataset, evaluation_dataset, batch_size, shuffle=False):
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)

    eval_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, eval_loader


def train_epoch(train_loader, scheduler, optimizer, model, hyperparameter_search, i, max_batches=None):
    model.train()
    train_loss = 0
    t_lossi = []
    
    for j,batch in enumerate(train_loader):
        _ids, at, lab = batch 
        out = model(input_ids=_ids.to(model.device), 
                    attention_mask=at.to(model.device), 
                    labels=lab.to(model.device)) #   logits = [256, 40, 31] B,T,C, loss is NLL
        train_loss += out.loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                       max_norm=1) #gradient clipping - safety net

        optimizer.zero_grad() # Zero gradients between each update
        out.loss.backward()   # Calculate gradients
        optimizer.step()      # Step

        if scheduler:         # Update learning rate
            scheduler.step()

        if j % 25 == 0 and j > 0:
            print("Average T loss at step {}: {}".format(j, train_loss / j ))
            t_lossi.append(train_loss / j)
            
        
        if max_batches is not None and j >= max_batches:
            break
            
    
    return model, t_lossi


def eval_epoch(eval_loader, model):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for j,batch in enumerate(eval_loader):
            
            _ids, at, lab = batch
            out = model(input_ids=_ids.to(model.device), 
                        attention_mask=at.to(model.device), 
                        labels=lab.to(model.device))
            eval_loss += out.loss.item()
            
    avg_eval_loss = eval_loss / len(eval_loader)
    return avg_eval_loss


def save_model(model, tokenizer, system):
    output_dir = f"./model_{system}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_model(params, model, training_dataset, evaluation_dataset, hyperparameter_search=True, system="A"):
    
    train_loader, eval_loader = ret_dataloaders(training_dataset, evaluation_dataset, params["batch_size"])
    optimizer = AdamW(params = model.parameters(), lr=params["lr"])
    
    total_steps = (len(training_dataset) // params["batch_size"]) * params["epochs"]
    warmup_steps = int(total_steps * params["warmup_steps"]) #standard 10th
    scheduler = get_scheduler("linear", 
                              optimizer=optimizer, 
                              num_warmup_steps= warmup_steps, ##set warmup steps to 0.1 * total num steps 
                              num_training_steps=total_steps)
    
    
    # Early stopping for hyperparameter search
    if hyperparameter_search:
        max_batches = 3000
    else:
        max_batches = None
    
    for i in range(1, params["epochs"] + 1):
        model, t_loss = train_epoch(train_loader, scheduler, optimizer, model, hyperparameter_search, i, max_batches)
        e_loss = eval_epoch(eval_loader, model)
        print("epoch: ", i)
        
    
    if hyperparameter_search:
        return t_loss, e_loss
    else:
        save_model(model, tokenizer, system)
        return model, t_loss, e_loss


def device_check():
    # Simple GPU check -
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device



def hyperparameter_combinations():
    # Currently checks all combinations
    
    learning_rates = [2e-5, 3e-5, 5e-5]
    batch_sizes = [16, 32]
    warmup_steps = [0, 0.1]


    hyperparameter_configs = []
    for lr in learning_rates:
        for bs in batch_sizes:
            for ws in warmup_steps:
                params = {
                "lr": lr,            # How much should the model adjust its parameters to fit the task per update
                "batch_size": bs,    # How large of a batch should the model use to guide its update
                "warmup_steps": ws,  # Should the LR "climb" up to the max value before decay or start at the top instantly
                "epochs" : 1         # Set to 1 since I use max batches in hyperparameter search anyways
            }
                hyperparameter_configs.append(params)
        
    return hyperparameter_configs



def hyperparameter_search(model_name, training_dataset, evaluation_dataset, tokenizer):
    
    device = device_check()
    hyperparameter_configs = hyperparameter_combinations()
    
    results = []
    for i,config in enumerate(hyperparameter_configs):
        model = init_model(model_name, tokenizer, training_dataset)
        t_loss, e_loss = train_model(config, model, training_dataset, evaluation_dataset, hyperparameter_search=True)
        results.append((config, t_loss, e_loss, i))
        
    
    sorted_by_eval = sorted(results, key=lambda x: x[2])
    best_hyperparameters = sorted_by_eval[0][0]   # Get top config of results sorted by average evaluation loss
    
    return best_hyperparameters, results


def init_model(model_name, tokenizer, training_dataset):
    
    device = device_check()
    num_labels = len(torch.unique(torch.cat([x[2] for x in training_dataset]))) - 1
    
    # The model probably doesnt need to be more complicated than this, the context is usually very short
    model = BertForTokenClassification.from_pretrained(model_name, 
                                                   num_labels=num_labels, 
                                                   vocab_size=tokenizer.vocab_size, 
                                                   ignore_mismatched_sizes=True).to(device)

    
    return model

