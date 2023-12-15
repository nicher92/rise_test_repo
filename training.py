from ray import tune, train
from torch.utils.data import DataLoader
import torch
from transformers import get_linear_schedule_with_warmup, AdamW, get_scheduler, BertForTokenClassification
from preprocess import load_dataset, prepare_data


def ret_dataloaders(training_dataset, evaluation_dataset, batch_size, shuffle=False):
    train_loader = DataLoader(training_dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle)
    
    eval_loader = DataLoader(evaluation_dataset,
                             batch_size=batch_size, 
                             shuffle=shuffle)
    
    return train_loader, eval_loader


def train_epoch(train_loader, scheduler, optimizer, model, hyperparameter_search, i):
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
            if hyperparameter_search:
                train.report({"step": j,
                              "epoch": i,
                              "training_loss": train_loss / j, 
                              "loss": None}) # Loss needs to be 0 here otherwise error from tune
            
            t_lossi.append(train_loss / j)
            
    return model, t_lossi
    


def eval_epoch(eval_loader, model, hyperparameter_search, i):
    model.eval()
    eval_loss = 0
    e_lossi = []
    with torch.no_grad():
        for j,batch in enumerate(eval_loader):
            
            _ids, at, lab = batch
            out = model(input_ids=_ids.to(model.device), 
                        attention_mask=at.to(model.device), 
                        labels=lab.to(model.device))
            eval_loss += out.loss.item()

            if j % 25 == 0 and j > 0:
                print("Average E loss at step {}: {}".format(j, eval_loss / j ))
                if hyperparameter_search:
                    train.report({"step": j, 
                                  "epoch": i, 
                                  "evaluation_loss": eval_loss / j, 
                                  "loss": None }) # Loss needs to be 0 here otherwise error from tune
                
                e_lossi.append(eval_loss / j )
        
    
    avg_eval_loss = eval_loss / len(eval_loader)
    if hyperparameter_search:
        train.report({"loss" : avg_eval_loss})
    return e_lossi




def train_model(params, model, training_dataset, evaluation_dataset, hyperparameter_search=True):
    train_loader, eval_loader = ret_dataloaders(training_dataset, evaluation_dataset, params["batch_size"])
    optimizer = AdamW(params = model.parameters(), lr=params["lr"])
    
    
    total_steps = (len(training_dataset) // params["batch_size"]) * params["epochs"]
    warmup_steps = int(total_steps * 0.1) # Standard 10th
    scheduler = get_scheduler("linear", 
                              optimizer=optimizer,           
                              num_warmup_steps= warmup_steps,  
                              num_training_steps=total_steps)  
    
    
    for i in range(1, params["epochs"] + 1):
        model, t_loss = train_epoch(train_loader, scheduler, optimizer, model, hyperparameter_search, i)
        e_loss = eval_epoch(eval_loader, model, hyperparameter_search, i)
        print("epoch: ", i)
        
    
    if not hyperparameter_search:
        return model, e_loss, t_loss


def device_check():
    # Simple GPU check

    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        gpu_per_trial = 1 / num_gpus
        cpu_per_trial = 0
    else:
        device = "cpu"
        gpu_per_trial = 0 
        cpu_per_trial = 2  # Something low

    return device, gpu_per_trial, cpu_per_trial


def random_hyperparameter_search(model, training_dataset, evaluation_dataset, gpu_per_trial, cpu_per_trial, num_trials=10):
    hyperparameter_search = True
    analysis = tune.run(
            tune.with_parameters(
                train_model,                                # Name of function - normally trainer.train or similar
                model=model,                                # Model input
                training_dataset=training_dataset,          # Syntax
                evaluation_dataset=evaluation_dataset,      # Syntax
                hyperparameter_search=hyperparameter_search,
            ),
            resources_per_trial={
                "gpu": gpu_per_trial,                        # Can be set to decimals for simultaneous trials if compute allows
                "cpu": cpu_per_trial
            },
            config={
                "lr": tune.choice([2e-5, 3e-5, 5e-5]),     # Per original bert paper
                "batch_size": tune.choice([16, 32, 64]),   # Batch sizes to test
                "epochs": tune.choice([1]),                # Epoch choices, only 1 since more are annoying
                    }, 
            num_samples=num_trials,                                # Total number of trials
            metric="loss",                                 # The metric to optimize - loss on evaluation set
            mode="min"                                     # As low as possible
            )

        
    return analysis.get_best_config(metric="loss", mode="min") # Returns the configuration that gives the least loss on the evaluation set


def load_model(modelname, num_labels, tokenizer, device):
    model = BertForTokenClassification.from_pretrained(modelname, 
                                                    num_labels=num_labels, 
                                                    vocab_size=tokenizer.vocab_size, 
                                                    ignore_mismatched_sizes=True).to(device)

    return model





    


