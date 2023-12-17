# rise_test_repo

# Environment setup
The requirements.txt were created using pipreqs. To use simply run:
pip install -r requirements.txt

# Usage
python3 main.py

There is one required argument --system and two optional arguments --search and --model_path
-- system determines whether to use all available labels or a subset.
-- search determines if a hyperparameter search is do be done prior to training or just use the best hyperparameters I found.
-- model_path can be used if you have already trained a model with this repository and want to re-run the testing part and should point to where your model is located, by default this is model_B or model_A depending on what model was trained.

Examples:
python main.py() --system=A --Yes
This code would run a hyperparameter search prior to training using all NER classes.

python main.py() --system=B --No
This code will use already found hyperparameters and use only 10 NER classes and the "O" Class.

# Recommendations
Use a GPU.
The hyperparameter search takes a while - and I have done it locally so you dont have to :) - but feel free to test it.



# Main findings and drawbacks
Drawbacks of method:
The classes are heavily imbalanced, and the model would have probably benefitted from some sampling strategy.
There are other model architectures that perform better on NER tasks, like LUKE which also uses entity embeddings.
The hyperparameter search was fairly simple and could have been more complex, i.e., using Bayesian or Population-based approaches. It also wasn’t done for very long and the results from it might not be indicative of the best hyperparameters after one full epoch.

Findings:
Overall, the model performs well, especially on more frequently occurring classes.
System A has a larger label space than system B, but this doesn’t seem to affect the predictive ability of the model on the classes. The examples in the subset of classes in B seem to be of sufficient quantity and quality – and the model good enough – to predict fairly well despite a larger label space. 
The similarity in performance indicate that the removed classes didn’t impact the model’s predictions on remaining classes much – the model probably didn’t confuse one of the removed classes for one of the remaining classes or vice versa OR the removed classes were so small that it didn’t matter much.







