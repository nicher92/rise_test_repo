# rise_test_repo

# Environment setup
Requirements.txt was created using pipreqs.
pip install -r requirements.txt

# Usage
There are 1 required arguments, --system and 2 optional arguments --search and --model_path
-- system determines whether to use all available labels or a subset.
-- search determines if a hyperparameter search is do be done prior to training or just use the best hyperparameters I found.
-- model_path is used if you have already trained a model using the repository and want to re-use it.

Examples:
python main.py() --system=A --Yes
This code would run a hyperparameter search prior to training using all NER classes.

python main.py() --system=B --model_path=model_B
This code will use an already trained model on system B and evaluate it on system B.


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
System A has a larger label space than system B, which should make it easier for the model to correctly predict the correct class - but this didnt seem to be the case. All labels not in the allowed classes are turned to “O” but this doesn’t make the O class that much bigger since many of the classes are small. 
The similarity in performance indicate that the removed classes didn’t impact the model’s predictions on remaining classes much – the model probably didn’t confuse one of the removed classes for one of the remaining classes or vice versa OR the removed classes were so small that it didn’t matter much.






