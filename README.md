# rise_test_repo

# Environment setup
pip install -r requirements.txt

# Usage
There are 2 required arguments, --system and --search.
-- system determines whether to use all available labels or a subset.
-- search determines if a hyperparameter search is do be done prior to training or just use the best hyperparameters I found.

Examples:
python main.py() --system=A --Yes
This code would run a hyperparameter search prior to training using all NER classes.

python main.py() --system=B --No
This code will use already found hyperparameters and use only 10 NER classes and the "O" Class.

# Recommendations
Use a GPU.
The hyperparameter search takes a while - and I have done it locally so you dont have to :) - but feel free to test it.

# Conclusions
Dataset
- Uneven label distribution
- Fairly large dataset
- Short texts (almost all under 70 tokens)

Model performance
- TODO

Drawbacks of method
- TODO






