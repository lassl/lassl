# About Configs

Each config files are sample files for various model types and training environments.  
A single config file contains three main arguments:  
1. **model configs**
2. **collator arguments**
3. **training arguments**


## 1. Model Configs
Model configs contain the structural information of model which you will train on.  
You can edit the model structure by setting suitable parameters according to the [link](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForPreTraining).  


## 2. Collator Arguments
In collator arguments, you can set all the hyperparameters applied to the data collator. Please check out for the [collator](https://github.com/lassl/lassl/blob/main/src/lassl/collators.py) for each model. All the collators are implemented based on the original paper.   


## 3. Training Arguments
Training arguments include all arguments to be used for training. Since all the training arguments will be the parameters of *HF Trainer*, we strongly suggest to take a look at the [link](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).