# models

Here is where you keep your config files and models. Example of config file used for training:

```yml
wandb_project: "intent_classifier"
dataset_name: "confusion"
sent_hl_units: 64
sent_dropout: 0.5
learning_rate: 0.0001
epochs: 1000
callback_patience: 100
validation_split: 0.1
```

After training your model, the model should be saved in this folder, and the complete config used in the process will be saved with the same name as the model saved, but with the suffix `_config.yml`.

