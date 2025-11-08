"""
This script works as a module and as a CLI tool.
To use it as a module, you can do:
```
from intent_classifier import IntentClassifier

classifier = IntentClassifier(config="configs/confusion.yml", examples_file="data/confusion_intents.yml")
classifier.train(save_model="models/confusion-clf-v1/")
classifier.predict(input_text="oi")
# classifier.cross_validation(n_splits=5)
```
Or, or you can use it as a CLI tool, you can do:
```
cd intent_classifier

python intent_classifier.py train \
    --config="models/confusion-v1_config.yml" \
    --examples_file="data/confusion_intents.yml" \
    --save_model="models/confusion-v1.keras"

python intent_classifier.py predict \
    --load_model="models/confusion-v1.keras" \
    --input_text="teste"
    
```
"""
# instalar alguns pacotes auxiliares

import os
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import yaml
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score

import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_text
import tensorflow_hub as hub
from tensorflow.keras.saving import register_keras_serializable

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback # WandbModelCheckpoint


@register_keras_serializable()
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, hub_url, trainable=False, **kwargs):
        super(HubLayer, self).__init__(**kwargs)
        self.hub_module = hub.load(hub_url)
        self.hub_module.trainable = trainable
    def call(self, inputs):
        return self.hub_module(inputs)

@dataclass
class Config:
    dataset_name: str = "undefined"
    codes : List[str] = None
    architecture: str = "v0.1.5"
    task: str = "undefined"
    stop_words_file: Optional[str] = None
    wandb_project: Optional[str] = None
    min_words: int = 1
    embedding_model: Union[str, List[str]] = 'https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2'
    sent_hl_units: Union[int, List[int]] = 32
    sent_dropout: Union[float, List[float]] = 0.1
    l1_reg: float = 0.01
    l2_reg: float = 0.01
    epochs: int = 500
    callback_patience: int = 20
    learning_rate: Union[float, List[float]] = 5e-3
    validation_split: float = 0.2

def remove_duplicate_words(text):
    words = text.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)


def fetch_model_from_wandb(url: str) -> str:
    """Download a model artifact from W&B or return local path.

    If ``url`` is a local file path or ``file://`` URL, it is returned as-is.
    Otherwise the artifact is downloaded via the W&B API using the key from
    ``WANDB_API_KEY`` and saved to the models/ folder. The path to the downloaded 
    model file is returned.
    """
    # Support file:// URLs and plain local paths for offline tests
    if url.startswith("file://"):
        local = url[7:]
        if os.path.exists(local):
            return local
    if os.path.exists(url):
        return url

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        wandb.login()
    else:
        wandb.login(key=api_key)
    
    api = wandb.Api()
    if ":" not in url:
        url = f"{url}:latest"
    artifact = api.artifact(url)
    
    # Create models directory if it doesn't exist
    models_dir = Path(os.path.join(os.path.dirname(__file__), "models"))
    models_dir.mkdir(exist_ok=True)
    
    # Download artifact to models directory
    path = artifact.download(root=models_dir)
    
    # Try to locate a Keras model file inside the downloaded directory
    for fname in os.listdir(path):
        if fname.endswith(".keras") or fname.endswith(".h5"):
            return os.path.join(path, fname)
    return path


class IntentClassifier:

    def __init__(self, config = None, load_model = None, examples_file = None):
        if load_model is None:
            self.model = None
        else:
            if os.path.exists(load_model):
                self.model = tf.keras.models.load_model(load_model)
                print(f"Loaded keras model from {load_model}.")
            else:
                # Try to load from W&B
                local_model_path = fetch_model_from_wandb(load_model) # local_model_path é a string do caminho
                self.model = tf.keras.models.load_model(local_model_path)
                print(f"Loaded keras model from {load_model}.")
            if self.model is None:
                raise ValueError(f"Model file not found: {load_model}. Try to load from W&B or provide a valid path to a Keras model file.")
        # Load config
        self._load_config(config, load_model)
        # Load intents from the examples file if provided
        self._load_intents(examples_file)
        # Initialize stop_words
        self._load_stop_words(self.config.stop_words_file)
        # Set up one-hot encoder
        self._setup_onehot_encoder()
        # Set up W&B
        if self.config.wandb_project:
              # Create wandb run instance
              self.wandb_run = wandb.init(project=self.config.wandb_project, 
                                          config=self.config.__dict__)
              # Create and log artifact
              if self.examples_file is not None:
                  artifact = wandb.Artifact("my_dataset", type="dataset")
                  artifact.add_file(examples_file) # Assuming 'examples_file' is the dataset file
                  self.wandb_run.log_artifact(artifact)

    def _load_config(self, config, load_model):
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = Config(**yaml.safe_load(f))
            print(f"Loaded config from {config}.")
        elif isinstance(config, Config):
            self.config = config
        elif config is None:
            # Load from a model
            if load_model is not None:
                # self.model = tf.keras.models.load_model(load_model)
                # print(f"Loaded keras model from {load_model}.")
                config_path = load_model.replace(".keras", "_config.yml")
                if not os.path.exists(config_path):
                    raise ValueError('The `config` object must be provided for this IntentClassifier.')
                with open(config_path, 'r') as f:
                    self.config = Config(**yaml.safe_load(f))
            else:
                # We must load the config of this model properly...
                raise ValueError('The `config` object must be provided for this IntentClassifier.')
        return self.config
    
    def _load_intents(self, examples_file):
        self.examples_file = examples_file
        if examples_file is not None:
            pprint(f"Loading intents from {examples_file}...")
            with open(examples_file, 'r') as f:
                self.intents_data = yaml.safe_load(f)
            # Preprocess intents
            input_text = []
            labels = []
            for i in self.intents_data:
                input_text += i['examples']
                labels += [i['intent']]*len(i['examples'])
            input_text = np.array(input_text)
            labels = np.array(labels)
            # Shuffle data
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            self.input_text = input_text[indices]
            self.input_text = tf.convert_to_tensor(self.input_text, dtype=tf.string)
            self.labels = labels[indices]
            self.codes = np.unique(self.labels)
            self.config.codes = self.codes.tolist()
        else: # Means that the example_file is not provided
            # Then the model will be used only to predict, no need to load training data
            self.codes = self.config.codes
        return self.codes
    
    def _load_stop_words(self, stop_words_file: str):
        if stop_words_file is None:
            self.stop_words = []
            return
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            self.stop_words = f.read().split('\n')
        print(f"Loaded {len(self.stop_words)} stop words from {stop_words_file}.")
        return self
    
    def _setup_onehot_encoder(self):
        assert self.codes is not None, "codes must be set before setting up the encoder."
        self.onehot_encoder = OneHotEncoder(categories=[self.codes],)\
                                  .fit(np.array(self.codes).reshape(-1, 1))
        return self.onehot_encoder

    def _get_callbacks(self):
        callbacks = []
        if self.config.callback_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                    patience=self.config.callback_patience,
                    restore_best_weights=True)
            )
        if self.config.wandb_project:
            callbacks.append(WandbMetricsLogger())
        
        # Configure ExponentialDecay
        if self.config.learning_rate is not None and not isinstance(self.config.learning_rate, str):
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=False
            )
            
            # Modified learning rate scheduler to properly handle epoch parameter
            def lr_scheduler(epoch, lr):
                return lr_schedule(epoch).numpy().astype(float)
            
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks.append(lr_scheduler_callback)
        return callbacks
    
    def finish_wandb(self):
        if self.config.wandb_project and self.wandb_run:
            self.wandb_run.finish()

    def preprocess_text(self, text):
        text = tf.strings.lower(text)
        if self.stop_words:
            words = tf.strings.split(text)
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(self.stop_words)), axis=1))
            text = tf.strings.reduce_join(words, separator=' ')
        if self.config.min_words:
            words = tf.strings.split(text)
            words = tf.boolean_mask(words, tf.reduce_all(tf.not_equal(words[:, None], tf.constant(["?", ".", ",", "!"])), axis=1))
            num_words = tf.shape(words)[0]
            if num_words <= self.config.min_words:
                # Instead of setting it to an empty string:
                # text = ""
                # We should create a dummy string with enough words
                padding = " ".join(["<>"] * (self.config.min_words + 1))
                text = tf.constant(padding)
        return tf.expand_dims(tf.strings.as_string(text), 0)  # Convert to 1-D Tensor

    def make_model(self, config: Config):
        # Set the random seed for reproducibility
        seed = 42
        tf.random.set_seed(seed)  # Assuming you have a random_seed in your config
        # Extract config values
        sent_hl_units, sent_dropout = config.sent_hl_units, config.sent_dropout
        l1_reg, l2_reg = config.l1_reg, config.l2_reg
        output_size = len(self.codes)
        # Build model
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)  # Set seed in initializer
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
        # Sentence encoder
        encoder = HubLayer(config.embedding_model, trainable=False, name="sent_encoder")(text_input)
        # Hidden layer
        sent_hl = tf.keras.layers.Dense(sent_hl_units,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                        activation=None,  # No activation here yet
                                        name='sent_hl')(encoder)
        sent_hl_norm = tf.keras.layers.BatchNormalization()(sent_hl)  # Add batch normalization
        sent_hl_activation = tf.keras.layers.Activation('relu')(sent_hl_norm)  # Activation after batch normalization
        sent_hl_dropout = tf.keras.layers.Dropout(sent_dropout, seed=seed)(sent_hl_activation)  # Set seed in dropout
        # Output layer
        sent_output = tf.keras.layers.Dense(output_size,
                                            kernel_initializer=initializer,
                                            activation='softmax',
                                            name="sent_output")(sent_hl_dropout)
        model = tf.keras.Model(inputs=text_input, outputs=sent_output)
        return model

    def train(self, save_model: Optional[str] = None, tf_verbosity: int = 1):
        pprint(self.config.__dict__)
        # Update task config parameter
        self.config.task = "train"
        assert self.examples_file is not None, "examples_file must be provided when the IntentClassifier was created."
        
        # Extract one-hot encoded labels
        labels_ohe = self.onehot_encoder\
                            .transform(self.labels.reshape(-1, 1))\
                            .toarray()
        # Split
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            self.input_text.numpy(), labels_ohe, # Convert to NumPy array for splitting
            test_size=self.config.validation_split,
            stratify=labels_ohe,      # Ensure class distribution is preserved
            random_state=42           # For reproducibility
        )
        # Now apply preprocessing using preprocess_text *after* splitting:
        X_train = tf.map_fn(self.preprocess_text, tf.constant(X_train_text), dtype=tf.string)
        X_val = tf.map_fn(self.preprocess_text, tf.constant(X_val_text), dtype=tf.string)
        # Extract config values
        learning_rate = self.config.learning_rate
        epochs = self.config.epochs
        # New model from scratch
        self.model = self.make_model(self.config)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.F1Score(average='macro')])
        # Train the model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            # batch_size=16,
            shuffle=True,
            epochs=epochs,
            verbose=tf_verbosity,
            callbacks=self._get_callbacks()
        )
        # Save model
        if save_model is not None:
            self.save_model(path=save_model)
        return self.model

    def save_model(self, path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        # Save model in SavedModel format
        if path.endswith('/'):
            # Remove trailing slash if present
            path = path.rstrip('/')
        # Save the model
        self.model.save(path)
        # Save config into a yaml file inside the model directory
        config_path = path.replace(".keras", "_config.yml") #os.path.join(os.path.dirname(path), f"{self.config.dataset_name}_config.yml")
        with open(config_path, 'w') as f:
            f.write(yaml.dump(self.config.__dict__))
        print(f"Model saved to {path}.")
        if self.config.wandb_project:
            # Crie e envie o artifact
            artifact = wandb.Artifact(
                name=f"{self.config.dataset_name}-clf-v1",
                type="model",
                description="Modelo Keras v1 para classificação de intenção"
            )
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)
            self.wandb_run.finish()

    def predict(self, input_text, true_labels: list = None, log_to_wandb: bool = False):
        self.config.task = "predict"  # Set the task to "predict"
        original_input_is_string = isinstance(input_text, str)
        if original_input_is_string:
            input_text_list = [input_text]  # Convert single string to a list for processing
        else:
            input_text_list = input_text
        # Preprocess each string in the list
        preprocessed_texts = [self.preprocess_text(tf.constant(text)) for text in input_text_list]
        preprocessed_texts = tf.concat(preprocessed_texts, axis=0) # Stack the tensors into a single tensor
        # Predict probabilities for all strings at once
        all_probs = self.model.predict(preprocessed_texts)
        results = []
        predicted_labels_for_log = []
        for i in range(all_probs.shape[0]):
            current_probs = all_probs[i] # Probabilities for the i-th input text
            # Determine the intent name with the highest probability
            highest_prob_idx = np.argmax(current_probs)
            highest_prob_intent_name = self.codes[highest_prob_idx]
            predicted_labels_for_log.append(highest_prob_intent_name)
            # Create a dictionary of probabilities for each intent name
            probs_dict = {code: float(current_probs[j]) for j, code in enumerate(self.codes)}
            results.append((highest_prob_intent_name, probs_dict))
        # Log to Wandb if requested
        if log_to_wandb and self.config.wandb_project:
            # Get the current run ID if it exists, otherwise start a new run
            run_id = wandb.run.id if wandb.run else wandb.util.generate_id()
            # Initialize wandb with the run ID
            with wandb.init(project=self.config.wandb_project, id=run_id, resume="allow"):
                wandb.log({
                    "inputs": input_text_list, # Log the list of original input texts
                    "true_labels": true_labels,
                    "predictions": predicted_labels_for_log # Use the extracted list of highest prob intents
                })
        # Return a single tuple if the original input was a string, otherwise a list of tuples
        if original_input_is_string:
            return results[0]
        return results

    def cross_validation(self, n_splits: int = 3):
        assert self.examples_file is not None, "examples_file must be provided when the IntentClassifier was created."
        # Update task config parameter
        self.config.task = "cross_validation"
        kf = StratifiedKFold(n_splits=n_splits)
        # Preprocess the entire dataset before cross-validation
        self.input_text = tf.map_fn(self.preprocess_text, self.input_text, dtype=tf.string)
        # Get one-hot encoded labels before the loop
        labels_ohe = self.onehot_encoder.transform(self.labels.reshape(-1, 1)).toarray()
        results = []
        for i, (train_index, test_index) in enumerate(kf.split(self.input_text)):
            print(f"Fold {i+1}/{n_splits}")
            # Create and log a new Wandb run for each fold
            with wandb.init(project=self.config.wandb_project, config=self.config.__dict__, group="cross_validation", reinit=True, job_type=f"fold_{i+1}"):
                # Create a new model for each fold
                model = self.make_model(self.config)
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                    metrics=[tf.keras.metrics.F1Score(average='macro')])
                # Train the model on the current fold
                model.fit(self.input_text[train_index], labels_ohe[train_index],
                          epochs=self.config.epochs, verbose=0,
                          callbacks=self._get_callbacks()) # WandbMetricsLogger is already added in _get_callbacks()
                # Predict on the test set for the current fold
                preds = model.predict(self.input_text[test_index])
                preds = self.onehot_encoder.inverse_transform(preds)
                labels = self.onehot_encoder.inverse_transform(labels_ohe[test_index])
                # Evaluate the model and store the results
                res = classification_report(labels, preds, output_dict=True)
                res['kappa'] = cohen_kappa_score(labels, preds)
                results.append(res)
                # Log fold-specific metrics
                wandb.log(res)
        # Calculate and print average metrics
        avg_f1 = np.mean([r['macro avg']['f1-score'] for r in results])
        avg_kappa = np.mean([r['kappa'] for r in results])
        print(f"Average f1-score: {avg_f1}")
        print(f"Average kappa: {avg_kappa}")
        # Log average metrics to a summary run
        with wandb.init(project=self.config.wandb_project, config=self.config.__dict__, group="cross_validation", reinit=True, job_type="summary"):
            wandb.log({"avg_f1": avg_f1, "avg_kappa": avg_kappa})
        if self.config.wandb_project and self.wandb_run:
            self.wandb_run.finish()
        return results


# This script works as a module and as a CLI tool
if __name__ == "__main__":
    import fire
    # Instead of fire.Fire(IntentClassifier),
    # Define the functions to be used by Fire CLI so that 
    #  it's not cluttered with all the functions in the IntentClassifier class
    def train(config: str, examples_file: str, save_model: str):
        """Train the model with the given configuration and examples."""
        classifier = IntentClassifier(config=config, examples_file=examples_file)
        classifier.train(save_model=save_model)
        print("Training completed successfully!")
    def predict(load_model: str, input_text: str):
        """Make predictions using a trained model."""
        classifier = IntentClassifier(load_model=load_model)
        predictions = classifier.predict(input_text)
        print(f"Predictions: {predictions}")
    def cross_validation(n_splits: int = 3):
        """Run cross-validation on the model."""
        classifier = IntentClassifier()
        results = classifier.cross_validation(n_splits=n_splits)
        print("Cross-validation completed successfully!")
    fire.Fire({
        'train': train,
        'predict': predict,
        'cross_validation': cross_validation
    }, serialize=False)

