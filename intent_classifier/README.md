# intent-classifier

How to install:
```
conda create -n intent-clf python=3.11
conda activate intent-clf
pip install -r requirements.txt
```

How to use:
```
python intent_classifier.py train \
    --examples_file="data/confusion_intents.yml" \
    --config="models/confusion-v1_config.yml" \
    --save_model="models/confusion-v1.keras"

python intent_classifier.py predict \
    --load_model="models/confusion-v1.keras" \
    --input_text="testing"
```