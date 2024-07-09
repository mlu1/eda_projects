---
license: apache-2.0
base_model: t5-small
tags:
- generated_from_keras_callback
model-index:
- name: Mluleki/dyu-fr-translation
  results: []
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# Mluleki/dyu-fr-translation

This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Train Loss: 3.0678
- Validation Loss: 2.8734
- Epoch: 9

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'AdamWeightDecay', 'learning_rate': 2e-05, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False, 'weight_decay_rate': 0.01}
- training_precision: float32

### Training results

| Train Loss | Validation Loss | Epoch |
|:----------:|:---------------:|:-----:|
| 3.6875     | 3.2490          | 0     |
| 3.4758     | 3.1470          | 1     |
| 3.3749     | 3.0798          | 2     |
| 3.3153     | 3.0285          | 3     |
| 3.2551     | 2.9931          | 4     |
| 3.2077     | 2.9603          | 5     |
| 3.1696     | 2.9331          | 6     |
| 3.1311     | 2.9081          | 7     |
| 3.0996     | 2.8899          | 8     |
| 3.0678     | 2.8734          | 9     |


### Framework versions

- Transformers 4.31.0
- TensorFlow 2.15.0
- Datasets 2.14.2
- Tokenizers 0.13.3
