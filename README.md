# ConditionalRandomField-pytorch
CRF implemented by pytorch

This implementation borrows mostly from [AllenNLP CRF module](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py) and [pytorch-crf](https://github.com/kmkurn/pytorch-crf).

Difference with pytorch-crf is using batch opperation when viterbi decode.

Difference with Allennlp is I discard top-k viterbi decode.

# Usage

```
from ConditionalRandomField import ConditionalRandomField, allowed_transitions

label_dic = {'O':0, 'B-a':1, 'I-a':2, 'O-a':3, 'U-a':4, 'L-a':5, 'U-b':6}
constraints = allowed_transitions(constraint_type='BIOUL', labels=label_dic)
crf = ConditionalRandomField(num_tags=len(label_dic), constraints=constraints)
```
