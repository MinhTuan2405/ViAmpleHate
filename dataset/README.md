# Down load dataset via these link
## ViHSD dataset
[https://huggingface.co/datasets/uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd)

code
```
from datasets import load_dataset

train = load_dataset("sonlam1102/vihsd", split="train")
dev = load_dataset("sonlam1102/vihsd", split="validation")
test = load_dataset("sonlam1102/vihsd", split="test")

```

## Kaggle HSD
[https://www.kaggle.com/datasets/cthng123/hate-speech-detection-vietnamese/data](https://www.kaggle.com/datasets/cthng123/hate-speech-detection-vietnamese/data)

code
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("cthng123/hate-speech-detection-vietnamese")

print("Path to dataset files:", path)
```


## Label overview

| Label | ID |
|-------|----|
| CLEAN | 0 |
| OFFENSIVE | 1 |
| HATE | 2 |
