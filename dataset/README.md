# Download dataset

## ViHSD dataset
[https://huggingface.co/datasets/uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd)

```python
from datasets import load_dataset

train = load_dataset("sonlam1102/vihsd", split="train")
dev   = load_dataset("sonlam1102/vihsd", split="validation")
test  = load_dataset("sonlam1102/vihsd", split="test")
```

## VOZ-HSD dataset
[https://huggingface.co/datasets/tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD)

```python
from datasets import load_dataset

train = load_dataset("tarudesu/VOZ-HSD", split="train")
```

## Label overview

### ViHSD (3 nhãn → mapping về binary)

| Label gốc | ID gốc | Label sau mapping | ID mới |
|-----------|--------|-------------------|--------|
| CLEAN     | 0      | NON-HATE          | 0      |
| OFFENSIVE | 1      | NON-HATE          | 0      |
| HATE      | 2      | HATE              | 1      |

### VOZ-HSD (2 nhãn)

| Label gốc | ID gốc | Label sau mapping | ID mới |
|-----------|--------|-------------------|--------|
| CLEAN     | 0      | NON-HATE          | 0      |
| HATE      | 1      | HATE              | 1      |

> **Lưu ý:** Label trong VOZ-HSD được tạo bởi AI classifier (ViSoBERT-HSD), không phải human-annotated.

# fastText word vectors

Link: https://fasttext.cc/docs/en/crawl-vectors.html

> Download file tiếng Việt: `cc.vi.300.vec.gz`

> Download trực tiếp: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz

```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```