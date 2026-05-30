# 6. Result Analysis or Error Analysis

## 6.1 Experimental Results

The experimental results show that transformer-based models generally outperform traditional lexical and static-embedding baselines for Vietnamese hate speech detection. This is expected because hate speech often depends on context, informal phrasing, and interactions between target mentions and hostile predicates. Sparse lexical models can capture frequent keywords, but they are less effective when hate speech is expressed implicitly or through varied social media language. Static-embedding sequence models provide better contextualization than lexical features, but they remain weaker than PhoBERT-based models because they do not benefit from large-scale Vietnamese language pretraining.

Among the transformer-based methods, the direct AmpleHate_PhoBERT baseline improves over general PhoBERT-based classification baselines by adding target-aware attention. However, its improvement is limited when the target extraction module is not well matched to Vietnamese. If most comments do not contain target entities detectable by the original NER component, the model frequently falls back to the `[CLS]` representation and behaves similarly to a standard PhoBERT classifier.

The proposed ViAmpleHate_PhoBERT model addresses this limitation by using Vietnamese-aware target extraction and by introducing a separate attack cue pathway. Across the evaluated settings, the proposed model improves the most relevant metrics for hate speech detection: macro-F1 and HATE-class F1. These gains indicate that the model is better at handling the minority hate class, which is the main target of the task.

The results also suggest that the benefit of ViAmpleHate depends on the quality and coverage of the extracted cues. When target and attack cues are detected more frequently and more accurately, relation-bank attention has more useful evidence to exploit. When cue coverage is low, the model must rely more heavily on the implicit `[CLS]` pathway, reducing the gap between ViAmpleHate and the direct AmpleHate or PhoBERT baselines.

Accuracy should be interpreted cautiously. In hate speech datasets, the majority class is usually `NON-HATE`, so a model can obtain high accuracy while still missing many hate speech instances. Macro-F1 and HATE-F1 are therefore more informative than accuracy. A useful hate speech detector should not only classify the majority class correctly but also recover the minority hate class with acceptable precision and recall.

Overall, the results support the main design motivation of ViAmpleHate. Vietnamese hate speech detection benefits from target-aware modeling, but target awareness must be adapted to Vietnamese linguistic patterns. Vietnamese NER, target cue mining, attack cue mining, relation-bank attention, and adaptive gating together provide a more suitable mechanism than directly transferring an English-oriented AmpleHate pipeline.

## 6.2. Error Analysis

The remaining errors can be grouped into several categories. The first category is confusion between offensive language and hate speech. Vietnamese social media comments often contain insults, profanity, or aggressive expressions that are offensive but not necessarily hate speech. If a model relies too strongly on attack cues, it may classify these comments as `HATE` even when no protected or social group is targeted. This leads to false positives.

The second category is implicit hate speech. Some hate speech comments do not contain explicit slurs, named entities, or obvious attack predicates. They may express hostility through sarcasm, coded references, stereotypes, indirect comparison, or shared social context. Such cases are difficult for cue-based models because the relevant target or attack relation is not directly visible in the token sequence.

The third category is ambiguous target reference. Vietnamese expressions such as group nouns, pronouns, and informal referential phrases can indicate a target group, but they can also occur in neutral or humorous comments. When these expressions are detected as target cues without sufficient hostile context, the model may overestimate the likelihood of hate speech.

The fourth category is incomplete cue coverage. Vietnamese online language changes quickly and contains many spelling variants, slang forms, abbreviations, and creative profanity. A fixed cue bank cannot cover all possible target and attack expressions. When important cues are missing from the lexicon or are not normalized correctly, the model may fail to activate the explicit target or attack pathways and may instead rely on the implicit sentence representation.

The fifth category is tokenization and span-alignment error. Vietnamese word segmentation, NER spans, and PhoBERT subword tokenization do not always align perfectly. Multiword expressions are especially difficult because a phrase may be segmented differently across preprocessing, NER, and tokenizer stages. If the extracted cue cannot be matched to the correct token positions, the attention module may attend to incomplete or irrelevant evidence.

The sixth category is threshold sensitivity. The optimal decision threshold can vary across datasets, domains, and class distributions. A threshold that improves macro-F1 on the validation set may not remain optimal under a different test distribution. This issue is especially important for deployment, where the proportion and style of hate speech may differ from the training data.

The seventh category is class imbalance. Because hate speech is typically much less frequent than non-hate speech, the model sees fewer positive examples during training. Weighted loss and threshold tuning reduce this problem but do not remove it entirely. Minority-class errors remain likely when hate speech examples are diverse and underrepresented.

These error patterns suggest several directions for improvement. First, the target and attack cue banks should be expanded through data-driven mining and manual validation. Second, per-instance prediction logs should be saved for systematic false-positive and false-negative analysis. Third, future models could incorporate target span supervision, sarcasm detection, or external social-context information. Finally, calibration methods may help make decision thresholds more stable across datasets and deployment domains.
