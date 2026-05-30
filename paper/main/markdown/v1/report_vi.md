# ViAmpleHate: A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection

> **Mục đích của file này.** Đây là bản thảo nội dung (content blueprint) để viết bài báo LaTeX 8–10 trang theo template **ACL Conference** (Overleaf: *Association for Computational Linguistics — ACL Conference*, `acl_latex.tex` + `acl.sty`).
> Mỗi mục dưới đây tương ứng một `\section{}` / `\subsection{}` trong LaTeX. Các công thức viết bằng cú pháp LaTeX để copy thẳng vào `equation`/`align`. Bảng viết ở dạng Markdown, cần chuyển sang `table` + `booktabs` (`\toprule`/`\midrule`/`\bottomrule`).
> Các chỗ cần chèn hình được đánh dấu bằng khối **`📌 [CHÈN HÌNH]`** — nêu rõ chèn hình gì, đặt ở đâu, lấy từ file nào (hoặc cần vẽ mới).
>
> **Ghi chú độ dài (ACL 2 cột):** Intro ~1 tr · Related Work ~0.75 tr · Dataset ~1 tr · Methodology ~2 tr (có hình kiến trúc) · Experiments ~1.5 tr · Results & Error Analysis ~2 tr (có bảng + confusion matrix) · Conclusion ~0.5 tr. Tổng ~8.5–9.5 tr + references.

---

## Abstract

*(~150–200 từ, viết cuối cùng. Gợi ý nội dung:)*

Hate speech detection on Vietnamese social media is challenging because hateful intent is often expressed through informal group references, slang, and implicit cues rather than explicit named entities. Target-aware models such as **AmpleHate** improve detection by attending to the relationship between a sentence and its hate target, but the original pipeline relies on English named-entity recognition (NER) and English-centric target categories, which transfer poorly to Vietnamese. We propose **ViAmpleHate**, a Vietnamese adaptation of AmpleHate built on **PhoBERT**. ViAmpleHate replaces English NER with Vietnamese NER, adds Vietnamese **target-cue** and **attack-cue** banks, models three relation channels (explicit target, implicit context, attack) through a **relation-bank attention**, corrects a batch-level attention contamination bug, and injects relation evidence through an **instance-adaptive gate** instead of a fixed scalar. Training combines weighted cross-entropy with a contrastive objective. On **ViHSD** and **VOZ-HSD** (binary NON-HATE/HATE setting), ViAmpleHate improves macro-F1 and the minority HATE-class F1 over a faithful AmpleHate baseline and over TF-IDF, BiLSTM, and PhoBERT-CNN baselines, confirming that target-aware modeling must be linguistically adapted to Vietnamese.

> 📌 **[CHÈN HÌNH — không bắt buộc]** Có thể thêm 1 *teaser figure* nhỏ ở cột phải trang 1 minh hoạ ý tưởng: câu tiếng Việt → tách *target cue* / *attack cue* → attention → dự đoán HATE. (Cần **vẽ mới**, ví dụ bằng PowerPoint/draw.io.) Nếu thiếu chỗ, bỏ qua và để Figure kiến trúc ở Section 4.

---

## 1. Introduction

*(~1 trang. Các luận điểm cần triển khai thành văn:)*

- **Bối cảnh & động lực.** Mạng xã hội tiếng Việt phát triển nhanh; nội dung thù ghét (hate speech) gây hại cho cá nhân và cộng đồng. Phát hiện hate speech tự động là cần thiết để kiểm duyệt nội dung quy mô lớn.
- **Thách thức đặc thù tiếng Việt.** (1) Văn bản nhiễu: teencode, viết tắt, lặp ký tự, emoji, sai chính tả. (2) Đối tượng bị tấn công thường được nói đến qua *đại từ, danh từ chỉ nhóm, cách gọi suồng sã*, không phải thực thể có tên (named entity). (3) Mất cân bằng lớp nghiêm trọng: lớp HATE là thiểu số.
- **Hướng tiếp cận target-aware & khoảng trống.** AmpleHate cho thấy việc mô hình hoá *quan hệ giữa câu và đối tượng bị nhắm tới* giúp phát hiện cả hate ẩn (implicit). Nhưng AmpleHate gốc dùng **NER tiếng Anh** ⇒ trên tiếng Việt hầu như không tìm được target hợp lệ (đo được: chỉ **21/24.048 ≈ 0,09%** mẫu train có target từ NER tiếng Anh), khiến mô hình thoái hoá về một bộ phân loại PhoBERT thuần dựa trên `[CLS]`.
- **Đóng góp (contributions).** Liệt kê dưới dạng bullet in đậm:
  1. **ViAmpleHate** — bản thích ứng tiếng Việt của AmpleHate trên PhoBERT, thay NER tiếng Anh bằng **NER tiếng Việt + ngân hàng cue đối tượng (target cue)** ⇒ tăng độ phủ target từ ~0,09% lên ~18,8–20,0%.
  2. **Tách kênh tín hiệu**: mô hình hoá riêng *target cue* (ai bị nói đến) và *attack cue* (có bị tấn công không) qua **relation-bank attention** ba kênh (explicit target / implicit context / attack).
  3. **Sửa lỗi attention theo batch** (từ `matmul(Q, Kᵀ)` cấp batch dễ trộn mẫu sang **batched `bmm` + mask** từng mẫu) và thay **fixed scalar injection** bằng **instance-adaptive gate**.
  4. **Mục tiêu huấn luyện** kết hợp weighted cross-entropy + contrastive loss để cải thiện tách lớp dưới mất cân bằng.
  5. Đánh giá trên **ViHSD** và **VOZ-HSD** (binary), so sánh với 5 baseline; cải thiện macro-F1 và HATE-F1.
- **Câu chốt cuối phần.** Nhấn mạnh: target-awareness có ích, nhưng *phải được thích ứng theo đặc trưng ngôn ngữ tiếng Việt* mới phát huy.

---

## 2. Related Work

### 2.1 AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection

*(~0,4 trang)*

- Tóm tắt ý tưởng AmpleHate: phát hiện implicit hate bằng cách **khuếch đại attention** giữa biểu diễn câu (`[CLS]`) và các **target** tiềm năng trích bằng NER; vector quan hệ target-aware được **bơm (inject)** vào biểu diễn câu trước khi phân loại.
- Cơ chế: với target token, tính HeadAttention $r = \text{softmax}(QK^\top/\sqrt d)\,V$, rồi $z = h_0 + e\cdot r$ với $e$ là hệ số cố định.
- **Hạn chế khi chuyển sang tiếng Việt:** phụ thuộc NER tiếng Anh và danh mục target theo kiểu tiếng Anh; cơ chế inject cố định; chỉ một kênh quan hệ (target), chưa mô hình hoá *attack*.
- Nêu rõ: công trình này **kế thừa ý tưởng target-aware của AmpleHate** nhưng tái thiết kế cho tiếng Việt.

### 2.2 ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts

*(~0,35 trang)*

- Giới thiệu ViTHSD: bộ dữ liệu/hướng tiếp cận **hate speech theo đối tượng (targeted)** cho mạng xã hội tiếng Việt; gán nhãn mức độ thù ghét theo từng target.
- Liên hệ: ViTHSD củng cố nhận định rằng **đối tượng (target) là tín hiệu cốt lõi** của hate speech tiếng Việt ⇒ ủng hộ động lực target-aware của ViAmpleHate.
- Khác biệt: ViAmpleHate không yêu cầu nhãn target ở mức span; thay vào đó dùng **cue bank + NER** để xác định vị trí target/attack một cách *không cần giám sát span*.

*(Tuỳ chọn ~0,1 tr: nhắc nhanh các baseline tiếng Việt: PhoBERT (Nguyen & Nguyen, 2020), PhoBERT-CNN, BiLSTM + fastText, TF-IDF — để dẫn vào phần baseline ở §5.)*

---

## 3. Dataset

### 3.1 ViHSD — Vietnamese Hate Speech Detection dataset

*(~0,3 tr)*

- ViHSD: bộ dữ liệu hate speech tiếng Việt, gồm bình luận mạng xã hội với 3 nhãn gốc: **CLEAN**, **OFFENSIVE**, **HATE**.
- Chia sẵn train/dev/test. Sau khi gộp về binary (xem §3.3), phân bố như Bảng 1.

### 3.2 VOZ-HSD — VOZ Hate Speech Detection dataset

*(~0,3 tr)*

- VOZ-HSD: bình luận từ diễn đàn VOZ (nguồn: `tarudesu/VOZ-HSD` trên HuggingFace).
- Dữ liệu được chia sẵn thành train/dev/test; phân bố lớp (sau reshape, xem §3.3) ở Bảng 1.

### 3.3 Reshape dataset (Binary reformulation)

*(~0,4 tr)*

- **Thao tác duy nhất trên dữ liệu là gán lại nhãn (relabel)**, không thay đổi/chọn lọc số lượng mẫu. Bài toán được đưa về **phân loại nhị phân**: gộp **CLEAN** và **OFFENSIVE** thành **NON-HATE**; giữ **HATE** làm lớp dương (positive).
- Lý do: tập trung vào *hate speech nhắm vào nhóm/đối tượng* thay vì toxicity/profanity nói chung; đồng thời tạo thiết lập nhãn nhất quán giữa hai bộ dữ liệu.
- Tiền xử lý chung: chuẩn hoá nhiễu (lowercase, bỏ URL, gộp ký tự lặp, chuẩn hoá teencode, ánh xạ emoji → nhãn ngữ dụng thô), **tách từ (word segmentation)** trước khi đưa vào PhoBERT.

**Bảng 1 — Thống kê dữ liệu (sau khi gộp binary).** *(LaTeX: `table` + `booktabs`)*

| Dataset | Split | NON-HATE | HATE | Tổng | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21,492 | 2,556 | 24,048 | 10.6% |
| ViHSD | Dev | 2,402 | 270 | 2,672 | 10.1% |
| ViHSD | Test | 5,992 | 688 | 6,680 | 10.3% |
| VOZ-HSD | Train | 26,993 | 3,007 | 30,000 | 10.0% |
| VOZ-HSD | Dev | 4,520 | 480 | 5,000 | 9.6% |
| VOZ-HSD | Test | 4,487 | 513 | 5,000 | 10.3% |

> 📌 **[CHÈN HÌNH] Figure 1 — Phân bố lớp.** Biểu đồ cột (bar chart) thể hiện mất cân bằng NON-HATE vs HATE trên cả 2 dataset (lấy số từ Bảng 1). **Cần vẽ mới** (matplotlib). Đặt ngay sau Bảng 1, cột đơn. Caption nhấn mạnh "HATE ≈ 10% ⇒ macro-F1/HATE-F1 quan trọng hơn accuracy".

---

## 4. Methodology

*(~2 trang — phần lõi, nên có hình kiến trúc tổng thể)*

Bài toán: phân loại nhị phân câu bình luận tiếng Việt $x$ thành NON-HATE hoặc HATE. ViAmpleHate kế thừa ý tưởng target-aware của AmpleHate và thích ứng cho tiếng Việt qua 5 thay đổi: (i) trích target theo tiếng Việt, (ii) tách target cue và attack cue, (iii) relation-bank attention, (iv) sửa attention theo batch, (v) inject quan hệ bằng adaptive gate.

> 📌 **[CHÈN HÌNH] Figure 2 — Kiến trúc tổng thể ViAmpleHate.** **Quan trọng nhất, cần vẽ mới** (draw.io/PowerPoint), đặt ở đầu Section 4, **full-width (`figure*`)** hoặc cột đơn. Sơ đồ luồng:
> `Comment → Tiền xử lý/chuẩn hoá → Word segmentation → PhoBERT encoder → [CLS] h₀ + token states H` →
> nhánh trích tín hiệu: `Vietnamese NER + Target cue bank → T_x` và `Attack cue bank → A_x` →
> `Relation-bank attention` 3 kênh: `r_exp (target) / r_imp (CLS) / r_atk (attack)` → `fuse W_r` → `adaptive gate g=σ(W_g[h₀;r])` → `z = h₀ + g·r` → `Linear → {NON-HATE, HATE}`.
> Ghi chú thêm loss `L = L_CE + α·L_CL` ở khối classifier.

### 4.1 Text Preprocessing

- Chuẩn hoá nhiễu: lowercase, bỏ URL/ký hiệu phi ngôn ngữ, gộp ký tự lặp, chuẩn hoá teencode, ánh xạ emoji → nhãn ngữ dụng thô (chế nhạo, giận dữ, ghê tởm, cười, nhấn mạnh).
- **Word segmentation** (bắt buộc vì PhoBERT huấn luyện trên văn bản đã tách từ), sau đó tokenize bằng PhoBERT tokenizer, cắt/đệm về độ dài cố định ($\text{max\_len}=256$).

### 4.2 Multi-Signal Target and Attack Extraction

- **Target signal**: Vietnamese NER (person/org/loc/GPE/nhóm) **hợp** với **target cue bank** (đại từ, danh từ chỉ nhóm, cách gọi suồng sã). Cue *không* tự nó là chỉ dấu hate — chỉ đánh dấu **vị trí target khả dĩ**.
- **Attack signal**: **attack cue bank** (vị từ công kích, lăng mạ, đe doạ, đánh giá tiêu cực — vd `khinh`, `ăn_bám`).
- **Tách biệt target vs attack**: target = *ai/cái gì được nói tới*; attack = *có bị đánh giá thù địch không*. Tránh coi mọi từ tục là target và mọi target là attack.
- Khớp cue: chuẩn hoá → tách từ → tokenize → **full token-sequence matching** (tránh khớp nhầm subtoken).
- **Fallback**: nếu một tập rỗng ⇒ dùng vị trí `[CLS]` (giữ biểu diễn ngầm cấp câu).

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{nếu}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{nếu}\ A_x=\varnothing
$$

### 4.3 PhoBERT Encoder

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768
$$
trong đó $h_0$ là biểu diễn `[CLS]` (ngữ cảnh toàn câu); các vị trí target/attack cung cấp bằng chứng cục bộ.

### 4.4 Relation-Bank Attention

Ba kênh quan hệ:
$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$
với mỗi HeadAttention trên ma trận token $E \in \mathbb{R}^{m\times d}$:
$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V
$$
Hợp nhất:
$$
r = W_r\,[\,r_{\text{exp}}\,;\,r_{\text{imp}}\,;\,r_{\text{atk}}\,] + b_r
$$

### 4.5 Corrected Batched Attention

AmpleHate gốc tính $QK^\top$ ở **cấp batch** ($Q,K\in\mathbb{R}^{B\times d}\Rightarrow QK^\top\in\mathbb{R}^{B\times B}$), vô tình **trộn thông tin giữa các mẫu**. ViAmpleHate dùng **batched attention** để mỗi mẫu chỉ attend vào cue của chính nó:
$$
Q\in\mathbb{R}^{B\times 1\times d},\ K\in\mathbb{R}^{B\times m\times d},\quad
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m}
$$
Vị trí padding bị **mask** ($\text{scores}_j = -\infty$ nếu $\text{mask}_j=0$) trước softmax.

### 4.6 Instance-Adaptive Relation Gate

Thay hệ số inject cố định $e$ của AmpleHate bằng cổng thích ứng theo từng mẫu:
$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r
$$
$z$ qua dropout + linear → logits cho {NON-HATE, HATE}. Cue mạnh ⇒ $g$ lớn (tăng đóng góp quan hệ); cue yếu/mơ hồ ⇒ dựa nhiều hơn vào `[CLS]`.

### 4.7 Training Objective

$$
L = L_{\text{CE}} + \alpha\, L_{\text{CL}}, \qquad \alpha = 0.1
$$
- **Weighted cross-entropy** $L_{\text{CE}} = -\sum_c w_c\,y_c\log \hat y_c$ với trọng số lớp $w_c$ (xử lý mất cân bằng), kèm label smoothing.
- **Contrastive loss** trên biểu diễn sau gate $z$ (cosine $s_{ij}=\cos(z_i,z_j)$):
$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-\text{margin})\Big]
$$
- **Chọn ngưỡng** cho lớp HATE trên tập validation theo macro-F1:
$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big)
$$

---

## 5. Experiments

*(~1,5 trang)*

### 5.1 Baselines

So sánh ViAmpleHate với 5 baseline (cùng thiết lập binary):
- **TF-IDF + Logistic Regression / SVM** — đặc trưng từ vựng thưa.
- **BiLSTM + fastText (vi)** — embedding tĩnh + mô hình chuỗi.
- **PhoBERT-CNN** — PhoBERT + trích đặc trưng cục bộ bằng CNN.
- **AmpleHate-PhoBERT (baseline)** — bản port trung thực của AmpleHate gốc: NER tiếng Anh, một HeadAttention, inject cố định $z=h_0+e\,r_{\text{base}}$ ($e=1.0$). Đây là **đối thủ trực tiếp** vì cùng encoder PhoBERT.

### 5.2 Baseline vs Proposed — các thay đổi cụ thể

**Bảng 2 — Đối chiếu kiến trúc/cấu hình (Baseline AmpleHate-PhoBERT vs ViAmpleHate-PhoBERT).** *(LaTeX `table`, có thể để `\small`)*

| Thành phần | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (đề xuất) |
|---|---|---|
| Encoder | PhoBERT-base | PhoBERT-base |
| Trích target | NER tiếng Anh (`dbmdz/bert-large-...-conll03-english`) | NER tiếng Việt (`NlpHUST/ner-vietnamese-electra-base`) + target cue bank |
| Độ phủ target | ~21/24.048 ≈ 0,09% mẫu train | ~18,8–20,0% (cue tiếng Việt) |
| Tín hiệu attack | Không mô hình hoá | Attack cue bank riêng |
| Attention | 1 HeadAttention | 3 kênh: target / implicit / attack |
| Tính attention | `matmul(Q,Kᵀ)` cấp batch (trộn mẫu) | `bmm` + mask theo từng mẫu |
| Hợp nhất | Inject cố định $h_0+e\,r$ ($e{=}1.0$) | Relation bank + **adaptive gate** $h_0+g\,r$ |
| Loss | Weighted CE | Weighted CE + Contrastive ($\alpha{=}0.1$) |
| max_len | 128 | 256 |
| Batch | 16 | 16 × grad-accum 2 (eff. 32) |
| NER khi eval | Tắt (⇒ fallback `[CLS]`) | Bật, nhất quán train/val/test (`USE_NER_AT_EVAL=True`) |

### 5.3 Implementation Details

**Bảng 3 — Siêu tham số.** *(LaTeX `table`)*

| Tham số | Giá trị |
|---|---|
| Encoder | `vinai/phobert-base` (768-d) |
| Vietnamese NER | `NlpHUST/ner-vietnamese-electra-base` |
| max_len | 256 |
| LR (encoder / head) | 2e-5 / 5e-5 |
| Dropout | 0.1 |
| Effective batch | 32 (16 × grad-accum 2) |
| Epochs | tối đa 8 (chọn theo val macro-F1) |
| α (contrastive) | 0.1 |
| Chọn checkpoint & ngưỡng | theo macro-F1 trên validation |

### 5.4 Evaluation Metrics

- **Macro-F1** và **HATE-class F1** là metric chính (phản ánh đúng lớp thiểu số hơn accuracy).
- **Accuracy** chỉ tham khảo (dễ cao giả tạo do mất cân bằng).
- Chọn ngưỡng $t^{*}$ theo macro-F1 trên validation, áp dụng cố định cho test.

### 5.5 Experimental Purpose

Mỗi thay đổi nhắm một hạn chế cụ thể của baseline: NER+cue tiếng Việt ↔ độ phủ target thấp; attack cue ↔ thiếu mô hình hoá thù địch; relation-bank ↔ cần tách target/context/attack; batched attention ↔ rò rỉ chéo mẫu; adaptive gate ↔ inject cố định; contrastive ↔ tách lớp dưới mất cân bằng.

---

## 6. Result Analysis / Error Analysis

### 6.1 Experimental Results

**Bảng 4 — Kết quả trên ViHSD (test).** *(LaTeX `table` + `booktabs`; **in đậm** số tốt nhất mỗi cột)*

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.8910 | 0.7393 | 0.5404 |
| TF-IDF + SVM | 0.9126 | 0.7131 | 0.4739 |
| BiLSTM + fastText | – | 0.7072 | 0.5060 |
| PhoBERT-CNN | – | 0.7571 | 0.5745 |
| AmpleHate-PhoBERT (baseline) | 0.9175 | 0.7792 | 0.6045 |
| **ViAmpleHate-PhoBERT (ours)** | **0.9205** | **0.7819** | **0.6081** |
| *Δ so với baseline* | *+0.0030* | *+0.0027* | *+0.0036* |

**Bảng 5 — Kết quả trên VOZ-HSD (test).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.9453 | 0.7745 | 0.5783 |
| TF-IDF + SVM | 0.9641 | 0.7831 | 0.5850 |
| BiLSTM + fastText | – | 0.6712 | 0.4187 |
| PhoBERT-CNN | – | 0.8150 | 0.6500 |
| AmpleHate-PhoBERT (baseline) | 0.9643 | 0.8185 | 0.6557 |
| **ViAmpleHate-PhoBERT (ours)** | 0.9420 | **0.8371** | **0.7065** |
| *Δ so với baseline* | *–0.0223* | *+0.0186* | *+0.0508* |

**Diễn giải (viết thành văn):**
- **Transformer > static-embedding > lexical.** PhoBERT-based vượt BiLSTM/TF-IDF nhờ biểu diễn ngữ cảnh tiếng Việt từ tiền huấn luyện quy mô lớn.
- **Baseline AmpleHate bị giới hạn** vì NER tiếng Anh hiếm khi tìm được target tiếng Việt ⇒ thường rơi về `[CLS]`, gần như PhoBERT thuần.
- **ViAmpleHate cải thiện đúng metric quan trọng**: macro-F1 và HATE-F1 tăng trên cả hai dataset; mức tăng **rõ rệt hơn trên VOZ-HSD** (HATE-F1 **+0.0508**). Trên ViHSD mức tăng nhỏ nhưng nhất quán; HATE-precision tăng (0.5972 → 0.6177) còn HATE-recall giảm nhẹ (0.6119 → 0.5988) ⇒ mô hình **thận trọng hơn nhưng chính xác hơn** khi gán HATE.
- **Accuracy giảm trên VOZ** trong khi macro/HATE-F1 tăng: minh hoạ vì sao **không nên dùng accuracy** làm metric chính cho dữ liệu mất cân bằng.
- **Phụ thuộc độ phủ cue**: lợi ích của ViAmpleHate tỉ lệ với chất lượng/độ phủ cue; cue ít ⇒ dựa nhiều vào `[CLS]` ⇒ thu hẹp khoảng cách với baseline.

> 📌 **[CHÈN HÌNH] Figure 3 — Đường cong huấn luyện.** Đặt trong §6.1. File có sẵn:
> `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png`
> (tuỳ chọn ghép cạnh baseline `.../baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/training_curves_amplehate.png`). Cột đơn. Caption: loss/F1 theo epoch, đánh dấu best epoch = 4 (val F1 = 0.7852).

> 📌 **[CHÈN HÌNH] Figure 4 — Confusion matrix (ViHSD): Baseline vs ViAmpleHate.** **Đặt cạnh nhau (side-by-side, `figure*` 2 cột)** trong §6.1 hoặc đầu §6.2. Files có sẵn:
> Baseline: `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/confusion_matrix_amplehate.png`
> Proposed: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate.png`
> Caption: nhấn việc giảm false-positive HATE (precision tăng).

> 📌 **[CHÈN HÌNH — tuỳ chọn] Figure 5 — Confusion matrix (VOZ-HSD) proposed.** File:
> `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate_vozhsd.png`. Dùng nếu còn chỗ; minh hoạ mức cải thiện HATE lớn hơn trên VOZ.

**Bảng 6 (tuỳ chọn) — Per-class P/R/F1 trên ViHSD (proposed).**

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON-HATE | 0.9541 | 0.9574 | – |
| HATE | 0.6177 | 0.5988 | 0.6081 |

### 6.2 Error Analysis

Phân loại lỗi còn lại thành các nhóm (viết thành văn, mỗi nhóm 2–3 câu):

1. **Offensive nhưng không hate.** Bình luận tục tĩu/công kích cá nhân nhưng không nhắm vào nhóm/đối tượng được bảo vệ ⇒ nếu dựa quá mạnh vào attack cue ⇒ **false positive**.
2. **Hate ẩn (implicit).** Mỉa mai, ám chỉ, định kiến, so sánh gián tiếp — không có slur/attack token rõ ⇒ cue-based khó bắt, phải dựa `[CLS]`.
3. **Target cue mơ hồ.** Đại từ/danh từ chỉ nhóm xuất hiện cả trong câu trung tính/hài hước ⇒ phát hiện target không kèm ngữ cảnh thù địch ⇒ **over-predict HATE**.
4. **Độ phủ cue chưa đủ.** Tiếng lóng/biến thể chính tả/viết tắt/profanity sáng tạo thay đổi nhanh; cue bank cố định không phủ hết ⇒ **false negative**.
5. **Lỗi tokenization & span-alignment.** NER, word segmentation, và subword của PhoBERT không luôn khớp span ⇒ cue khớp sai vị trí ⇒ attention attend vào bằng chứng thiếu/sai.
6. **Nhạy ngưỡng (threshold sensitivity).** $t^{*}$ tối ưu trên validation có thể không tối ưu khi phân bố test/domain đổi ⇒ ảnh hưởng triển khai.
7. **Mất cân bằng lớp.** HATE thiểu số ⇒ ít mẫu dương khi train; weighted loss + threshold giảm chứ không loại bỏ lỗi lớp thiểu số.

**Hướng cải thiện (dẫn vào §7.2):** mở rộng cue bank bằng khai phá dữ liệu + thẩm định thủ công; lưu **log dự đoán theo từng mẫu** để phân tích FP/FN có hệ thống (theo độ phủ cue, giá trị gate, độ tự tin); bổ sung giám sát span target, phát hiện mỉa mai, ngữ cảnh xã hội; **hiệu chỉnh xác suất (calibration)** để ngưỡng ổn định hơn.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

- ViAmpleHate **tổng quát hoá AmpleHate cho tiếng Việt** qua 5 thích ứng: trích target tiếng Việt, tách target/attack cue, relation-bank attention, sửa attention theo batch, và adaptive gate; huấn luyện với CE + contrastive.
- Trên ViHSD và VOZ-HSD (binary), mô hình **cải thiện macro-F1 và HATE-F1** so với baseline AmpleHate và các baseline TF-IDF/BiLSTM/PhoBERT-CNN.
- Thông điệp chính: **target-awareness có ích, nhưng phải được thích ứng theo đặc trưng ngôn ngữ tiếng Việt** (target thường là cách gọi nhóm suồng sã, không phải named entity).

### 7.2 Future Work

- Mở rộng & tự động khai phá target/attack cue bank; thẩm định thủ công.
- Phân tích lỗi theo từng mẫu (gate value, cue coverage, confidence); bổ sung giám sát span, phát hiện mỉa mai, ngữ cảnh người dùng/diễn ngôn.
- Mở rộng sang thiết lập đa nhãn/đa target (liên hệ ViTHSD) thay vì chỉ binary.
- Probability calibration để ngưỡng ổn định khi triển khai cross-domain.

---

## References

> *(Cần kiểm tra & hoàn thiện thông tin thư mục đầy đủ trước khi nộp — dùng `\bibliography{}` với `acl_natbib` hoặc nhập `.bib`. Danh sách tối thiểu cần có:)*

- **AmpleHate** — "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection" *(điền tác giả/venue/năm)*.
- **ViTHSD** — "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts" *(điền chi tiết)*.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts." *(IEA/AIE — kiểm tra)*.
- **VOZ-HSD** — bộ dữ liệu `tarudesu/VOZ-HSD` (HuggingFace) *(điền trích dẫn/URL)*.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*.
- **BERT** — Devlin et al. (2019). *NAACL*.
- **fastText** — Bojanowski et al. (2017). *TACL*.
- **Contrastive / SupCon** — Khosla et al. (2020). *NeurIPS* *(nếu dùng supervised contrastive)*.
- **Vietnamese NER** — `NlpHUST/ner-vietnamese-electra-base` *(điền trích dẫn/URL)*.
- *(Tuỳ chọn)* VnCoreNLP / RDRSegmenter cho word segmentation; các khảo sát hate speech detection.

---

### Phụ lục: Checklist chuyển sang LaTeX ACL

- [ ] Tải template ACL (`acl_latex.tex`, `acl.sty`, `acl_natbib.bst`) từ Overleaf.
- [ ] Đổi mỗi `##`/`###` → `\section`/`\subsection`; bảng Markdown → `tabular`+`booktabs`.
- [ ] Công thức: copy phần `$...$`/`$$...$$` vào `equation`/`align`.
- [ ] Hình: tạo Figure 1 (phân bố lớp) & Figure 2 (kiến trúc — **vẽ mới**); Figure 3–5 dùng PNG có sẵn trong `notebooks/.../output/`.
- [ ] In đậm số tốt nhất trong Bảng 4–5; thêm dòng Δ.
- [ ] Viết Abstract (~180 từ) sau cùng; thêm `\section*{Limitations}` (cue bank cố định, threshold sensitivity, hate ẩn) — ACL yêu cầu mục Limitations.
- [ ] Kiểm tra references đầy đủ, dùng `\citep`/`\citet`.
- [ ] Đảm bảo ≤ 8 trang nội dung (ACL) + references/appendix không tính trang.
</content>
</invoke>
