# ViAmpleHate: Phương pháp dựa trên AmpleHate và PhoBERT cho Phát hiện Phát ngôn Thù ghét tiếng Việt (Bản mở rộng v3)

> Bản mở rộng (~10 trang). So với v1/v2, bản này bổ sung: phần Nghiên cứu liên quan rộng hơn (5 tiểu mục), một tiểu mục về xây dựng cue bank, phân tích độ phủ đối tượng, một bảng ablation (để bạn điền số từ thí nghiệm), phân tích định tính kèm ví dụ minh hoạ, một mục Thảo luận riêng, một mục Tuyên bố Đạo đức, và các phụ lục mở rộng. Vị trí cần chèn hình đánh dấu bằng **📌 [CHÈN HÌNH]**; các số liệu bạn còn phải chạy được đánh dấu **[TODO]**.

## Tóm tắt

Phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt là bài toán khó vì ý đồ thù ghét hiếm khi được biểu đạt qua thực thể có tên rõ ràng; thay vào đó nó ẩn trong cách gọi nhóm suồng sã, tiếng lóng, mỉa mai và ám chỉ gián tiếp. Những mô hình hướng-đối-tượng như AmpleHate đã cho thấy việc chú ý đến quan hệ giữa câu nói và đối tượng bị nhắm tới giúp phát hiện cả thù ghét ẩn, nhưng quy trình gốc của AmpleHate phụ thuộc vào nhận dạng thực thể có tên (NER) tiếng Anh và các phạm trù đối tượng kiểu tiếng Anh, nên chuyển sang tiếng Việt rất kém: trên dữ liệu huấn luyện của chúng tôi, NER tiếng Anh chỉ tìm được đối tượng dùng được ở khoảng 0,09% bình luận, khiến mô hình thoái hoá thành một bộ phân loại câu thuần. Chúng tôi đề xuất **ViAmpleHate**, một bản thích ứng của AmpleHate cho tiếng Việt trên nền PhoBERT. ViAmpleHate (i) thay NER tiếng Anh bằng NER tiếng Việt cùng một ngân hàng *cue đối tượng* được xây thủ công, nâng độ phủ đối tượng lên khoảng 18,8–20,0%; (ii) bổ sung một ngân hàng *cue tấn công* riêng và mô hình hoá ba kênh quan hệ — đối tượng tường minh, ngữ cảnh ngầm, và tấn công — qua một cơ chế *relation-bank attention*; (iii) sửa một lỗi tính attention theo lô làm rò rỉ thông tin giữa các mẫu; và (iv) thay cơ chế bơm bằng hệ số cố định bằng một *cổng thích ứng theo từng mẫu*, huấn luyện với cross-entropy có trọng số kết hợp hàm tương phản. Trên ViHSD và VOZ-HSD ở thiết lập nhị phân NON-HATE/HATE, ViAmpleHate cải thiện macro-F1 và F1 của lớp thiểu số HATE so với baseline AmpleHate trung thực cũng như các baseline TF-IDF, BiLSTM và PhoBERT-CNN, với mức tăng lớn nhất ở lớp thiểu số (HATE-F1 +0,0508 trên VOZ-HSD). Phân tích cho thấy lợi ích tăng theo độ phủ cue, khẳng định rằng mô hình hoá hướng-đối-tượng phải được thích ứng theo ngôn ngữ tiếng Việt chứ không thể bê nguyên từ tiếng Anh.

## 1. Giới thiệu

Sự phát triển nhanh của mạng xã hội tiếng Việt đi kèm với sự lan rộng tương ứng của nội dung thù ghét và lăng mạ. Loại nội dung này gây hại cho cá nhân và cộng đồng bị nhắm tới, làm xấu môi trường thảo luận trực tuyến, và tạo rủi ro pháp lý lẫn uy tín cho các nền tảng. Vì khối lượng nội dung do người dùng tạo ra vượt xa khả năng kiểm duyệt thủ công, phát hiện thù ghét tự động trở thành một nhu cầu thực tế. Tuy nhiên, phần lớn tiến bộ tập trung vào tiếng Anh và các ngôn ngữ giàu tài nguyên, và những phương pháp hiệu quả ở đó thường suy giảm khi áp dụng cho tiếng Việt.

Phát hiện thù ghét tiếng Việt khó vì ba lý do đan xen. Thứ nhất, ngôn ngữ mạng xã hội rất nhiễu: người dùng viết tắt, dùng teencode, kéo dài ký tự để nhấn mạnh, chèn emoji mang nghĩa ngữ dụng, và viết sai chính tả — cố ý hoặc không — kể cả nguỵ trang từ tục để né bộ lọc. Thứ hai, và quan trọng nhất với công trình này, *đối tượng* bị tấn công hiếm khi là thực thể có tên. Trong khi hate speech tiếng Anh thường nêu tên một người, tổ chức hay quốc tịch, sự thù địch trong tiếng Việt thường nhắm vào một nhóm qua đại từ, danh từ thân tộc hay chỉ nhóm, nhãn vùng miền, hoặc cách gọi suồng sã. Thứ ba, dữ liệu mất cân bằng nghiêm trọng: trong cả hai bộ dữ liệu chúng tôi nghiên cứu, lớp HATE chỉ chiếm khoảng một phần mười số bình luận, nên một mô hình có thể đạt accuracy cao mà gần như không phát hiện được hate.

Một hướng tiếp cận hứa hẹn là mô hình hoá hate speech theo *hướng-đối-tượng*. AmpleHate cho thấy việc chú ý tường minh đến quan hệ giữa câu và đối tượng tiềm năng cải thiện việc phát hiện thù ghét *ẩn*, nơi sự thù địch được truyền tải mà không có slur công khai. AmpleHate trích đối tượng bằng NER, tính một vector attention hướng-đối-tượng, rồi bơm nó vào biểu diễn câu trước khi phân loại. Cách này hấp dẫn với tiếng Việt vì đối tượng rất trung tâm — nhưng quy trình gốc xây quanh NER và phạm trù đối tượng tiếng Anh. Khi chạy nó trên tiếng Việt với NER tiếng Anh, nó chỉ tìm được đối tượng dùng được ở 21 trong 24.048 bình luận huấn luyện (khoảng 0,09%). Với đại đa số đầu vào, mô hình rơi về biểu diễn `[CLS]` toàn cục và hành xử như một bộ phân loại PhoBERT thông thường, vứt bỏ đúng tín hiệu hướng-đối-tượng vốn là động lực của nó.

Chúng tôi đề xuất **ViAmpleHate**, một bản thích ứng của AmpleHate cho tiếng Việt, tái thiết kế từng khâu của pipeline quanh cách người Việt thực sự biểu đạt sự thù ghét. Đóng góp của chúng tôi gồm:

1. **Trích đối tượng theo tiếng Việt.** Thay NER tiếng Anh bằng mô hình NER tiếng Việt và bổ sung một ngân hàng *cue đối tượng* gồm đại từ, danh từ chỉ nhóm và cách gọi suồng sã, nâng tỉ lệ bình luận có đối tượng được phát hiện từ ~0,09% lên ~18,8–20,0%.
2. **Tách kênh đối tượng và tấn công.** Bổ sung một ngân hàng *cue tấn công* cho các vị từ thù địch và mô hình hoá cue đối tượng và cue tấn công như hai tín hiệu riêng qua một cơ chế **relation-bank attention** ba kênh.
3. **Sửa lỗi cài đặt và cơ chế hợp nhất.** Sửa cách tính attention theo lô làm trộn thông tin giữa các mẫu, và thay cơ chế bơm bằng hệ số cố định của AmpleHate bằng một **cổng thích ứng theo từng mẫu**.
4. **Mục tiêu huấn luyện.** Kết hợp cross-entropy có trọng số với một hàm tương phản giúp tách rõ HATE/NON-HATE dưới điều kiện mất cân bằng.
5. **Nghiên cứu thực nghiệm.** Đánh giá trên ViHSD và VOZ-HSD với năm baseline và phân tích lợi ích đến từ đâu, gồm phân tích độ phủ đối tượng và nghiên cứu định tính các lỗi còn lại.

Thông điệp lớn hơn là: tính hướng-đối-tượng là một thiên kiến quy nạp thực sự hữu ích cho phát hiện thù ghét, nhưng chỉ khi nó được thiết kế phù hợp với bề mặt ngôn ngữ của ngôn ngữ đích, thay vì bê nguyên từ tiếng Anh.

## 2. Các nghiên cứu liên quan

### 2.1 Phát hiện thù ghét và thù ghét ẩn

Phát hiện thù ghét tự động đã được nghiên cứu rộng rãi, tiến từ các bộ phân loại dựa trên từ điển và đặc trưng sang mô hình chuỗi sâu và gần đây là các transformer tiền huấn luyện. Một khó khăn dai dẳng là thù ghét *ẩn*: thông điệp truyền tải sự thù địch mà không dùng slur công khai, dựa vào mỉa mai, định kiến, ám chỉ mã hoá, hoặc kiến thức nội nhóm. Thù ghét ẩn thường phổ biến hơn slur công khai và khó hơn nhiều cho các hệ thống dựa từ khoá, thúc đẩy các hướng tiếp cận biết suy luận *ai* bị nhắm tới và *bằng cách nào*.

### 2.2 AmpleHate

AmpleHate xử lý thù ghét ẩn bằng cách khuếch đại attention giữa biểu diễn toàn cục của câu và các đối tượng tiềm năng. Nó trích token đối tượng bằng NER, tính một vector HeadAttention với truy vấn từ `[CLS]` và khoá/giá trị từ token đối tượng, rồi bơm kết quả với một hệ số cố định trước khi phân loại. Ý tưởng cốt lõi — đối tượng là tín hiệu chịu lực — là động lực trực tiếp cho công trình của chúng tôi. Hạn chế của nó với tiếng Việt cũng trực tiếp không kém: giả định NER và phạm trù đối tượng kiểu tiếng Anh, bơm thông tin với cường độ cố định cho mọi mẫu, và chỉ mô hình hoá một quan hệ (đối tượng) trong khi bỏ qua vị từ thù địch, tức *tấn công*, vốn biến một cách nhắc tên thành thù ghét.

### 2.3 Phát hiện thù ghét tiếng Việt

Tiếng Việt ngày càng có nhiều tài nguyên và mô hình cho hate speech. ViHSD cung cấp một bộ dữ liệu ba lớp (CLEAN/OFFENSIVE/HATE) quy mô lớn và là benchmark tiêu chuẩn. ViTHSD đặt lại bài toán quanh *đối tượng*, gán mức độ thù ghét cho từng đối tượng trong bình luận; điều này củng cố vai trò trung tâm của đối tượng trong hate speech tiếng Việt và ủng hộ thiết kế hướng-đối-tượng của chúng tôi. Khác ViTHSD, ViAmpleHate không cần nhãn đối tượng ở mức span; nó định vị đối tượng và tấn công bằng NER cộng cue bank, nên vẫn áp dụng được cho dữ liệu chỉ có nhãn mức câu.

### 2.4 PhoBERT và các mô hình tiền huấn luyện tiếng Việt

PhoBERT là một mô hình kiểu RoBERTa được tiền huấn luyện trên kho ngữ liệu tiếng Việt lớn và là bộ mã hoá mặc định cho phân loại văn bản tiếng Việt. Vì PhoBERT được huấn luyện trên văn bản đã *tách từ*, bước tách từ phải đi trước token hoá. Trên nền PhoBERT, các kiến trúc lai như PhoBERT-CNN thêm bộ trích đặc trưng cục bộ lên trên embedding ngữ cảnh. Chúng tôi dùng PhoBERT-base làm bộ mã hoá chung cho cả baseline lẫn mô hình đề xuất để mọi khác biệt đều quy về cách mô hình hoá hướng-đối-tượng chứ không phải backbone.

### 2.5 Học tương phản cho phân loại văn bản

Các mục tiêu tương phản khuyến khích biểu diễn của các mẫu cùng lớp gần nhau và các mẫu khác lớp xa nhau, cải thiện độ bền và khả năng tách lớp, đặc biệt khi mất cân bằng. Chúng tôi dùng một hạng tương phản có giám sát trên biểu diễn sau cổng như một mục tiêu phụ bên cạnh cross-entropy có trọng số, nhằm siết chặt ranh giới giữa lớp NON-HATE thường gặp và lớp HATE hiếm.

## 3. Dữ liệu

### 3.1 ViHSD

ViHSD là bộ dữ liệu hate speech tiếng Việt gồm bình luận mạng xã hội gán ba nhãn gốc (CLEAN, OFFENSIVE, HATE), chia sẵn train/validation/test. Sau khi gán nhị phân (Mục 3.3), phân bố từng tập ở Bảng 1.

### 3.2 VOZ-HSD

VOZ-HSD, được phát hành kèm bài báo ViHateT5, gồm bình luận từ diễn đàn VOZ, nguồn gốc từ `tarudesu/VOZ-HSD` trên HuggingFace. Cũng như ViHSD, dữ liệu được chia train/validation/test, và phân bố sau gán nhãn ở Bảng 1.

### 3.3 Gán lại nhãn nhị phân và tiền xử lý

Thao tác duy nhất chúng tôi thực hiện là *gán lại nhãn*; không thay đổi hay chọn lọc số lượng mẫu. Chúng tôi đưa bài toán về nhị phân bằng cách gộp CLEAN và OFFENSIVE thành NON-HATE và giữ HATE làm lớp dương. Điều này tập trung bài toán vào thù ghét nhắm vào nhóm, thay vì thô tục hay công kích nói chung, và tạo không gian nhãn nhất quán giữa hai bộ dữ liệu. Mỗi bình luận sau đó được chuẩn hoá (chuyển chữ thường; bỏ URL/ký hiệu; gộp ký tự kéo dài; chuẩn hoá teencode; ánh xạ emoji thường gặp sang nhãn ngữ dụng thô như chế nhạo, giận dữ, ghê tởm, cười), tách từ — bắt buộc vì PhoBERT huấn luyện trên văn bản đã tách từ — rồi token hoá bằng tokenizer của PhoBERT.

**Bảng 1 — Thống kê dữ liệu sau khi gán nhị phân.**

| Bộ dữ liệu | Tập | NON-HATE | HATE | Tổng | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21.492 | 2.556 | 24.048 | 10,6% |
| ViHSD | Dev | 2.402 | 270 | 2.672 | 10,1% |
| ViHSD | Test | 5.992 | 688 | 6.680 | 10,3% |
| VOZ-HSD | Train | 26.993 | 3.007 | 30.000 | 10,0% |
| VOZ-HSD | Dev | 4.520 | 480 | 5.000 | 9,6% |
| VOZ-HSD | Test | 4.487 | 513 | 5.000 | 10,3% |

> 📌 **[CHÈN HÌNH] Hình 1 — Phân bố lớp.** Biểu đồ cột NON-HATE vs HATE trên cả hai bộ (số từ Bảng 1). **Vẽ mới** (matplotlib). Chú thích nhấn tỉ lệ HATE ~10% và việc chọn macro-F1/HATE-F1 thay cho accuracy.

### 3.4 Xây dựng cue bank

Ngân hàng cue đối tượng và cue tấn công là trung tâm của ViAmpleHate nên chúng tôi mô tả cách xây. Cả hai ngân hàng được gieo mầm thủ công từ việc khảo sát tập huấn luyện và từ các biểu thức quy chiếu/xúc phạm thường gặp trong tiếng Việt, sau đó chuẩn hoá khớp với pipeline tiền xử lý. **Ngân hàng cue đối tượng** chứa các biểu thức quy chiếu thường giới thiệu một người hay nhóm — đại từ, danh từ thân tộc và chỉ nhóm, nhãn vùng miền và nhân khẩu, cách gọi suồng sã — nhưng *bản thân không* là chỉ dấu thù ghét. **Ngân hàng cue tấn công** chứa các vị từ thù địch: lăng mạ, từ phi nhân hoá, đe doạ, và đánh giá rất tiêu cực. Giữ hai ngân hàng tách biệt là có chủ đích: một cách nhắc đối tượng mà không có vị từ thù địch thường không phải thù ghét, và một vị từ thù địch không kèm đối tượng nhóm rõ ràng thường chỉ là thô tục. Mỗi cue được chuẩn hoá, tách từ, token hoá, rồi khớp với chuỗi token bằng khớp toàn chuỗi token để một cụm tiếng Việt nhiều token được khớp như một đơn vị thay vì qua một subtoken lạc. Phụ lục B liệt kê các mục tiêu biểu.

## 4. Phương pháp

### 4.1 Phát biểu bài toán

Cho bình luận tiếng Việt `x`, mô hình dự đoán `y ∈ {NON-HATE, HATE}`. ViAmpleHate kế thừa ý tưởng hướng-đối-tượng của AmpleHate và thích ứng qua năm thay đổi: trích đối tượng tiếng Việt, tách kênh cue đối tượng/tấn công, relation-bank attention, sửa cơ chế attention theo lô, và bơm bằng cổng thích ứng (Hình 2).

> 📌 **[CHÈN HÌNH] Hình 2 — Kiến trúc tổng thể của ViAmpleHate.** Hình quan trọng nhất; **vẽ mới** (draw.io/PowerPoint), full-width. Thể hiện: bình luận → chuẩn hoá → tách từ → PhoBERT → `[CLS]` `h₀` và trạng thái token `H`; nhánh A (NER tiếng Việt + cue đối tượng → `T_x`) và nhánh B (cue tấn công → `A_x`); ba kênh relation-bank attention `r_exp / r_imp / r_atk` → hợp nhất qua `W_r`; cổng thích ứng `g = σ(W_g[h₀;r])` → `z = h₀ + g·r` → bộ phân loại tuyến tính → {NON-HATE, HATE}. Ghi chú hàm mất mát `L = L_CE + α·L_CL`.

### 4.2 Tiền xử lý văn bản

Như Mục 3.3: chuẩn hoá, tách từ, token hoá bằng tokenizer PhoBERT, cắt/đệm về `max_len = 256`.

### 4.3 Trích tín hiệu đối tượng và tấn công đa nguồn

Tập chỉ số đối tượng kết hợp NER tiếng Việt và khớp cue đối tượng; tập chỉ số tấn công đến từ khớp cue tấn công. Nếu một tập rỗng, mô hình lùi về `[CLS]` để luôn có một biểu diễn câu ngầm:

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{nếu}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{nếu}\ A_x=\varnothing
$$

### 4.4 Bộ mã hoá PhoBERT

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768,
$$

với `h_0` là biểu diễn `[CLS]` nắm ngữ cảnh toàn cục, còn các vị trí đối tượng/tấn công cung cấp bằng chứng cục bộ.

### 4.5 Relation-bank attention

Ba góc nhìn quan hệ — đối tượng tường minh, ngữ cảnh ngầm (từ neo `[CLS]`), và tấn công:

$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$

với mỗi HeadAttention trên ma trận quan hệ `E ∈ R^{m×d}`:

$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V.
$$

Ba vector được nối và chiếu thành biểu diễn quan hệ hợp nhất: $r = W_r[\,r_{\text{exp}};r_{\text{imp}};r_{\text{atk}}\,] + b_r$.

### 4.6 Sửa lỗi attention theo lô

Cài đặt kiểu AmpleHate gốc tính attention tương đương `QKᵀ` trên toàn lô, nên với `Q,K ∈ R^{B×d}` ta được `QKᵀ ∈ R^{B×B}`, vô tình trộn thông tin giữa các mẫu. ViAmpleHate dùng attention theo lô để mỗi mẫu chỉ attend vào token của chính nó, với `Q ∈ R^{B×1×d}`, `K ∈ R^{B×m×d}`:

$$
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m},
$$

và các vị trí đệm bị mask trước softmax (`scores_j = -∞` nếu `mask_j = 0`).

### 4.7 Cổng quan hệ thích ứng theo từng mẫu

Thay hệ số cố định bằng cổng theo từng mẫu:

$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r.
$$

Khi bằng chứng đối tượng/tấn công rõ, mô hình tăng `g` và dựa nhiều hơn vào vector quan hệ; khi cue yếu hoặc vắng, nó giảm `g` và dựa vào biểu diễn toàn cục. Biểu diễn hợp nhất `z` qua dropout và một lớp tuyến tính để cho ra logits.

### 4.8 Mục tiêu huấn luyện

Huấn luyện với `L = L_CE + α·L_CL`, `α = 0,1`. Cross-entropy có trọng số `L_CE = -Σ_c w_c y_c log ŷ_c` xử lý mất cân bằng bằng cách tăng trọng số lớp HATE, kèm label smoothing. Hạng tương phản trên `z` (với `s_ij = cos(z_i, z_j)`) kéo các cặp cùng lớp lại gần và đẩy các cặp khác lớp ra xa:

$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-m)\Big].
$$

### 4.9 Suy luận và chọn ngưỡng

Khi suy luận, ta tính xác suất HATE và áp một ngưỡng quyết định chọn trên validation bằng cách tối đa macro-F1, rồi cố định cho test:

$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big).
$$

### 4.10 Chi phí tính toán

Các thành phần bổ sung nhẹ so với bộ mã hoá PhoBERT. Ba mô-đun HeadAttention và cổng làm việc trên truy vấn `[CLS]` đã gộp và một số ít token cue mỗi mẫu, nên chi phí thêm chủ yếu là vài phép chiếu tuyến tính; attention theo lô đã sửa loại bỏ tương tác `B×B` thừa và thực ra rẻ hơn khi huấn luyện. Việc khớp cue được thực hiện một lần trong tiền xử lý.

## 5. Thực nghiệm

### 5.1 Các baseline

So sánh với năm baseline, đều ở thiết lập nhị phân. **TF-IDF + LR** và **TF-IDF + SVM** dùng đặc trưng từ vựng thưa. **BiLSTM + fastText** kết hợp embedding tĩnh với mô hình hồi tiếp. **PhoBERT-CNN** xếp bộ trích đặc trưng CNN lên embedding PhoBERT. **AmpleHate-PhoBERT** là bản port trung thực của AmpleHate gốc với NER tiếng Anh, một HeadAttention, và bơm cố định `z = h_0 + e·r_base` (`e = 1,0`); vì dùng chung bộ mã hoá PhoBERT-base, đây là đối thủ trực tiếp cô lập tác động của các thay đổi mà chúng tôi đề xuất. Bảng 2 tóm tắt khác biệt.

**Bảng 2 — Đối chiếu kiến trúc và cấu hình.**

| Thành phần | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (đề xuất) |
|---|---|---|
| Bộ mã hoá | PhoBERT-base | PhoBERT-base |
| Trích đối tượng | NER tiếng Anh (`dbmdz/bert-large-...-conll03-english`) | NER tiếng Việt (`NlpHUST/ner-vietnamese-electra-base`) + cue đối tượng |
| Độ phủ đối tượng | ~21/24.048 ≈ 0,09% train | ~18,8–20,0% nhờ cue tiếng Việt |
| Tín hiệu tấn công | Không mô hình hoá | Ngân hàng cue tấn công riêng |
| Attention | Một HeadAttention | Ba kênh: đối tượng / ngầm / tấn công |
| Cách tính attention | `matmul` cấp lô (trộn mẫu) | `bmm` + mask theo từng mẫu |
| Hợp nhất | Bơm cố định `h₀ + e·r` (`e = 1,0`) | Relation bank + cổng thích ứng `h₀ + g·r` |
| Hàm mất mát | Weighted CE | Weighted CE + tương phản (`α = 0,1`) |
| Độ dài tối đa | 128 | 256 |
| Lô | 16 | 16 × tích luỹ gradient 2 (hiệu dụng 32) |
| NER khi đánh giá | Tắt (⇒ fallback `[CLS]`) | Bật, nhất quán các tập |

### 5.2 Chi tiết cài đặt

Bộ mã hoá `vinai/phobert-base` (768-d); NER tiếng Việt `NlpHUST/ner-vietnamese-electra-base`. Độ dài tối đa 256. Tốc độ học 2e-5 (mã hoá) / 5e-5 (đầu phân loại); dropout 0,1. Lô hiệu dụng 32 (16 × tích luỹ 2). Huấn luyện tối đa tám epoch, chọn checkpoint tốt nhất theo macro-F1 validation. `α = 0,1`, ngưỡng chọn trên validation theo macro-F1. Siêu tham số đầy đủ ở Phụ lục A.

### 5.3 Độ đo đánh giá

Hai độ đo chính là macro-F1 và HATE-class F1, phản ánh hiệu năng lớp thiểu số tốt hơn accuracy nhiều: với ~10% mẫu dương, một bộ dự đoán đa số tầm thường đã đạt >0,89 accuracy mà không phát hiện được hate nào. Macro-F1 trung bình F1 từng lớp; HATE-F1 cô lập lớp quan tâm.

### 5.4 Thiết kế thực nghiệm

Ngoài điểm số cuối, thực nghiệm được thiết kế để kiểm chứng mỗi thay đổi xử lý một hạn chế cụ thể của baseline: NER và cue tiếng Việt ↔ độ phủ thấp; cue tấn công ↔ thiếu mô hình hoá thù địch; relation-bank ↔ tách đối tượng/ngữ cảnh/tấn công; attention theo lô đã sửa ↔ rò rỉ chéo mẫu; cổng thích ứng ↔ bơm cố định; tương phản ↔ tách lớp. Mục 6.2–6.3 khảo sát trực tiếp các yếu tố này.

## 6. Kết quả và phân tích

### 6.1 Kết quả chính

**Bảng 3 — Kết quả trên ViHSD (test). In đậm số tốt nhất mỗi cột.**

| Mô hình | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0,8910 | 0,7393 | 0,5404 |
| TF-IDF + SVM | 0,9126 | 0,7131 | 0,4739 |
| BiLSTM + fastText | – | 0,7072 | 0,5060 |
| PhoBERT-CNN | – | 0,7571 | 0,5745 |
| AmpleHate (baseline) | 0,9175 | 0,7792 | 0,6045 |
| **ViAmpleHate (đề xuất)** | **0,9205** | **0,7819** | **0,6081** |
| *Δ so với baseline* | *+0,0030* | *+0,0027* | *+0,0036* |

**Bảng 4 — Kết quả trên VOZ-HSD (test). In đậm số tốt nhất mỗi cột.**

| Mô hình | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0,9453 | 0,7745 | 0,5783 |
| TF-IDF + SVM | 0,9641 | 0,7831 | 0,5850 |
| BiLSTM + fastText | – | 0,6712 | 0,4187 |
| PhoBERT-CNN | – | 0,8150 | 0,6500 |
| AmpleHate (baseline) | 0,9643 | 0,8185 | 0,6557 |
| **ViAmpleHate (đề xuất)** | 0,9420 | **0,8371** | **0,7065** |
| *Δ so với baseline* | *–0,0223* | *+0,0186* | *+0,0508* |

Các mô hình transformer vượt rõ các baseline từ vựng và embedding tĩnh ở những độ đo quan trọng, khẳng định biểu diễn ngữ cảnh là cần thiết khi hate phụ thuộc ngữ cảnh, cách diễn đạt suồng sã, và tương tác giữa cách nhắc tên với vị từ thù địch. Trong nhóm transformer, baseline AmpleHate cải thiện so với PhoBERT thuần nhờ attention hướng-đối-tượng, nhưng lợi ích bị chặn vì NER tiếng Anh hiếm khi tìm được đối tượng tiếng Việt hợp lệ nên hầu hết đầu vào rơi về `[CLS]`.

ViAmpleHate cải thiện macro-F1 và HATE-F1 trên cả hai bộ. Mức tăng nhỏ nhưng nhất quán trên ViHSD và lớn hơn hẳn trên VOZ-HSD, nơi HATE-F1 tăng 0,0508. Đáng chú ý, accuracy *giảm* trên VOZ-HSD trong khi macro-F1 và HATE-F1 đều tăng — minh chứng trực tiếp rằng accuracy là độ đo tiêu đề sai dưới mất cân bằng, vì mô hình đang chuyển năng lực từ lớp đa số dễ sang lớp thiểu số khó.

### 6.2 Tác động của trích đối tượng tiếng Việt

Cơ chế chủ đạo đằng sau các cải thiện này là độ phủ đối tượng. Với NER tiếng Anh, chỉ ~0,09% bình luận train có đối tượng được phát hiện, nên nhánh hướng-đối-tượng của baseline gần như không bao giờ kích hoạt. Thay NER tiếng Anh bằng NER tiếng Việt và thêm cue đối tượng nâng độ phủ lên ~18,8–20,0% số bình luận được kiểm — hơn hai bậc độ lớn — chính là thứ cho relation-bank attention có cái để attend. Điều này cũng giải thích khoảng cách giữa hai bộ dữ liệu: nơi cue kích hoạt thường xuyên và chính xác hơn, các kênh quan hệ đóng góp nhiều hơn và mức cải thiện lớn hơn; nơi độ phủ thấp, mô hình dựa vào kênh `[CLS]` ngầm và khoảng cách thu hẹp.

> 📌 **[CHÈN HÌNH] Hình 3 — Đường cong huấn luyện.** Dùng PNG sẵn có `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png` (tuỳ chọn đặt cạnh đường cong baseline). Chú thích: epoch tốt nhất 4, macro-F1 validation = 0,7852.

### 6.3 Nghiên cứu ablation

Để cô lập đóng góp từng thành phần, ta loại bỏ từng cái một khỏi mô hình đầy đủ và huấn luyện lại với cùng thiết lập, báo cáo macro-F1 và HATE-F1 trên test. Bảng 5 là khung; các giá trị cần điền từ thí nghiệm ablation của bạn.

**Bảng 5 — Ablation trên ViHSD (test). [TODO: điền từ thí nghiệm.]**

| Cấu hình | Macro-F1 | HATE-F1 |
|---|---:|---:|
| ViAmpleHate đầy đủ | 0,7819 | 0,6081 |
| − kênh tấn công | [TODO] | [TODO] |
| − cổng thích ứng (bơm cố định) | [TODO] | [TODO] |
| − hàm tương phản (chỉ CE) | [TODO] | [TODO] |
| − cue đối tượng (chỉ NER) | [TODO] | [TODO] |
| − attention theo lô đã sửa | [TODO] | [TODO] |

> 📌 **[CHÈN HÌNH — tuỳ chọn] Hình 4 — Phân bố giá trị cổng.** Histogram giá trị cổng `g` trên test, tách theo lớp dự đoán. **Vẽ mới** nếu bạn log giá trị cổng; minh hoạ rằng các dự đoán HATE tự tin thường dùng `g` lớn hơn.

### 6.4 Phân tích định tính

Để cụ thể hoá các kiểu lỗi, Bảng 6 trình bày các loại bình luận minh hoạ (ví dụ dựng lại phản ánh các nhóm quan sát được; thay bằng mẫu test thật, đã ẩn danh, cho bản nộp). Khuôn mẫu là ViAmpleHate hữu ích nhất khi một đối tượng tường minh đồng xuất hiện với một vị từ tấn công, và ít hữu ích nhất với hate ẩn không nêu đối tượng và không dùng từ tấn công công khai.

**Bảng 6 — Ca minh hoạ (dựng lại; thay bằng ví dụ thật).**

| Loại bình luận | Có cue đối tượng? | Có cue tấn công? | Nhãn vàng | Xu hướng |
|---|---|---|---|---|
| Nhãn nhóm + vị từ thù địch | có | có | HATE | đúng HATE |
| Thô tục, không có đối tượng nhóm | không | có | NON-HATE | rủi ro dương tính giả |
| Mỉa mai/ẩn, không cue công khai | không | không | HATE | rủi ro âm tính giả |
| Nhãn nhóm trong ngữ cảnh trung tính | có | không | NON-HATE | thường đúng nhờ cổng |

> 📌 **[CHÈN HÌNH] Hình 5 — Ma trận nhầm lẫn (ViHSD): baseline vs ViAmpleHate.** Đặt cạnh nhau (full-width). Dùng `confusion_matrix_amplehate.png` và `confusion_matrix_viamplehate.png` từ các thư mục `output/`; nhấn việc giảm dương tính giả ở lớp HATE.

**Bảng 7 — Kết quả theo lớp của ViAmpleHate trên ViHSD (test).**

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON-HATE | 0,9541 | 0,9574 | – |
| HATE | 0,6177 | 0,5988 | 0,6081 |

### 6.5 Thảo luận

Ba quan sát nổi bật. Thứ nhất, cải thiện tập trung ở lớp thiểu số: trên ViHSD, precision HATE tăng (0,5972 → 0,6177) trong khi recall giảm nhẹ (0,6119 → 0,5988), nên mô hình thận trọng hơn nhưng chính xác hơn — điều mong muốn khi quy kết nhầm thù ghét tốn kém. Thứ hai, mức tăng lớn hơn trên VOZ-HSD đi kèm accuracy giảm cho thấy mô hình đánh đổi accuracy lớp đa số lấy chất lượng lớp thiểu số, đúng đánh đổi mà người dùng thực tế mong muốn khi lớp thiểu số là đối tượng quan tâm. Thứ ba, sự phụ thuộc vào độ phủ cue gợi một con đường rẻ tiền để cải thiện thêm: mở rộng và tinh chỉnh cue bank sẽ mang lại cải thiện bổ sung mà không cần đổi kiến trúc.

## 7. Phân tích lỗi

Chúng tôi gom các lỗi còn lại thành bảy nhóm lặp lại. **(1) Thô tục nhưng không thù ghét:** lăng mạ/thô tục nhắm vào cá nhân chứ không phải nhóm được bảo vệ; tăng trọng số cue tấn công gây dương tính giả. **(2) Thù ghét ẩn:** không slur, không thực thể tên, không vị từ tấn công công khai, biểu đạt thù địch qua mỉa mai, định kiến, hay ngữ cảnh xã hội; cue extraction tìm được ít để attend. **(3) Quy chiếu đối tượng mơ hồ:** đại từ và danh từ chỉ nhóm cũng xuất hiện trong bình luận trung tính/hài hước, nên cue đối tượng không kèm ngữ cảnh thù địch dễ làm over-predict HATE. **(4) Độ phủ cue chưa đủ:** tiếng lóng, biến thể chính tả, viết tắt, và thô tục sáng tạo/bị kiểm duyệt thay đổi nhanh mà cue bank cố định không phủ hết, gây âm tính giả. **(5) Lỗi token hoá và canh chỉnh span:** span NER, tách từ, và token hoá subword của PhoBERT có thể lệch — nhất là với cụm nhiều từ — nên cue có thể khớp sai vị trí. **(6) Nhạy ngưỡng:** ngưỡng tinh chỉnh trên validation có thể không chuyển tốt khi phân bố đổi. **(7) Mất cân bằng lớp:** với ít mẫu dương, lỗi lớp thiểu số tồn tại dù có weighted loss và tinh chỉnh ngưỡng. Những điều này gợi: mở rộng cue bank qua khai phá dữ liệu cộng thẩm định thủ công; log dự đoán theo từng mẫu (kèm giá trị cổng, độ phủ cue, độ tự tin) để phân tích FP/FN hệ thống; thêm giám sát span hoặc phát hiện mỉa mai; và hiệu chỉnh xác suất cho ngưỡng ổn định hơn.

## 8. Kết luận và hướng phát triển

ViAmpleHate tổng quát hoá AmpleHate cho phát hiện thù ghét tiếng Việt qua năm thích ứng — trích đối tượng tiếng Việt, tách kênh cue đối tượng/tấn công, relation-bank attention, sửa cơ chế attention theo lô, và bơm quan hệ thích ứng — huấn luyện với cross-entropy có trọng số cộng hàm tương phản. Trên ViHSD và VOZ-HSD, nó cải thiện macro-F1 và HATE-F1 so với baseline AmpleHate và các baseline TF-IDF, BiLSTM, PhoBERT-CNN, với mức tăng lớn nhất ở lớp thiểu số. Phân tích gắn các cải thiện này với độ phủ đối tượng — thứ mà cue bank tiếng Việt nâng hơn hai bậc độ lớn so với NER tiếng Anh. Bài học lớn hơn: tính hướng-đối-tượng chỉ hữu ích khi được thích ứng theo bề mặt ngôn ngữ đích, nơi đối tượng thù ghét thường là cách gọi nhóm suồng sã chứ không phải thực thể có tên. Hướng tương lai: mở rộng và tự động khai phá cue bank kèm thẩm định thủ công; phân tích lỗi mức từng mẫu dùng giá trị cổng và độ tự tin; thêm giám sát span, phát hiện mỉa mai, ngữ cảnh người dùng/diễn ngôn; mở rộng sang đa nhãn và đa đối tượng theo tinh thần ViTHSD; và hiệu chỉnh xác suất cho triển khai liên miền ổn định.

## Giới hạn (Limitations)

Ngân hàng cue đối tượng và tấn công được xây một phần thủ công và không thể phủ hết không gian tiếng lóng, biến thể chính tả, và thô tục sáng tạo thay đổi nhanh của tiếng Việt, làm chặn recall với hate ẩn hoặc mới. Hiệu năng phụ thuộc chất lượng NER tiếng Việt, tách từ, và token hoá phía trước, mà sự lệch có thể đặt sai cue. Ngưỡng quyết định tinh chỉnh trên validation và có thể không tối ưu khi phân bố đổi. Mô hình giải quyết thiết lập nhị phân NON-HATE/HATE và chưa xử lý đa đối tượng hay mức độ thù ghét; bảng ablation ở Mục 6.3 là khung và nên được điền bằng các thí nghiệm có kiểm soát trước khi đưa ra khẳng định mạnh ở mức thành phần.

## Tuyên bố Đạo đức (Ethics Statement)

Phát hiện thù ghét là lưỡng dụng: cùng một mô hình hỗ trợ kiểm duyệt có thể, nếu triển khai sai, bóp nghẹt phát ngôn hợp pháp hoặc gắn cờ thiên lệch một số cộng đồng. Dữ liệu của chúng tôi là tài nguyên tiếng Việt đã công bố, dùng theo điều khoản nghiên cứu dự kiến của chúng; chúng tôi không thu thập dữ liệu người dùng mới. Vì chú thích và cue bank phản ánh phán đoán của người chú thích và tác giả, hệ thống có thể mang thiên kiến với một số phương ngữ, vùng miền, hoặc nhóm, và đầu ra nên được xem như hỗ trợ quyết định cho người kiểm duyệt chứ không phải cưỡng chế tự động. Chúng tôi tránh tái hiện slur thật trong bài, dùng ví dụ dựng lại hoặc đã che. Chúng tôi khuyến nghị hiệu chỉnh, rà soát có người trong vòng lặp, và kiểm toán thiên kiến liên tục trong mọi triển khai.

## Tài liệu tham khảo

- **AmpleHate** — Lee, Y., Hahn, J., Ahn, H., Han, Y.-S. (2025). "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection." *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (Suzhou, Trung Quốc), tr. 28862–28874. ACL. DOI: 10.18653/v1/2025.emnlp-main.1469.
- **ViTHSD** — Vo, C. N., Huynh, K. B., Luu, S. T., Do, T.-H. (2025). "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts." *Journal of Computational Social Science* 8(2):30. Springer. DOI: 10.1007/s42001-024-00348-6.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts." *Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices* (LNCS 12798), tr. 415–426. Springer, Cham. DOI: 10.1007/978-3-030-79457-6_35.
- **VOZ-HSD** — Thanh Nguyen, L. (2024). "VOZ-HSD: A Hate Speech Detection Dataset from the VOZ Forum." Bộ dữ liệu Hugging Face, `tarudesu/VOZ-HSD`. Phát hành kèm ViHateT5.
- **ViHateT5** — Thanh Nguyen, L. (2024). "ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model." *Findings of the ACL 2024* (Bangkok, Thái Lan), tr. 5948–5961. ACL. DOI: 10.18653/v1/2024.findings-acl.355.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*, tr. 1037–1042.
- **BERT** — Devlin và cộng sự (2019). *NAACL*.
- **fastText** — Bojanowski và cộng sự (2017). *TACL*.
- **Supervised Contrastive Learning** — Khosla và cộng sự (2020). *NeurIPS*.
- **NER tiếng Việt** — `NlpHUST/ner-vietnamese-electra-base`.

---

## Phụ lục A — Siêu tham số

| Tham số | Giá trị |
|---|---|
| Bộ mã hoá | `vinai/phobert-base` (768-d) |
| NER tiếng Việt | `NlpHUST/ner-vietnamese-electra-base` |
| max_len | 256 |
| LR (mã hoá / đầu) | 2e-5 / 5e-5 |
| Dropout | 0,1 |
| Lô hiệu dụng | 32 (16 × tích luỹ 2) |
| Epochs | tối đa 8 (tốt nhất theo val macro-F1) |
| α (tương phản) | 0,1 |
| ViHSD epoch tốt nhất / val F1 / ngưỡng | 4 / 0,7852 / 0,43 |

## Phụ lục B — Ví dụ cue bank

> Chỉ là mục tiêu biểu; ngân hàng đầy đủ phát hành kèm mã nguồn. Cue đối tượng mang tính quy chiếu (bản thân không thù ghét): cách gọi nhóm/người suồng sã, nhãn vùng miền và nhân khẩu, đại từ chỉ nhóm. Cue tấn công là vị từ thù địch thể hiện khinh miệt, phi nhân hoá, hay ăn bám (vd `khinh`, `ăn_bám`).

## Phụ lục C — Khả năng tái lập

Mọi mô hình dựa PhoBERT dùng chung `vinai/phobert-base`; chỉ khác nhau ở mô hình hoá đối tượng/quan hệ, cách tính attention, hợp nhất, và hàm mất mát giữa baseline và ViAmpleHate. Checkpoint tốt nhất chọn theo macro-F1 validation, ngưỡng HATE chọn trên validation và cố định cho test. Việc khớp cue thực hiện một lần trong tiền xử lý bằng khớp toàn chuỗi token với chuỗi token của PhoBERT.
</content>
