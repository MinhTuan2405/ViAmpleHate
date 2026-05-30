# ViAmpleHate: Phương pháp dựa trên AmpleHate và PhoBERT cho bài toán Phát hiện Phát ngôn Thù ghét tiếng Việt

> Tài liệu này là bản thảo nội dung viết bằng văn xuôi, dùng để chuyển thành bài báo LaTeX 8–10 trang theo mẫu hội nghị ACL. Các công thức được viết sẵn theo cú pháp LaTeX, bảng ở dạng Markdown (cần đổi sang `booktabs`), và những vị trí cần chèn hình được đánh dấu bằng khối **📌 [CHÈN HÌNH]**.

## Tóm tắt (Abstract)

Phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt là một bài toán khó, bởi ý đồ thù ghét thường không được biểu đạt qua các thực thể có tên rõ ràng mà ẩn trong cách gọi nhóm suồng sã, tiếng lóng và những ám chỉ gián tiếp. Những mô hình hướng-đối-tượng (target-aware) như AmpleHate đã cho thấy rằng việc chú ý đến quan hệ giữa câu nói và đối tượng bị nhắm tới giúp phát hiện cả những trường hợp thù ghét ẩn. Tuy nhiên, quy trình gốc của AmpleHate phụ thuộc vào nhận dạng thực thể có tên (NER) tiếng Anh và các phạm trù đối tượng mang tính tiếng Anh, nên chuyển giao sang tiếng Việt rất kém hiệu quả. Chúng tôi đề xuất **ViAmpleHate**, một bản thích ứng của AmpleHate cho tiếng Việt, xây dựng trên nền **PhoBERT**. ViAmpleHate thay NER tiếng Anh bằng NER tiếng Việt, bổ sung hai ngân hàng tín hiệu là *cue đối tượng* và *cue tấn công*, mô hình hoá ba kênh quan hệ (đối tượng tường minh, ngữ cảnh ngầm, và tấn công) thông qua một cơ chế *relation-bank attention*, sửa một lỗi trộn mẫu trong tính attention theo lô, và thay cơ chế bơm thông tin bằng hệ số cố định bằng một *cổng thích ứng theo từng mẫu*. Mô hình được huấn luyện kết hợp hàm mất mát cross-entropy có trọng số với một hàm tương phản (contrastive). Trên hai bộ dữ liệu **ViHSD** và **VOZ-HSD** ở thiết lập nhị phân NON-HATE/HATE, ViAmpleHate cải thiện macro-F1 và F1 của lớp thiểu số HATE so với một baseline AmpleHate trung thực cũng như các baseline TF-IDF, BiLSTM và PhoBERT-CNN. Kết quả này khẳng định rằng việc mô hình hoá hướng-đối-tượng cần được thích ứng theo đặc trưng ngôn ngữ tiếng Việt mới phát huy tác dụng.

## 1. Giới thiệu

Sự bùng nổ của mạng xã hội tiếng Việt đi kèm với sự lan rộng của các nội dung thù ghét, gây tổn hại cho cá nhân lẫn cộng đồng và đặt ra nhu cầu cấp thiết về các công cụ kiểm duyệt tự động ở quy mô lớn. Tuy nhiên, phát hiện phát ngôn thù ghét trong tiếng Việt vấp phải nhiều khó khăn đặc thù. Trước hết, ngôn ngữ mạng xã hội rất nhiễu: người dùng viết tắt, dùng teencode, kéo dài ký tự, xen lẫn emoji và viết sai chính tả một cách cố ý hoặc vô tình. Thứ hai, đối tượng bị tấn công trong tiếng Việt hiếm khi là một thực thể có tên; thay vào đó, người ta thường nhắm tới một nhóm qua đại từ, danh từ chỉ nhóm hoặc cách gọi suồng sã. Thứ ba, dữ liệu mất cân bằng nghiêm trọng: lớp HATE chỉ chiếm khoảng một phần mười số mẫu.

Hướng tiếp cận hướng-đối-tượng, tiêu biểu là AmpleHate, đã chứng minh rằng việc mô hình hoá quan hệ giữa một câu và đối tượng mà nó nhắm tới giúp phát hiện được cả những phát ngôn thù ghét ẩn. Vấn đề là AmpleHate gốc xác định đối tượng bằng NER tiếng Anh. Khi áp dụng cho tiếng Việt, bộ NER này gần như không tìm được đối tượng hợp lệ nào: trên tập huấn luyện ViHSD, chỉ 21 trên 24.048 mẫu (khoảng 0,09%) có đối tượng được NER tiếng Anh phát hiện. Hệ quả là mô hình gần như luôn rơi về biểu diễn câu tổng quát ở token `[CLS]`, và thoái hoá thành một bộ phân loại PhoBERT thông thường, đánh mất toàn bộ lợi thế hướng-đối-tượng.

Trong bài báo này, chúng tôi đề xuất ViAmpleHate nhằm khắc phục đúng những điểm yếu đó. Đóng góp của chúng tôi gồm năm điểm. **Thứ nhất**, chúng tôi thay NER tiếng Anh bằng NER tiếng Việt và bổ sung một ngân hàng *cue đối tượng* gồm các đại từ, danh từ chỉ nhóm và cách gọi suồng sã; nhờ đó độ phủ đối tượng tăng từ khoảng 0,09% lên khoảng 18,8–20,0%. **Thứ hai**, chúng tôi tách bạch hai loại tín hiệu: cue đối tượng trả lời cho câu hỏi "ai đang bị nói tới", còn cue tấn công trả lời cho câu hỏi "đối tượng đó có đang bị công kích hay không", và mô hình hoá chúng qua một cơ chế relation-bank attention gồm ba kênh. **Thứ ba**, chúng tôi sửa một lỗi trong cách tính attention theo lô khiến thông tin bị trộn giữa các mẫu, và thay cơ chế bơm thông tin bằng hệ số cố định bằng một cổng thích ứng tính riêng cho từng mẫu. **Thứ tư**, chúng tôi huấn luyện mô hình bằng tổ hợp cross-entropy có trọng số và hàm tương phản để cải thiện khả năng tách lớp trong điều kiện mất cân bằng. **Thứ năm**, chúng tôi đánh giá trên ViHSD và VOZ-HSD, so sánh với năm baseline khác nhau và quan sát thấy cải thiện ở cả macro-F1 lẫn F1 của lớp HATE. Thông điệp xuyên suốt là: tính hướng-đối-tượng thực sự hữu ích, nhưng chỉ khi nó được thiết kế phù hợp với cách người Việt biểu đạt sự thù ghét.

## 2. Các nghiên cứu liên quan

### 2.1. AmpleHate: Khuếch đại attention cho phát hiện thù ghét ẩn

AmpleHate hướng tới việc phát hiện phát ngôn thù ghét ẩn bằng cách khuếch đại sự chú ý (attention) giữa biểu diễn toàn câu ở token `[CLS]` và các đối tượng tiềm năng được trích ra bằng NER. Một vector quan hệ hướng-đối-tượng được tính qua cơ chế HeadAttention, trong đó truy vấn xuất phát từ `[CLS]` còn khoá và giá trị xuất phát từ các token đối tượng; vector này sau đó được bơm vào biểu diễn câu theo công thức $z = h_0 + e\cdot r$ với $e$ là một hệ số cố định, trước khi đưa vào bộ phân loại. Cách làm này hiệu quả với tiếng Anh, nhưng bộc lộ ba hạn chế khi chuyển sang tiếng Việt: nó phụ thuộc vào NER và phạm trù đối tượng kiểu tiếng Anh, nó bơm thông tin với một cường độ cố định cho mọi mẫu, và nó chỉ mô hình hoá một kênh quan hệ duy nhất là đối tượng mà bỏ qua tín hiệu tấn công. Công trình của chúng tôi kế thừa trực tiếp ý tưởng hướng-đối-tượng của AmpleHate nhưng tái thiết kế lại pipeline cho tiếng Việt.

### 2.2. ViTHSD: Khai thác sự thù ghét theo đối tượng cho phát hiện thù ghét trên mạng xã hội tiếng Việt

ViTHSD tiếp cận bài toán theo hướng phát hiện thù ghét gắn với từng đối tượng cụ thể, gán mức độ thù ghét cho từng đối tượng được nhắm tới trong câu. Công trình này củng cố quan sát rằng đối tượng là tín hiệu cốt lõi của phát ngôn thù ghét tiếng Việt, qua đó ủng hộ động lực hướng-đối-tượng của ViAmpleHate. Điểm khác biệt là ViAmpleHate không yêu cầu nhãn đối tượng ở mức span; thay vào đó, chúng tôi dùng kết hợp NER và các ngân hàng cue để xác định vị trí đối tượng và tấn công mà không cần giám sát span. Bên cạnh đó, để định vị bối cảnh, các baseline tiếng Việt phổ biến mà chúng tôi so sánh gồm PhoBERT, PhoBERT-CNN, BiLSTM với embedding fastText, và mô hình TF-IDF cổ điển.

## 3. Dữ liệu

### 3.1. ViHSD — Bộ dữ liệu phát hiện phát ngôn thù ghét tiếng Việt

ViHSD là một bộ dữ liệu phát hiện phát ngôn thù ghét tiếng Việt gồm các bình luận trên mạng xã hội, được gán ba nhãn gốc là CLEAN, OFFENSIVE và HATE, và được chia sẵn thành các tập huấn luyện, kiểm định và kiểm tra. Sau khi chuyển về thiết lập nhị phân (trình bày ở mục 3.3), phân bố lớp của ba tập được thể hiện trong Bảng 1.

### 3.2. VOZ-HSD — Bộ dữ liệu phát hiện thù ghét trên diễn đàn VOZ

VOZ-HSD gồm các bình luận thu thập từ diễn đàn VOZ, với nguồn gốc từ tập `tarudesu/VOZ-HSD` trên HuggingFace. Dữ liệu cũng được chia thành các tập huấn luyện, kiểm định và kiểm tra; phân bố lớp sau khi gán lại nhãn được trình bày chung trong Bảng 1.

### 3.3. Gán lại nhãn dữ liệu (đưa về nhị phân)

Thao tác duy nhất mà chúng tôi thực hiện trên dữ liệu là gán lại nhãn, hoàn toàn không thay đổi hay chọn lọc số lượng mẫu. Cụ thể, chúng tôi đưa bài toán về phân loại nhị phân bằng cách gộp hai nhãn CLEAN và OFFENSIVE thành một lớp NON-HATE, đồng thời giữ nguyên HATE làm lớp dương. Lựa chọn này giúp tập trung vào phát ngôn thù ghét nhắm vào một nhóm hay đối tượng cụ thể, thay vì sự thô tục hay công kích nói chung, và đồng thời tạo ra một thiết lập nhãn nhất quán giữa hai bộ dữ liệu. Trước khi đưa vào mô hình, mọi bình luận đều được chuẩn hoá để giảm nhiễu: chuyển về chữ thường, loại bỏ URL và các ký hiệu phi ngôn ngữ, gộp các ký tự lặp, chuẩn hoá teencode, và ánh xạ một số emoji sang các nhãn ngữ dụng thô như chế nhạo, giận dữ, ghê tởm hay cười cợt. Sau đó văn bản được tách từ — bước bắt buộc vì PhoBERT được huấn luyện trên văn bản đã tách từ — rồi mới được token hoá.

**Bảng 1 — Thống kê dữ liệu sau khi gán nhị phân.**

| Bộ dữ liệu | Tập | NON-HATE | HATE | Tổng | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21.492 | 2.556 | 24.048 | 10,6% |
| ViHSD | Dev | 2.402 | 270 | 2.672 | 10,1% |
| ViHSD | Test | 5.992 | 688 | 6.680 | 10,3% |
| VOZ-HSD | Train | 26.993 | 3.007 | 30.000 | 10,0% |
| VOZ-HSD | Dev | 4.520 | 480 | 5.000 | 9,6% |
| VOZ-HSD | Test | 4.487 | 513 | 5.000 | 10,3% |

> 📌 **[CHÈN HÌNH] Hình 1 — Phân bố lớp.** Biểu đồ cột thể hiện mức mất cân bằng giữa NON-HATE và HATE trên cả hai bộ dữ liệu (số liệu từ Bảng 1). **Cần vẽ mới** bằng matplotlib, đặt ngay sau Bảng 1, một cột. Chú thích nên nhấn mạnh rằng vì HATE chỉ chiếm khoảng 10% nên macro-F1 và HATE-F1 quan trọng hơn accuracy.

## 4. Phương pháp

Chúng tôi phát biểu bài toán dưới dạng phân loại nhị phân: với một bình luận tiếng Việt $x$, mô hình dự đoán nhãn NON-HATE hoặc HATE. ViAmpleHate kế thừa ý tưởng hướng-đối-tượng của AmpleHate và thích ứng nó cho tiếng Việt qua năm thay đổi: trích đối tượng theo tiếng Việt, tách riêng cue đối tượng và cue tấn công, dùng relation-bank attention, sửa cơ chế attention theo lô, và bơm thông tin quan hệ qua một cổng thích ứng.

> 📌 **[CHÈN HÌNH] Hình 2 — Kiến trúc tổng thể của ViAmpleHate.** Đây là hình quan trọng nhất, **cần vẽ mới** (draw.io hoặc PowerPoint) và đặt ở đầu mục 4. Sơ đồ nên thể hiện luồng: bình luận đầu vào → chuẩn hoá → tách từ → bộ mã hoá PhoBERT cho ra `[CLS]` $h_0$ và các trạng thái token $H$; từ đó tách hai nhánh tín hiệu, một nhánh dùng NER tiếng Việt cùng ngân hàng cue đối tượng để tạo tập $T_x$, một nhánh dùng ngân hàng cue tấn công để tạo tập $A_x$; ba kênh relation-bank attention tạo ra $r_{exp}$, $r_{imp}$, $r_{atk}$; các vector này được hợp nhất qua $W_r$, rồi đi qua cổng thích ứng $g=\sigma(W_g[h_0;r])$ để tạo $z = h_0 + g\cdot r$, cuối cùng qua một lớp tuyến tính để dự đoán hai nhãn. Nên ghi chú hàm mất mát $L = L_{CE} + \alpha L_{CL}$ ở khối phân loại.

### 4.1. Tiền xử lý văn bản

Văn bản mạng xã hội tiếng Việt được chuẩn hoá để giảm nhiễu trước khi mã hoá: chuyển chữ thường, loại bỏ URL và các phần phi ngôn ngữ, gộp ký tự lặp, chuẩn hoá các biến thể teencode, và ánh xạ một số emoji sang nhãn ngữ dụng thô. Sau khi chuẩn hoá, mỗi bình luận được tách từ rồi token hoá bằng tokenizer của PhoBERT và cắt hoặc đệm về độ dài cố định, ở đây là 256 token. Bước tách từ là bắt buộc vì PhoBERT được tiền huấn luyện trên văn bản tiếng Việt đã tách từ.

### 4.2. Trích tín hiệu đối tượng và tấn công đa nguồn

ViAmpleHate dùng một chiến lược trích tín hiệu đa nguồn để xác định các vị trí có ý nghĩa ngôn ngữ trong chuỗi đầu vào. Tín hiệu đối tượng được lấy từ NER tiếng Việt — phát hiện người, tổ chức, địa điểm, thực thể địa chính trị và các nhóm có tên — kết hợp với một ngân hàng cue đối tượng chứa các biểu thức quy chiếu tiếng Việt thường dùng để giới thiệu một người hay một nhóm. Bản thân các cue này không phải là chỉ dấu thù ghét; chúng chỉ đánh dấu các vị trí đối tượng khả dĩ. Song song, một ngân hàng cue tấn công chứa các vị từ công kích, lăng mạ, đe doạ và đánh giá tiêu cực. Việc tách riêng cue đối tượng và cue tấn công là có chủ đích: cue đối tượng giúp trả lời ai đang được nói tới, còn cue tấn công giúp xác định liệu có sự đánh giá thù địch nào hướng vào đối tượng đó hay không. Sự tách bạch này ngăn mô hình hiểu nhầm mọi từ thô tục là đối tượng, hay mọi cách gọi đối tượng là tấn công. Mỗi cụm cue được chuẩn hoá, tách từ, token hoá rồi đối khớp với chuỗi token của PhoBERT bằng phép khớp toàn chuỗi token, nhằm tránh khớp nhầm vào một mảnh subtoken. Nếu không tìm thấy cue tường minh nào, tập tương ứng sẽ lùi về vị trí `[CLS]`, cho phép mô hình giữ lại một biểu diễn câu ngầm.

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{nếu}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{nếu}\ A_x=\varnothing
$$

### 4.3. Bộ mã hoá PhoBERT

Đầu vào đã chuẩn hoá và tách từ được mã hoá bằng PhoBERT, cho ra dãy trạng thái ẩn cuối cùng

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768,
$$

trong đó $h_0$ là biểu diễn `[CLS]` đại diện cho ngữ cảnh toàn câu, còn các vị trí đối tượng và tấn công đã trích cung cấp bằng chứng cục bộ cho suy luận hướng-đối-tượng.

### 4.4. Relation-bank attention

ViAmpleHate xây dựng một ngân hàng quan hệ gồm ba góc nhìn: một quan hệ đối tượng tường minh từ các token đối tượng, một quan hệ ngữ cảnh ngầm từ neo `[CLS]`, và một quan hệ tấn công từ các token tấn công.

$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$

Mỗi mô-đun HeadAttention, với ma trận token quan hệ $E \in \mathbb{R}^{m\times d}$, được tính như sau:

$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V.
$$

Ba vector quan hệ được nối lại và chiếu thành một biểu diễn quan hệ hợp nhất:

$$
r = W_r\,[\,r_{\text{exp}}\,;\,r_{\text{imp}}\,;\,r_{\text{atk}}\,] + b_r.
$$

Vector quan hệ hợp nhất này nắm bắt các thông tin bổ trợ lẫn nhau từ việc nhắc đến đối tượng, ngữ cảnh ở cấp câu, và các biểu thức tấn công.

### 4.5. Sửa lỗi attention theo lô

Cách cài đặt theo kiểu AmpleHate gốc tính attention bằng một phép nhân ma trận tương đương với $QK^\top$ trên toàn lô. Khi $Q,K\in\mathbb{R}^{B\times d}$ thì $QK^\top\in\mathbb{R}^{B\times B}$, vô tình trộn thông tin giữa các mẫu khác nhau trong cùng một lô. ViAmpleHate sửa lỗi này bằng attention theo lô, sao cho mỗi mẫu chỉ chú ý đến các token đối tượng hoặc tấn công của chính nó:

$$
Q\in\mathbb{R}^{B\times 1\times d},\quad K\in\mathbb{R}^{B\times m\times d},\quad
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m}.
$$

Các vị trí đệm bị triệt tiêu bằng mặt nạ attention trước phép softmax: $\text{scores}_j = -\infty$ nếu $\text{mask}_j=0$.

### 4.6. Cổng quan hệ thích ứng theo từng mẫu

Cơ chế AmpleHate gốc dùng một số vô hướng cố định để điều khiển lượng attention hướng-đối-tượng được bơm vào biểu diễn câu. ViAmpleHate thay cường độ cố định đó bằng một cổng thích ứng tính riêng cho từng mẫu:

$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r.
$$

Cổng này cho phép mô hình tự quyết định mức độ ảnh hưởng của thông tin quan hệ lên từng dự đoán. Nếu một bình luận có bằng chứng đối tượng và tấn công rõ ràng, mô hình có thể tăng trọng số cho vector quan hệ; nếu các cue yếu, mơ hồ hoặc vắng mặt, mô hình có thể dựa nhiều hơn vào biểu diễn câu tổng quát. Biểu diễn cuối cùng $z$ đi qua dropout và một lớp tuyến tính để cho ra logits cho NON-HATE và HATE.

### 4.7. Mục tiêu huấn luyện

Mô hình được huấn luyện bằng tổ hợp cross-entropy có trọng số và hàm tương phản:

$$
L = L_{\text{CE}} + \alpha\, L_{\text{CL}}, \qquad \alpha = 0.1.
$$

Thành phần cross-entropy có trọng số, $L_{\text{CE}} = -\sum_c w_c\,y_c\log \hat y_c$, xử lý mất cân bằng lớp bằng cách gán trọng số cao hơn cho lớp HATE thiểu số, kèm theo label smoothing để giảm dự đoán quá tự tin. Thành phần tương phản được áp lên biểu diễn $z$ sau cổng, khuyến khích các mẫu cùng lớp có biểu diễn gần nhau và các mẫu khác lớp tách xa nhau theo độ tương đồng cosine:

$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-\text{margin})\Big].
$$

Khi đánh giá, ngưỡng quyết định cho lớp HATE được chọn trên tập kiểm định bằng cách tối đa hoá macro-F1, sau đó áp dụng cố định cho tập kiểm tra:

$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big).
$$

## 5. Thực nghiệm

### 5.1. Các baseline

Chúng tôi so sánh ViAmpleHate với năm baseline, tất cả đều ở thiết lập nhị phân. Hai baseline cổ điển là TF-IDF kết hợp Hồi quy Logistic và TF-IDF kết hợp SVM, sử dụng đặc trưng từ vựng thưa. Tiếp theo là BiLSTM với embedding fastText tiếng Việt, kết hợp embedding tĩnh với mô hình chuỗi. Mạnh hơn là PhoBERT-CNN, dùng PhoBERT làm bộ mã hoá ngữ cảnh và CNN để trích đặc trưng cục bộ. Cuối cùng, baseline quan trọng nhất là AmpleHate-PhoBERT, một bản port trung thực của AmpleHate gốc với NER tiếng Anh, một mô-đun HeadAttention duy nhất và cơ chế bơm cố định $z = h_0 + e\,r_{\text{base}}$ với $e=1.0$. Đây là đối thủ trực tiếp vì dùng cùng bộ mã hoá PhoBERT, nên mọi khác biệt về kết quả phản ánh đúng tác động của các thay đổi mà chúng tôi đề xuất.

### 5.2. Đối chiếu giữa baseline và mô hình đề xuất

Bảng 2 tóm tắt các khác biệt cụ thể về kiến trúc và cấu hình giữa baseline AmpleHate-PhoBERT và ViAmpleHate-PhoBERT.

**Bảng 2 — Đối chiếu kiến trúc và cấu hình.**

| Thành phần | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (đề xuất) |
|---|---|---|
| Bộ mã hoá | PhoBERT-base | PhoBERT-base |
| Trích đối tượng | NER tiếng Anh (`dbmdz/bert-large-...-conll03-english`) | NER tiếng Việt (`NlpHUST/ner-vietnamese-electra-base`) + ngân hàng cue đối tượng |
| Độ phủ đối tượng | ~21/24.048 ≈ 0,09% mẫu train | ~18,8–20,0% nhờ cue tiếng Việt |
| Tín hiệu tấn công | Không mô hình hoá | Ngân hàng cue tấn công riêng |
| Attention | Một HeadAttention | Ba kênh: đối tượng / ngầm / tấn công |
| Cách tính attention | `matmul(Q,Kᵀ)` cấp lô (trộn mẫu) | `bmm` + mặt nạ theo từng mẫu |
| Hợp nhất | Bơm cố định $h_0+e\,r$ ($e{=}1.0$) | Relation bank + cổng thích ứng $h_0+g\,r$ |
| Hàm mất mát | Weighted CE | Weighted CE + Contrastive ($\alpha{=}0.1$) |
| Độ dài tối đa | 128 | 256 |
| Lô | 16 | 16 × tích luỹ gradient 2 (hiệu dụng 32) |
| NER khi đánh giá | Tắt (dẫn đến fallback `[CLS]`) | Bật, nhất quán train/val/test |

### 5.3. Chi tiết cài đặt

Bộ mã hoá là `vinai/phobert-base` với chiều ẩn 768, NER tiếng Việt là `NlpHUST/ner-vietnamese-electra-base`. Độ dài tối đa của chuỗi là 256 token. Tốc độ học là 2e-5 cho bộ mã hoá và 5e-5 cho phần đầu phân loại, dropout 0,1. Kích thước lô hiệu dụng là 32, đạt được bằng lô 16 với tích luỹ gradient hai bước. Mô hình được huấn luyện tối đa tám epoch, chọn checkpoint tốt nhất theo macro-F1 trên tập kiểm định. Trọng số tương phản $\alpha$ đặt bằng 0,1, và ngưỡng quyết định cũng được chọn trên tập kiểm định theo macro-F1.

### 5.4. Độ đo đánh giá

Hai độ đo chính là macro-F1 và F1 của lớp HATE, vì chúng phản ánh đúng hiệu năng trên lớp thiểu số hơn so với accuracy. Accuracy chỉ được dùng để tham khảo, bởi trong dữ liệu mất cân bằng, một mô hình có thể đạt accuracy cao chỉ nhờ dự đoán đúng phần lớn các mẫu NON-HATE mà vẫn bỏ sót nhiều phát ngôn thù ghét. Ngưỡng tối ưu được chọn trên tập kiểm định rồi cố định khi dự đoán trên tập kiểm tra.

### 5.5. Mục đích thực nghiệm

Mục đích không chỉ là so sánh điểm số cuối cùng, mà còn để kiểm chứng rằng mỗi thay đổi giải quyết một hạn chế cụ thể của baseline. NER và cue tiếng Việt nhằm xử lý độ phủ đối tượng thấp; cue tấn công nhằm bù đắp việc thiếu mô hình hoá sự thù địch; relation-bank attention đáp ứng nhu cầu tách riêng đối tượng, ngữ cảnh và tấn công; attention theo lô đã sửa khắc phục hiện tượng rò rỉ thông tin chéo giữa các mẫu; cổng thích ứng thay thế cho cơ chế bơm cố định; và hàm tương phản cải thiện khả năng tách lớp dưới điều kiện mất cân bằng.

## 6. Phân tích kết quả và lỗi

### 6.1. Kết quả thực nghiệm

Bảng 3 và Bảng 4 trình bày kết quả trên tập kiểm tra của ViHSD và VOZ-HSD.

**Bảng 3 — Kết quả trên ViHSD (tập kiểm tra).**

| Mô hình | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0,8910 | 0,7393 | 0,5404 |
| TF-IDF + SVM | 0,9126 | 0,7131 | 0,4739 |
| BiLSTM + fastText | – | 0,7072 | 0,5060 |
| PhoBERT-CNN | – | 0,7571 | 0,5745 |
| AmpleHate-PhoBERT (baseline) | 0,9175 | 0,7792 | 0,6045 |
| **ViAmpleHate-PhoBERT (đề xuất)** | **0,9205** | **0,7819** | **0,6081** |
| *Δ so với baseline* | *+0,0030* | *+0,0027* | *+0,0036* |

**Bảng 4 — Kết quả trên VOZ-HSD (tập kiểm tra).**

| Mô hình | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0,9453 | 0,7745 | 0,5783 |
| TF-IDF + SVM | 0,9641 | 0,7831 | 0,5850 |
| BiLSTM + fastText | – | 0,6712 | 0,4187 |
| PhoBERT-CNN | – | 0,8150 | 0,6500 |
| AmpleHate-PhoBERT (baseline) | 0,9643 | 0,8185 | 0,6557 |
| **ViAmpleHate-PhoBERT (đề xuất)** | 0,9420 | **0,8371** | **0,7065** |
| *Δ so với baseline* | *–0,0223* | *+0,0186* | *+0,0508* |

Kết quả cho thấy các mô hình dựa trên transformer nhìn chung vượt trội so với các baseline từ vựng và embedding tĩnh. Điều này hợp lý, vì phát ngôn thù ghét thường phụ thuộc vào ngữ cảnh, cách diễn đạt suồng sã và tương tác giữa việc nhắc đến đối tượng với các vị từ thù địch — những thứ mà mô hình từ vựng thưa khó nắm bắt và embedding tĩnh chỉ nắm bắt một phần. Baseline AmpleHate đã cải thiện so với một bộ phân loại PhoBERT thuần nhờ attention hướng-đối-tượng, nhưng lợi ích bị hạn chế vì NER tiếng Anh hiếm khi phát hiện được đối tượng tiếng Việt hợp lệ, khiến mô hình thường rơi về biểu diễn `[CLS]`.

ViAmpleHate cải thiện đúng hai độ đo quan trọng nhất là macro-F1 và HATE-F1 trên cả hai bộ dữ liệu, và mức cải thiện rõ rệt hơn hẳn trên VOZ-HSD, nơi HATE-F1 tăng tới 0,0508. Trên ViHSD, mức tăng nhỏ nhưng nhất quán: precision của lớp HATE tăng từ 0,5972 lên 0,6177 trong khi recall giảm nhẹ từ 0,6119 xuống 0,5988, cho thấy mô hình trở nên thận trọng hơn nhưng chính xác hơn khi gán nhãn HATE. Đáng chú ý, trên VOZ-HSD accuracy giảm trong khi macro-F1 và HATE-F1 đều tăng — một minh hoạ trực tiếp cho lý do không nên dùng accuracy làm độ đo chính trên dữ liệu mất cân bằng. Cuối cùng, lợi ích của ViAmpleHate phụ thuộc vào chất lượng và độ phủ của các cue được trích: khi cue được phát hiện nhiều và chính xác, relation-bank attention có nhiều bằng chứng hữu ích để khai thác; khi độ phủ cue thấp, mô hình buộc phải dựa nhiều hơn vào nhánh `[CLS]` ngầm và khoảng cách với baseline thu hẹp lại.

> 📌 **[CHÈN HÌNH] Hình 3 — Đường cong huấn luyện.** Đặt trong mục 6.1. File sẵn có: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png` (có thể ghép cạnh đường cong của baseline tại `.../baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/training_curves_amplehate.png`). Một cột. Chú thích nên ghi rõ epoch tốt nhất là 4 với macro-F1 kiểm định 0,7852.

> 📌 **[CHÈN HÌNH] Hình 4 — Ma trận nhầm lẫn trên ViHSD: baseline so với ViAmpleHate.** Đặt cạnh nhau (full-width hai cột) trong mục 6.1 hoặc đầu 6.2. File sẵn có: baseline tại `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/confusion_matrix_amplehate.png` và đề xuất tại `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate.png`. Chú thích nên nhấn việc giảm số dương tính giả ở lớp HATE.

> 📌 **[CHÈN HÌNH — tuỳ chọn] Hình 5 — Ma trận nhầm lẫn trên VOZ-HSD (mô hình đề xuất).** File: `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate_vozhsd.png`. Dùng nếu còn chỗ, để minh hoạ mức cải thiện HATE lớn hơn trên VOZ.

**Bảng 5 (tuỳ chọn) — Precision/Recall/F1 theo lớp trên ViHSD (mô hình đề xuất).**

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON-HATE | 0,9541 | 0,9574 | – |
| HATE | 0,6177 | 0,5988 | 0,6081 |

### 6.2. Phân tích lỗi

Các lỗi còn lại có thể chia thành một số nhóm. Nhóm thứ nhất là sự nhầm lẫn giữa ngôn ngữ công kích và phát ngôn thù ghét: nhiều bình luận tiếng Việt chứa lăng mạ hay thô tục nhưng không nhắm vào một nhóm được bảo vệ, và nếu mô hình dựa quá mạnh vào cue tấn công thì sẽ gán nhầm chúng thành HATE, gây ra dương tính giả. Nhóm thứ hai là thù ghét ẩn: một số bình luận thù ghét không chứa slur, thực thể có tên hay vị từ tấn công rõ ràng mà biểu đạt sự thù địch qua mỉa mai, ám chỉ, định kiến hay bối cảnh xã hội chung, khiến mô hình dựa trên cue khó phát hiện. Nhóm thứ ba là quy chiếu đối tượng mơ hồ: các đại từ và danh từ chỉ nhóm trong tiếng Việt có thể xuất hiện cả trong những bình luận trung tính hay hài hước, nên việc phát hiện một cue đối tượng mà thiếu ngữ cảnh thù địch có thể khiến mô hình đánh giá quá cao khả năng thù ghét.

Nhóm thứ tư là độ phủ cue chưa đầy đủ: ngôn ngữ mạng tiếng Việt thay đổi nhanh với vô số biến thể chính tả, tiếng lóng, viết tắt và cách viết thô tục sáng tạo, nên một ngân hàng cue cố định không thể phủ hết, dẫn tới âm tính giả khi các cue quan trọng bị thiếu hoặc không được chuẩn hoá đúng. Nhóm thứ năm là lỗi token hoá và canh chỉnh span: NER, tách từ và token hoá subword của PhoBERT không phải lúc nào cũng khớp nhau, đặc biệt với các biểu thức nhiều từ, nên nếu cue không khớp đúng vị trí thì mô-đun attention có thể chú ý vào bằng chứng thiếu hoặc không liên quan. Nhóm thứ sáu là độ nhạy với ngưỡng: ngưỡng tối ưu trên tập kiểm định không nhất thiết còn tối ưu khi phân bố hay miền dữ liệu thay đổi, điều đặc biệt quan trọng khi triển khai. Nhóm thứ bảy là mất cân bằng lớp: vì HATE thường ít hơn nhiều so với NON-HATE, mô hình thấy ít mẫu dương trong huấn luyện, và dù hàm mất mát có trọng số cùng việc tinh chỉnh ngưỡng giúp giảm bớt, lỗi ở lớp thiểu số vẫn còn.

Những lỗi này gợi ra một số hướng cải thiện, dẫn trực tiếp vào phần kết luận: mở rộng các ngân hàng cue đối tượng và tấn công qua khai phá dữ liệu kết hợp thẩm định thủ công; lưu nhật ký dự đoán theo từng mẫu để phân tích dương tính giả và âm tính giả một cách hệ thống theo độ phủ cue, giá trị cổng và độ tự tin; bổ sung giám sát span đối tượng, phát hiện mỉa mai hay thông tin bối cảnh xã hội; và áp dụng các phương pháp hiệu chỉnh xác suất để ngưỡng quyết định ổn định hơn giữa các miền.

## 7. Kết luận và hướng phát triển

### 7.1. Kết luận

ViAmpleHate tổng quát hoá AmpleHate cho bài toán phát hiện phát ngôn thù ghét tiếng Việt thông qua năm thích ứng chính: trích đối tượng theo tiếng Việt, mô hình hoá riêng cue đối tượng và cue tấn công, relation-bank attention, sửa cơ chế attention theo lô, và bơm thông tin quan hệ thích ứng; mô hình được huấn luyện với tổ hợp cross-entropy và hàm tương phản. Trên ViHSD và VOZ-HSD ở thiết lập nhị phân, mô hình cải thiện macro-F1 và HATE-F1 so với baseline AmpleHate cũng như các baseline TF-IDF, BiLSTM và PhoBERT-CNN. Thông điệp cốt lõi là tính hướng-đối-tượng thực sự hữu ích, nhưng nó phải được thích ứng theo các khuôn mẫu ngôn ngữ tiếng Việt, nơi đối tượng bị thù ghét thường được biểu đạt qua cách gọi nhóm suồng sã thay vì thực thể có tên.

### 7.2. Hướng phát triển

Trong tương lai, chúng tôi dự định mở rộng và tự động khai phá các ngân hàng cue đối tượng và tấn công, kèm thẩm định thủ công, để cải thiện độ phủ. Chúng tôi cũng muốn phân tích lỗi ở mức từng mẫu dựa trên giá trị cổng, độ phủ cue và độ tự tin của dự đoán, đồng thời bổ sung giám sát span, phát hiện mỉa mai và ngữ cảnh người dùng hay diễn ngôn. Một hướng khác là mở rộng từ thiết lập nhị phân sang đa nhãn hoặc đa đối tượng, kết nối với tinh thần của ViTHSD. Cuối cùng, hiệu chỉnh xác suất có thể giúp ngưỡng quyết định ổn định hơn khi triển khai liên miền.

## Tài liệu tham khảo

> *(Cần kiểm tra và hoàn thiện thông tin thư mục đầy đủ trước khi nộp; dùng `\bibliography{}` với `acl_natbib` hoặc nhập tệp `.bib`. Danh sách tối thiểu:)*

- **AmpleHate** — "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection" *(điền tác giả, hội nghị, năm)*.
- **ViTHSD** — "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts" *(điền chi tiết)*.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts" *(kiểm tra hội nghị)*.
- **VOZ-HSD** — bộ dữ liệu `tarudesu/VOZ-HSD` trên HuggingFace *(điền trích dẫn/URL)*.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*.
- **BERT** — Devlin và cộng sự (2019). *NAACL*.
- **fastText** — Bojanowski và cộng sự (2017). *TACL*.
- **Supervised Contrastive Learning** — Khosla và cộng sự (2020). *NeurIPS* *(nếu áp dụng)*.
- **NER tiếng Việt** — `NlpHUST/ner-vietnamese-electra-base` *(điền trích dẫn/URL)*.

---

### Phụ lục: Danh sách kiểm tra khi chuyển sang LaTeX ACL

- [ ] Tải mẫu ACL (`acl_latex.tex`, `acl.sty`, `acl_natbib.bst`) từ Overleaf.
- [ ] Đổi mỗi tiêu đề `##`/`###` sang `\section`/`\subsection`; bảng Markdown sang `tabular` + `booktabs`.
- [ ] Đưa các công thức `$...$` và `$$...$$` vào môi trường `equation`/`align`.
- [ ] Tạo Hình 1 (phân bố lớp) và Hình 2 (kiến trúc — **vẽ mới**); Hình 3–5 dùng các PNG sẵn có trong `notebooks/.../output/`.
- [ ] In đậm số tốt nhất trong Bảng 3–4 và thêm dòng Δ.
- [ ] Viết Abstract (~180 từ) sau cùng; thêm mục `\section*{Limitations}` (ngân hàng cue cố định, độ nhạy ngưỡng, thù ghét ẩn) — ACL yêu cầu mục Limitations.
- [ ] Hoàn thiện danh mục tài liệu tham khảo, dùng `\citep`/`\citet`.
- [ ] Đảm bảo phần nội dung không quá 8 trang theo quy định ACL.
</content>
