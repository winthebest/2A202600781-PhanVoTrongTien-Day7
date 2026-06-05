# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phan Võ Trọng Tiển
**Nhóm:** B4
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai đoạn text có embedding gần cùng hướng, tức là ý nghĩa/ngữ cảnh rất giống nhau dù cách diễn đạt có thể khác. Giá trị càng gần 1 thì mức tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: "Hôm nay trời mưa to nên tôi mang áo mưa."
- Sentence B: "Vì trời đang mưa lớn, tôi đã mặc áo mưa khi ra ngoài."
- Tại sao tương đồng: Cả hai đều diễn tả cùng một tình huống và hành động ứng phó với trời mưa.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi đang học cách xây dựng hệ thống tìm kiếm văn bản."
- Sentence B: "Mèo của tôi thích nằm ngủ dưới nắng buổi sáng."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (kỹ thuật vs đời sống thú cưng), nên ngữ nghĩa xa nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector (mẫu ngữ nghĩa), nên ít bị ảnh hưởng bởi độ lớn vector. Với text embeddings, điều quan trọng là mức giống nhau về ý nghĩa hơn là độ dài vector tuyệt đối, vì vậy cosine thường ổn định và phù hợp hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50))`  
> `= ceil(9950 / 450)`  
> `= ceil(22.11...) = 23`
>  
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100:  
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, tức là số chunk tăng từ 23 lên 25. Overlap lớn hơn giúp giữ ngữ cảnh xuyên biên chunk tốt hơn, giảm nguy cơ mất thông tin nằm ở ranh giới giữa hai chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Du lịch & đặt tour Vinpearl (vé vui chơi, combo khách sạn, spa, golf)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn bộ tài liệu sản phẩm từ Vinpearl Booking vì đây là domain thực tế, có cấu trúc markdown đồng nhất (tiêu đề, URL, giá, các mục `##`), phù hợp để thử nghiệm chunking và metadata filter. Dữ liệu đa dạng (attraction, hotel combo, spa, golf) giúp benchmark queries phong phú và dễ đánh giá retrieval.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | VinWonders Nha Trang | booking.vinpearl.com | 6.198 | `product_type=attraction`, `destination=Nha Trang`, giá, URL |
| 2 | Grand World | booking.vinpearl.com | 10.289 | `product_type=entertainment`, `destination=Hà Nội`, giá, URL |
| 3 | Vinpearl Safari Phú Quốc | booking.vinpearl.com | 4.668 | `product_type=attraction`, `destination=Phú Quốc`, giá, URL |
| 4 | Aquafield Nha Trang | booking.vinpearl.com | 7.074 | `product_type=spa`, `destination=Nha Trang`, giá, URL |
| 5 | [Cần Thơ] 2N1Đ phòng Deluxe + Bữa sáng | booking.vinpearl.com | 4.995 | `product_type=hotel_combo`, `destination=Cần Thơ`, giá, URL |
| 6 | [Vinpearl Golf Phú Quốc] Voucher Tee time | booking.vinpearl.com | 2.847 | `product_type=golf_voucher`, `destination=Phú Quốc`, giá, URL |
| 7 | VinWonders Phú Quốc | booking.vinpearl.com | 7.031 | `product_type=attraction`, `destination=Phú Quốc`, giá, URL |
| 8 | Aquafield Ocean City Hà Nội | booking.vinpearl.com | 4.758 | `product_type=spa`, `destination=Hà Nội`, giá, URL |
| 9 | [HCM-Nha Trang] ROAM 2022 Combo 3N2Đ | booking.vinpearl.com | 11.655 | `product_type=hotel_combo`, `destination=Nha Trang`, giá, URL |
| 10 | [Grand World Phú Quốc] Vé Bảo Tàng Teddy Bear | booking.vinpearl.com | 3.225 | `product_type=attraction`, `destination=Phú Quốc`, giá, URL |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `title` | string | VinWonders Nha Trang | Nhận diện sản phẩm trong kết quả và prompt RAG |
| `url` | string | https://booking.vinpearl.com/... | Truy vết nguồn, hiển thị link đặt tour |
| `original_price` / `current_price` | string | 600.000 đ / 500.000 đ | Trả lời câu hỏi về giá và ưu đãi |
| `section` | string | Điều khoản chung | Giữ ngữ cảnh theo mục nội dung (mô tả, điều khoản, hướng dẫn) |
| `chunk_index` | int | 0, 1, 2... | Thứ tự chunk trong một tài liệu |
| `product_type` | string | spa, golf_voucher, hotel_combo | Lọc trước khi search (vd. chỉ golf hoặc chỉ combo) |
| `destination` | string | Nha Trang, Phú Quốc, Cần Thơ | Lọc theo điểm đến, tránh nhầm sản phẩm cùng loại |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên 3 tài liệu (kết quả từ `scripts/run_part3_benchmark.py`):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| VinWonders Nha Trang | FixedSizeChunker (`fixed_size`) | 35 | 177 | Thấp–TB — dễ cắt giữa đoạn mô tả dài |
| VinWonders Nha Trang | SentenceChunker (`by_sentences`) | 13 | 477 | Cao — giữ ranh giới câu |
| VinWonders Nha Trang | RecursiveChunker (`recursive`) | 19 | 326 | TB–Cao — tôn trọng đoạn xuống dòng |
| Grand World | FixedSizeChunker | 58 | 177 | Thấp — chunk nhỏ, mất ngữ cảnh mục dài |
| Grand World | SentenceChunker | 22 | 468 | Cao |
| Grand World | RecursiveChunker | 31 | 332 | TB–Cao |
| Vinpearl Safari Phú Quốc | FixedSizeChunker | 26 | 180 | Thấp–TB |
| Vinpearl Safari Phú Quốc | SentenceChunker | 10 | 467 | Cao |
| Vinpearl Safari Phú Quốc | RecursiveChunker | 14 | 334 | TB–Cao |

### Strategy Của Tôi

**Loại:** Custom — `SectionChunker` (chunk theo header markdown `##`)

**Mô tả cách hoạt động:**
> `SectionChunker` tách tài liệu theo ranh giới section (`## Mô tả`, `## Điều khoản`, `## Hướng dẫn sử dụng`...). Mỗi section là một đơn vị ngữ nghĩa tự nhiên trên trang sản phẩm Vinpearl. Nếu một section vượt `chunk_size` (800 ký tự), chunker fallback sang `RecursiveChunker` để không tạo chunk quá lớn. Metadata `section` được gán theo header tương ứng, khớp schema trong `src/data/schema.js`.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu nhóm có cấu trúc markdown đồng nhất: phần giá/URL ở đầu, nội dung chia theo mục rõ ràng. Câu hỏi benchmark thường nhắm vào một mục cụ thể (giá, điều khoản, dịch vụ bao gồm). Chunk theo section giúp retrieval trả về đúng block thông tin thay vì cắt giữa một đoạn mô tả dài như `FixedSizeChunker`.

**Code snippet (nếu custom):**
```python
class SectionChunker:
  def chunk(self, text: str) -> list[str]:
      parts = re.split(r"(?=^## )", text.strip(), flags=re.MULTILINE)
      # Gom theo section; section quá dài thì RecursiveChunker fallback
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| VinWonders Nha Trang | best baseline (SentenceChunker) | 13 | 477 | TB — đúng câu nhưng có thể trộn nhiều mục |
| VinWonders Nha Trang | **SectionChunker (của tôi)** | 6 | 1.033 | Cao — tách rõ Mô tả / Điều khoản / Hướng dẫn |
| Grand World | best baseline (SentenceChunker) | 22 | 468 | TB |
| Grand World | **SectionChunker (của tôi)** | 9 | 1.143 | Cao — giữ trọn section Mô tả lớn sau khi sub-chunk |
| Vinpearl Safari Phú Quốc | best baseline (SentenceChunker) | 10 | 467 | TB |
| Vinpearl Safari Phú Quốc | **SectionChunker (của tôi)** | 5 | 934 | Cao — Night Safari / Điều khoản tách riêng |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Phan Võ Trọng Tiển | SectionChunker + metadata filter | 8 | Khớp cấu trúc trang sản phẩm, filter theo `destination`/`product_type` | Section quá dài vẫn cần sub-chunk |
| Võ Tấn Trung | Sliding Window 1000/100 | 8/10 | Giữ được nhiều ngữ cảnh trong mỗi chunk, phù hợp với câu hỏi về giá, điều khoản, chính sách hoàn huỷ và hướng dẫn sử dụng | Chunk dài nên đôi khi kéo theo thông tin phụ không cần thiết |
| Đào Văn Tuân | ParentChildChunker (parent=800, child=200) | 8/10 (Hit@3: 4/5) | Child ngắn (200 ký tự) → vector chính xác; parent giữ đủ context cho LLM; khớp cấu trúc 2 lớp tự nhiên của data Vinpearl (`##` section → bullet) | Chunk count tăng gấp đôi so với baseline (62–81 chunks); Q1 thất bại do all-MiniLM-L6-v2 (tiếng Anh) embed "giá vé" không đủ mạnh |
 Nguyễn Bá Thành |Semantic Chunker | 10/10 (Hit@3: 100%) | Coherence cao nhất (0.782), các câu được gom theo ngữ nghĩa đồng nhất, tránh cắt đứt ngữ cảnh. | Tốc độ xử lý chậm hơn (11.86s) vì phải tính embedding cho từng câu đơn lẻ. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain Vinpearl, **SectionChunker + metadata filter** phù hợp nhất vì tài liệu có section chuẩn và câu hỏi thường theo mục (giá, điều khoản, combo bao gồm). SentenceChunker là baseline tốt thứ hai; FixedSizeChunker kém hơn khi mô tả sản phẩm dài vì dễ cắt giữa thông tin quan trọng.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Mình dùng regex `(?:(?<=[.!?])\s+)` để tách câu tại vị trí sau dấu kết thúc câu (`.`, `!`, `?`) và trước khoảng trắng/newline. Sau khi split, mình `strip()` từng câu và bỏ các phần rỗng để tránh chunk chứa whitespace thừa. Cuối cùng, các câu được gom theo cửa sổ `max_sentences_per_chunk`; nếu text rỗng hoặc chỉ có khoảng trắng thì trả về list rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk()` chỉ đóng vai trò điều phối: kiểm tra input rỗng, chuẩn hóa danh sách separator, gọi `_split`, rồi dọn kết quả bằng cách bỏ chunk rỗng. Trong `_split`, base case là: (1) đoạn hiện tại đã nhỏ hơn hoặc bằng `chunk_size` thì trả thẳng, hoặc (2) hết separator thì cắt cứng theo độ dài cố định. Nếu còn separator, thuật toán thử tách theo separator hiện tại, gom vào buffer sao cho không vượt ngưỡng; đoạn nào vẫn quá dài sẽ recurse với separator ưu tiên thấp hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mình thiết kế store theo 2 chế độ: ưu tiên ChromaDB nếu import/init được, nếu không thì fallback in-memory list để luôn chạy được trong môi trường lab. `add_documents()` tạo record chuẩn hóa (id nội bộ, content, metadata có `doc_id`, embedding) rồi thêm vào backend tương ứng. `search()` embed query và chấm điểm tương đồng bằng dot product; kết quả được sort giảm dần theo `score` và cắt `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> Với `search_with_filter`, mình lọc metadata trước rồi mới chạy similarity search trên tập candidate đã lọc để đảm bảo kết quả đúng ngữ cảnh (ví dụ theo `department`, `lang`). Nếu không có filter thì reuse `search()` để tránh lặp logic. `delete_document()` xóa toàn bộ chunk theo `doc_id` (ở metadata), trả về `True` nếu có dữ liệu bị xóa và `False` nếu không tìm thấy document cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> `answer()` thực hiện luồng RAG cơ bản: retrieve top-k chunk liên quan từ store, ghép các chunk thành phần `Context`, sau đó dựng prompt có instruction + context + question. Prompt được viết theo hướng “chỉ trả lời dựa trên context, thiếu dữ liệu thì nói không đủ thông tin” để giảm hallucination. Sau cùng, agent gọi `llm_fn(prompt)` và trả trực tiếp output của model.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Phương pháp: embed từng câu bằng `_mock_embed`, tính cosine similarity bằng `compute_similarity()`. Ngưỡng đánh giá: **high** nếu score ≥ 0.1, **low** nếu score < 0.1 (dự đoán được ghi **trước** khi chạy code).*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | VinWonders Nha Trang | VinWonders Nha Trang | high | 1.0000 | ✓ |
| 2 | Giá vé VinWonders Nha Trang là 500000 đồng | Giá hiện tại VinWonders Nha Trang là 500000 đồng | high | 0.0248 | ✗ |
| 3 | Giá vé VinWonders Nha Trang | Thời tiết Hà Nội hôm nay trời mưa nhiều | low | −0.0806 | ✓ |
| 4 | Night Safari tại Vinpearl Safari Phú Quốc | Safari ban đêm bằng xe điện khám phá động vật | high | 0.1449 | ✓ |
| 5 | Aquafield Nha Trang có 7 phòng xông hơi trị liệu | Aquafield có phòng Băng tuyết, Gỗ bách, Sương mây | high | 0.0400 | ✗ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là **Pair 2**: hai câu cùng nói về giá vé VinWonders Nha Trang nhưng chỉ đạt **0.0248** (gần như không tương đồng), trong khi Pair 4 — diễn đạt khác nhưng cùng chủ đề Night Safari — lại đạt **0.1449**. Điều này cho thấy `_mock_embed` sinh vector từ hash MD5 của chuỗi ký tự, **không mã hóa ngữ nghĩa**; chỉ câu giống hệt (Pair 1) mới có cosine = 1. Embedding thật (ví dụ `all-MiniLM-L6-v2`) học được mối quan hệ ngữ nghĩa giữa các cách diễn đạt khác nhau, còn mock chỉ phù hợp để test pipeline kỹ thuật — không dùng để đánh giá chất lượng retrieval theo nghĩa.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu? | Giá hiện tại **500.000 đ** (giá gốc 600.000 đ), theo trang VinWonders Nha Trang. |
| 2 | Aquafield Nha Trang có những phòng trị liệu xông hơi nào? | Có 7 phòng: Băng tuyết, Gỗ bách (Hinoki), Sương mây, Đá muối Himalaya, Bulgama, Than củi, Hoàng thổ — mỗi phòng có nhiệt độ/độ ẩm riêng. |
| 3 | Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì? | 01 đêm phòng Deluxe (2 người lớn + 2 trẻ dưới 4 tuổi), 01 bữa sáng, miễn phụ thu cuối tuần, thuế phí dịch vụ. |
| 4 | Voucher golf Sunrise áp dụng tee-time và ngày nào? *(cần filter `product_type=golf_voucher`)* | Sunrise: tee-time **trước 8:00**, nhóm từ 2 khách, **thứ 2–thứ 6**, không áp dụng ngày lễ 30/4, 01/05, 02/9. |
| 5 | Night Safari tại Vinpearl Safari Phú Quốc là gì? | Hành trình khám phá động vật về đêm bằng xe điện — trải nghiệm safari ban đêm duy nhất tại Việt Nam. |

### Kết Quả Của Tôi

*Nguồn: `output/benchmark_results.json` — `python scripts/run_part3_benchmark.py`. RAG: `SectionChunker(800)` + `EmbeddingStore` in-memory + `_mock_embed` + `KnowledgeBaseAgent` (`context_grounded_llm`).*

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu? | **Aquafield Nha Trang** — section *Tổng quan* (nói về 7 phòng trị liệu, không phải giá VinWonders) | 0.3131 | **Không** — sai sản phẩm | Không xác nhận được giá VinWonders — context retrieve không có dòng giá đúng sản phẩm |
| 2 | Aquafield Nha Trang có những phòng trị liệu xông hơi nào? | **Aquafield Nha Trang** — section *Tổng quan* (khu trị liệu, máy bán hàng — không liệt kê tên phòng) | 0.3301 | **Một phần** — đúng tài liệu, chunk chưa chứa danh sách 7 phòng | Chunk retrieve không liệt kê đủ 7 phòng Aquafield |
| 3 | Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì? | **Vinpearl Hotel Cần Thơ** — section *Tổng quan* (mô tả khách sạn, không phải mục Bao gồm) | 0.2110 | **Có trong top-3** — top-2 là section *Bao gồm* (score 0.1473) | **Đúng:** 01 đêm Deluxe, 01 bữa sáng, miễn phụ thu cuối tuần, thuế phí (từ chunk *Bao gồm* trong top-3) |
| 4 | Voucher golf Sunrise áp dụng tee-time và ngày nào? | **Vinpearl Golf Phú Quốc** — section *Điều khoản và Thời gian sử dụng* | 0.1024 | **Có** — đúng sản phẩm và section điều khoản | **Đúng:** Sunrise — tee-time trước 8:00, nhóm từ 2 khách, thứ 2–thứ 6, đến 31/10/2026 |
| 5 | Night Safari tại Vinpearl Safari Phú Quốc là gì? | **VinWonders Phú Quốc** — section *Tổng quan* (sai sản phẩm) | 0.2887 | **Không** — top-3 không có Safari | Agent thừa nhận không đủ thông tin Night Safari trong context retrieve |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **2** / 5 (query 3 và 4; query 2 chỉ đúng tài liệu một phần)

**Nhận xét nhanh:** Agent dùng `context_grounded_llm` — chỉ trả lời từ chunk đã retrieve, không hallucinate khi thiếu dữ liệu (query 1, 5). Query 3–4 cho thấy RAG hoạt động tốt khi retrieval + filter đúng. Query 1–2, 5 cho thấy **grounding quality** phụ thuộc retrieval: retrieve sai → agent cũng không trả lời đúng dù không bịa thêm.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Việc sử dụng metadata filter giúp cải thiện đáng kể độ chính xác khi tìm kiếm trong kho dữ liệu lớn, thay vì chỉ dựa vào vector similarity đơn thuần

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Nhóm khác đã sử dụng nhiều phương pháp khác nhau cho từng bài toán, tuy nhiên đa số đều lựa chọn agentic chunking làm core 

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Tôi sẽ làm sạch dữ liệu Markdown tốt hơn (xóa các ký tự đặc biệt) trước khi đưa vào store để embedding có độ chính xác cao hơ

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 8 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | 80 / 100 |
