# Review hard cases sau offline mining — bước 1–2–3

Tool này chỉ chuẩn bị index và kiểm tra nhãn. Không sửa RecordIO, không chạy model,
không tự xác nhận danh tính, và chưa tích hợp sampler/training.

## Đầu vào

```text
rec_out/
  train_crawl_000.rec + .idx
  ...
  train_crawl_031.rec + .idx
  train_public_clean.rec + .idx
  train_synthetic_clean.rec + .idx

hard_mining/
  file_prefix=train_crawl_000/*.parquet
  ...
  file_prefix=train_public_clean/*.parquet
  file_prefix=train_synthetic_clean/*.parquet
  hard_class_neighbors.csv
  hard_images_review.csv
```

Embeddings và CSV có thể ở hai thư mục khác nhau. Tool tự tìm các partition hiện
có, không hardcode 32 shards. Mỗi prefix phải có cả `.rec` và `.idx`.
CSV/Parquet phải thuộc cùng phiên bản `rec_out` và class mapping. Không truyền
`og_data/train` để đọc các record của `rec_out`.

Dependencies: Python >=3.9, `pyarrow`, `Pillow`. Không cần torch, GPU hoặc MXNet.
Reader dùng framing RecordIO/MXImageRecord theo dmlc `recordio.h` và MXNet
`recordio.py`; hỗ trợ scalar/array labels, header/no-header files và magic-word
continuation chunks. Chỉ đọc records được chọn xem, kiểm tra label thực trong
record so với metadata rồi decode JPEG/PNG bằng Pillow.

## Bước 1–2: lập index metadata và kiểm tra CSV

Chạy từ root `face_embed`, sau khi đặt script tại `tools/hard_case_review.py`:

```bash
python3 tools/hard_case_review.py index \
  --rec-dir /path/to/rec_out \
  --embeddings-dir /path/to/hard_mining \
  --hard-dir /path/to/hard_mining \
  --db /path/to/hard_review/index.sqlite
```

Chỉ đọc `identity`, `rec_idx`, `file_prefix`, không đọc vector embeddings.
SQLite chứa:

- `sources`: prefix và ID của source.
- `images`: `(source_id, rec_idx, identity)` cùng index tra identity → ảnh.
- `hard_neg`: unique unordered identity pairs; hai directed rows A→B/B→A gộp một.
- `hard_pos`: ảnh hard + label + genuine score.
- `meta`: source paths, counts, file inventory và hashes của hai CSV.

Metadata Parquet được đọc streaming mặc định 50.000 rows/batch. Dữ liệu nằm trên
disk; cần chỗ trống cho SQLite và indexes (có thể nhiều GB với hàng chục triệu
ảnh). Thời gian phụ thuộc SSD và số rows, không phải chỉ số Parquet files. SQLite
temporary indexes được spill xuống disk.

Tool báo lỗi nếu CSV tham chiếu class không tồn tại, image reference không tồn tại,
hoặc label không khớp Parquet. Khi lỗi build, temporary DB bị xóa; source data giữ
nguyên. DB có sẵn không bị ghi đè: muốn rebuild, chọn tên DB mới.

`index.sqlite.json` lưu summary của lần build. Index này giữ **RecordIO
references**, chưa tính `global_idx` của ConcatDataset. Khi viết sampler sau này,
cần map references theo thứ tự dataset loader thật, không dùng `rec_idx` như
global dataset index.

## Sau index: thống kê trước khi chọn ảnh review

```bash
python3 tools/hard_case_review.py stats \
  --db /path/to/hard_review/index.sqlite \
  --output-dir /path/to/hard_review/stats_001 \
  --top-identities 30
```

Không đọc ảnh hay embeddings, không thay đổi DB. Outputs: `summary.txt`,
`summary.json`, `top_negative_identities.csv`, `top_positive_identities.csv`,
`positive_by_source.csv`. Báo số unique negative pairs (gộp A→B/B→A), hard images,
affected identities, score histograms và độ tập trung lỗi theo identity.
Negative concentration là share của **hai endpoints/pair**, không phải share của
unique pairs. Identity xuất hiện nhiều là ưu tiên xem, chưa phải bằng chứng nhãn sai.
Thống kê trên pool lớn vẫn có thể mất thời gian và disk cho temporary SQL grouping.

Đọc stats trước để quyết định lượng và nhóm cần review. Review là audit theo mẫu,
không phải yêu cầu duyệt tay mọi dòng CSV.

## Bước 3: xuất một mẫu ảnh và trang review

Để audit theo nguồn và khoảng score thay vì random gộp, dùng sampling-plan JSON:

```bash
python3 tools/hard_case_review.py review \
  --db /path/to/hard_review/index.sqlite \
  --output-dir /path/to/hard_review/view_stratified_001 \
  --sampling-plan /path/to/review_plan.json \
  --images-per-class 4 --seed 2048
```

Plan là list các object có `name`, `kind` (`negative`/`positive`), `count`,
`score_min` inclusive, `score_max` exclusive và tùy chọn `source_glob` cho positive.
Ví dụ `{ "name":"crawl", "kind":"positive", "source_glob":"train_crawl_*",
"score_min":0.2, "score_max":0.4, "count":20 }`.
Tool reservoir-samples từng stratum với bộ nhớ bounded, dedup overlap, và ghi
eligible/requested/selected vào `manifest.json`. Khi dùng plan, các flags
negative-limit/positive-limit/random-each không được dùng. Quá trình sampling có
thể scan nhiều rows, nhưng không đọc embeddings/ảnh trước khi chọn xong.
Sample phân tầng này ưu tiên chẩn đoán; không lấy tỷ lệ label noise gộp không trọng
số làm ước lượng toàn pool vì các strata có sampling fractions khác nhau.

Nếu không có plan, dùng mode extreme + random như sau:

```bash
python3 tools/hard_case_review.py review \
  --db /path/to/hard_review/index.sqlite \
  --output-dir /path/to/hard_review/view_001 \
  --negative-limit 100 \
  --positive-limit 100 \
  --random-each 25 \
  --images-per-class 6 \
  --seed 2048
```

Ví dụ trên lấy 100 negative pairs score cao nhất + 25 random negative pairs,
100 positive images score thấp nhất + 25 random positive images; nếu pool nhỏ hơn
thì lấy phần có sẵn. Random sample không trùng phần extreme. Reference images
được lấy mẫu đều theo ảnh trong identity, không nhất thiết là medoid, không phải
ảnh đẹp nhất hoặc ảnh tạo score negative gốc. Số ảnh per identity và seed chỉnh
được; cùng seed/db cho cùng lựa chọn.

Outputs:

```text
view_001/
  index.html
  page_002.html ...       # 25 cases/trang mặc định
  sheets/*.jpg
  manifest.json          # case selection, references, scores, read/label errors
```

Viewer quét `.idx` một lần cho mỗi prefix cần dùng, chỉ giữ offsets của các ảnh
cần hiển thị. Không nạp cả `.idx` của file hàng chục triệu ảnh vào RAM. File `.rec`
được mở lazily với tối đa 4 handles; điều chỉnh `--max-open-files` nếu cần.

Mở viewer qua HTTP localhost để quyết định được lưu chung giữa các trang:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory /path/to/hard_review/view_001
```

Mở `http://127.0.0.1:8765/` trong browser. Không cần Internet hoặc upload ảnh.

### Quyết định mỗi case

- **Approve labels**: negative — hai identity thực sự khác người, các ảnh có nhãn
  đúng; positive — ảnh hard đúng người của identity reference.
- **Exclude**: xác định nhãn sai/identity trộn/case không phù hợp.
- **Unsure**: chưa thể xác nhận, tạm không đưa vào approved pool.
- Chưa bấm: `pending`, không nằm trong approved pool.

Nếu có read/label error hoặc positive chỉ có một ảnh, viewer chặn Approve. Nhãn
đúng theo metadata không chứng minh cùng người: người review vẫn phải xem ảnh.
Ảnh trong một identity có thể chưa được xem hết; Approve pair trên một sample
không tự chứng nhận mọi ảnh của hai identities.

### Tải kết quả

Các nút download xuất CSV từ decisions đã lưu trong browser:

- `review_decisions.csv`: tất cả cases trong viewer, có trạng thái kể cả pending.
- `approved_identity_pairs.csv`: class_a, class_b, score, status=keep.
- `approved_images.csv`: file_prefix, rec_idx, label, genuine_score, status=keep.
- `excluded_identity_pairs.csv`: các pair exclude/unsure.
- `excluded_images.csv`: các ảnh exclude/unsure.

Các file được browser tải về, không tự ghi ngược vào thư mục server. Lưu chúng
trong thư mục review của run. Approved CSV chỉ ghi những case cụ thể đã xem;
**không hiểu rằng phần CSV gốc chưa xuất hiện trong viewer đều được approve**.
Ở quy mô lớn, các decisions dùng để audit chất lượng/lọc các nhóm có lỗi: sau khi
đánh giá pool, sampler có thể dùng pool sau lọc với cap identity/edge. Không yêu cầu
duyệt tay toàn bộ pool, nhưng cũng không tự coi sampling audit là chứng nhận tất cả nhãn.
Quyết định lưu trong localStorage của cùng browser/origin; export trước khi đổi
browser, port, máy hoặc xóa browser data. Tạo output viewer mới nếu muốn lấy mẫu
mới; thư mục output không rỗng sẽ bị từ chối để tránh ảnh/manifest cũ lẫn run mới.

## Xác nhận và giới hạn

- Kiểm tra tự động: reference/label nhất quán, RecordIO decode đúng, lỗi đọc ảnh.
- Kiểm tra con người: danh tính thực, label noise, chất lượng pool.
- Không sửa threshold mining, không gọi các medoid candidates là benchmark FMR
  chuẩn, không tự remap/merge/delete identities.
- Chưa chạy trên dataset thật từ máy training. Đã chạy fixtures Parquet + RecordIO
  synthetic: sparse/zero-based record keys, scalar/array labels, continuation,
  CSV mismatch, symmetric dedup, pagination và errors.

Chạy tests từ root repo nếu đặt test trong `tools/tests/`:

```bash
python3 -m unittest discover -s tools/tests -p 'test_hard_case_review.py' -v
```

Sau stats và sampling review, công việc tiếp theo là mixed BatchSampler dùng pool đã lọc;
tool này chưa thay đổi training behavior.
