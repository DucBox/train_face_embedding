# Hard-Negative-Aware PartialFC Sampling

Ghi lại bối cảnh, giả thuyết nguyên nhân và giải pháp cho vấn đề FNMR tăng vọt ở vùng
FMR thấp (1e-5 → 1e-6) khi benchmark trên NIST, cùng các tham số liên quan trong
`configs/`. Tài liệu này dành cho việc tra cứu lại lý do thiết kế, không phải hướng
dẫn sử dụng thông thường (xem `README.md` cho phần đó).

## 1. Hiện tượng quan sát

Model (ViT-L depth-36, loss CosFace qua `CombinedMarginLoss`, train trên
webface_synthetic + public + train_i qua PartialFC) cho kết quả tốt trên NIST ở
vùng FMR 10⁻¹ → 10⁻⁵ (FNMR thấp, rank cao), nhưng FNMR **tăng nhanh bất thường**
khi threshold đẩy lên vùng FMR 10⁻⁵ → 10⁻⁶ (security cao).

## 2. Giả thuyết nguyên nhân

Vùng FMR cực thấp bị quyết định bởi đúng phần "đuôi" khó nhất của phân phối score:
cặp đúng người nhưng ảnh khó (đuôi dưới của genuine score) và cặp khác người nhưng
trông rất giống nhau — song sinh, anh em, người giống nhau tự nhiên (đuôi trên của
impostor score). Hai giả thuyết được xem xét:

1. **Loss function** — `CombinedMarginLoss` với `margin_list=(1.0, 0.0, 0.4)` thực
   chất chạy theo công thức CosFace (margin trừ cố định `m=0.4`, scale `s=64`,
   xem `losses.py:42-56`). Margin này áp **đều cho mọi sample**, không phân biệt
   độ khó/chất lượng ảnh — khác với AdaFace (đã có sẵn trong `losses.py`, chỉ cần
   đổi `config.loss`), nơi margin co giãn theo norm embedding (proxy cho chất
   lượng ảnh).
2. **Sampling trong PartialFC** — `PartialFC_V2.sample()` (`partial_fc_v2.py`)
   chọn negative class **hoàn toàn random uniform** mỗi step (chỉ đảm bảo class
   positive trong batch luôn được giữ). Với `num_classes≈3.6M` và
   `sample_rate=0.3`, xác suất 2 identity "trông giống nhau" thật sự bị đặt cùng
   một step để model học margin riêng cho đúng cặp đó là rất thấp và **không hề
   tăng theo độ giống nhau thực tế** — random không biết ai giống ai.

Đã loại trừ: **label collision/split giữa các `.rec`** — đã audit và xác nhận ID
là global, liên tục giữa các file; một identity có ảnh rải ở nhiều file là bình
thường và được `ConcatDataset`/`PartialFC` xử lý đúng (xem thảo luận audit trong
lịch sử trao đổi, không lặp lại chi tiết ở đây).

**Đánh giá**: sampling random được cho là nguyên nhân chính (tạo lỗ hổng cấu trúc
thật ở đúng vùng đuôi cần quan tâm), loss tĩnh là nguyên nhân phụ (tối ưu trung
bình, không tối ưu đuôi).

## 3. Giải pháp đã chọn

Hai cơ chế độc lập, **đều opt-in qua config, mặc định tắt** để không ảnh hưởng
training hiện có:

### 3.1. Hard-negative-aware sampling (`config.hard_neg_mining`)

Thay vì chỉ random, thiên vị một phần ngân sách sample mỗi step về phía các class
"khó" (dễ nhầm), gồm:

- **Lazy kNN cache** (`hard_neg_topk`, `hard_neg_refresh_interval`): với mỗi class
  xuất hiện làm positive, tính top-k class center gần nhất (cosine similarity giữa
  class centers), lưu cache; chỉ tính lại khi cache cũ quá `hard_neg_refresh_interval`
  step. Rẻ vì chỉ tính cho các class thực sự xuất hiện, không phải toàn bộ 3.6M.
- **Confusion queue** (`hard_neg_queue_size`): FIFO ghi lại class nào vừa cho
  non-target logit cao nhất với một positive (= "suýt nhầm" thật, lấy từ tín hiệu
  forward đã có, không cần tính thêm) — bổ sung tín hiệu "động" cho kNN cache "tĩnh".
- **Ngân sách giới hạn** (`hard_neg_ratio`): hard-negative tối đa chiếm
  `hard_neg_ratio × num_sample` mỗi step, phần còn lại vẫn random — tránh model học
  lệch/oscillation giữa vài class hay bị đẩy qua đẩy lại.
- **Warmup** (`hard_neg_warmup_epoch`): tắt mining trong N epoch đầu, vì embedding
  space lúc đó chưa có cấu trúc — "hard neighbor" tính từ model chưa học gì là
  nhiễu, có hại hơn lợi.

Code: `partial_fc_v2.py` (`_refresh_neighbor_cache`, `_update_confusion_queue`,
nhánh mới trong `sample()`).

### 3.2. PartialFC sample_rate schedule (`config.sample_rate_schedule`)

Cho phép tăng `sample_rate` theo epoch (vd `[[0, 0.3], [60, 0.5], [67, 1.0]]`),
áp dụng qua `PartialFC_V2.set_sample_rate()` ở đầu mỗi epoch trong `train_v2.py`.
Ở `sample_rate=1.0`, không còn subsampling — mọi negative class đều tham gia mỗi
step (full softmax), đảm bảo 100% cặp khó được model "đối đầu" trực tiếp, đánh đổi
bằng compute/memory cao hơn (xem mục 5).

## 4. Lý do chọn giải pháp này (so với các phương án khác đã xem xét)

- **Tại sao không tính full similarity mỗi step để mining** (so toàn bộ embedding
  batch với toàn bộ class centers mỗi step): chi phí FLOPs đúng bằng full-FC
  forward — triệt tiêu lý do PartialFC tồn tại ở scale 3.6M class. Bị loại.
- **Tại sao chọn lazy/cached kNN thay vì luôn tính lại**: vì hard-negative thật sự
  cần quan tâm chỉ là các class đang xuất hiện trong batch (số lượng nhỏ mỗi
  step), tính theo nhu cầu (lazy) giúp chi phí gần như free so với tính toàn bộ.
- **Tại sao thêm cả confusion queue, không chỉ kNN cache**: kNN cache dựa trên
  *khoảng cách giữa class centers* — là một proxy tĩnh, có thể trễ so với hành vi
  thật của model trên dữ liệu cụ thể. Confusion queue tận dụng chính tín hiệu
  *forward* thật (non-target logit cao nhất) — rẻ (không tính thêm gì, đã có sẵn
  trong logits) và phản ánh đúng hành vi nhầm lẫn hiện tại của model, bổ sung cho
  phần kNN tĩnh.
- **Tại sao giới hạn ngân sách hard-negative (không để mining chiếm 100%)**: hard
  mining quá mạnh là pitfall đã biết (OHEM, CurricularFace đều phải kiểm soát) —
  dễ gây mất coverage tổng thể hoặc oscillation. Giữ phần lớn ngân sách vẫn random
  để không đánh đổi cái đang tốt (FMR 10⁻¹→10⁻⁵) để lấy cái đang xấu (đuôi).
- **Tại sao thêm warmup**: tránh mining dựa trên embedding gần-ngẫu-nhiên ở giai
  đoạn đầu — đảm bảo "hard" được xác định là hard thật, không phải nhiễu của model
  chưa học gì.
- **Tại sao thêm cả `sample_rate_schedule` riêng, không chỉ dựa vào mining**: đây
  là cơ chế "brute-force" bổ sung, không cần code thông minh — đảm bảo *chắc chắn*
  (không phụ thuộc heuristic nào) rằng giai đoạn fine-tune cuối, mọi negative đều
  được model thấy. Chi phí compute cao nên chỉ áp dụng ở vài epoch cuối (ước tính
  làm tăng tổng thời gian train toàn run dưới ~1%, xem mục 5).
- **Tại sao không đổi sang AdaFace ngay**: AdaFace giải quyết margin theo *chất
  lượng ảnh* (proxy: norm embedding), là một biến độc lập với vấn đề *sampling*.
  Đã có sẵn trong code (`config.loss = "adaface"`), có thể A/B test riêng, không
  trộn chung với thay đổi sampling để dễ tách bạch nguyên nhân khi đánh giá hiệu
  quả.
- **Tại sao mọi tham số đều qua `configs/`, không hardcode**: để A/B test được
  (so sánh on/off, đổi `hard_neg_ratio`/`hard_neg_warmup_epoch`/schedule mà không
  sửa code), và để không ảnh hưởng các lần train khác đang dùng cùng codebase.

## 5. Chi phí ước tính (lý thuyết, cần đo thật trước khi chạy full)

Với hardware thực tế (8× H200, `num_classes=3,666,172` → `num_local≈458,271`
class/GPU do chia theo `world_size=8`, không phải 64 như tên file config gợi ý):

| sample_rate | num_sample/GPU | logits tensor (fp32) ước tính |
|---|---|---|
| 0.3 | 137,481 | ~1.7 GB |
| 0.5 | 229,135 | ~2.8 GB |
| 1.0 | 458,271 | ~5.6 GB |

FC compute ước tính chỉ chiếm ~4-5% (ở 0.3) đến ~14-16% (ở 1.0) so với compute của
backbone ViT-L depth-36 — với schedule ramp dần (đa số epoch ở 0.3-0.5, full chỉ
vài epoch cuối), tổng thời gian train cả run tăng thêm ước tính dưới ~1%. Đây là
con số lý thuyết (FLOPs), cần verify bằng smoke-test thật (dùng
`config.rec = "synthetic"` để không cần data thật, chạy vài trăm step, đo
`torch.cuda.max_memory_allocated()` và thời gian/step) trước khi cam kết schedule
cho full training run.

## 6. Hạn chế còn tồn (chấp nhận, chưa fix)

- Mining chỉ hoạt động **trong phạm vi shard của từng rank** — `PartialFC_V2` chia
  class theo rank (`class_start`, `num_local`), nên hard-negative ở rank khác
  không được xét tới trừ khi thêm cross-rank communication (chưa làm, cần đánh
  giá riêng chi phí network trước khi đầu tư).
- `neighbor_cache`/`confusion_queue` không được lưu vào checkpoint
  (`persistent=False`) — sau resume, cache rebuild lại từ đầu (lazy, rẻ, không
  sai logic, chỉ là vài step đầu chưa có ứng viên hard).
- `module_partial_fc.global_step` (đếm theo micro-step, dùng để tính warmup) **đã
  được fix** để lưu/khôi phục đúng qua checkpoint (`hard_neg_step` trong
  `train_v2.py`) — tránh việc mỗi lần crash/resume lại vô tình tắt mining oan
  trong `hard_neg_warmup_epoch` tiếp theo.

## 7. Tham số liên quan (`configs/base.py`)

```python
config.sample_rate_schedule = None      # [[start_epoch, sample_rate], ...]
config.hard_neg_mining = False
config.hard_neg_ratio = 0.2
config.hard_neg_topk = 50
config.hard_neg_warmup_epoch = 10
config.hard_neg_refresh_interval = 2000
config.hard_neg_queue_size = 8192
```

Mặc định tất cả giữ hành vi training như trước khi có thay đổi này. Bật bằng cách
override trong config job cụ thể (xem ví dụ trong
`configs/wf42m_pfc03_40epoch_64gpu_vit_l.py`).
