# Hướng dẫn toàn bộ project `sim_fas_cf` (package `simfas`)

Tài liệu này mô tả mọi thứ đã làm trong project, gồm mô hình hệ thống, môi trường RL, các mạng nơ-ron, toàn bộ hyperparameter, cách train, đánh giá, inference, và lý do của từng quyết định.

- Mọi con số lấy từ `configs/default.yaml` và từ chính code, tính ngày 2026-10-23. Ngày 2026-10-31 đã cập nhật các mục 4.1, 5, 6.3, 6.6 và 12 cho khớp với YAML và code hiện tại.
- Hai tài liệu đi kèm:
  - `docs/ASSUMPTIONS.md`: danh sách giả định và các lỗi tìm thấy trong bản thảo.
  - `docs/EQUATION_TO_CODE.md`: bảng nối từng phương trình với hàm và test tương ứng.

---
Thứ tự đọc: geometry.py -> fas.py -> channels.py -> sim_transfer.py -> association.py -> phy.py -> reward.py
Phase 1: channels.py + sim_transfer.py
Phase 2: association.py + phy.py + reward.py 
Phase 3: env.py + action_maps.py
Phase 4: mappo/ 
  + distribution.py 
  + actor.py 
  + trainer.py 
  + training.py, evaluate.py 

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| Mục tiêu | Cài đặt đúng mô hình trong bản thảo `demo_test.pdf` (SIM ở AP + FAS 2-D ở UE, cell-free mMIMO) và so sánh khả năng hội tụ của **MAPPO (đề xuất)**, **PPO**, **DDPG**, **TD3** và **Random** |
| Vị trí | `/ESSD2/ttmanh/manh_2/FA_SIM/sim_fas_cf/` |
| Quan hệ với code cũ | `FA_SIM/sim_fas_mappo` (bản dùng RZF) **được giữ nguyên**. Code mới thay RZF bằng việc gán anten cho UE theo AGA, đúng như paper. Một số module đã kiểm chứng (SIM Rayleigh–Sommerfeld, tương quan j0, MMSE, sampler cổng, PPO) được chuyển sang và sửa lại |
| Môi trường Python | conda env `tuanmanh` (Python 3.11, torch 2.11 + CUDA 12.8) |
| Kiểm thử | `python -m pytest -q`: **129 test** (2026-10-31). Phần vật lý được so với code tham chiếu viết bằng vòng lặp theo từng phương trình |
| Spec gốc | `CLAUDE_CODE_SIM_FAS_CF_MIMO_IMPLEMENTATION_SPEC.md`, do bạn cung cấp |

### Luồng xử lý một bước quyết định
```
reset (đầu episode): vị trí UE -> path loss + shadowing -> kênh H̄ (giữ cố định T bước) -> AGA (F) -> AP điều khiển FAS
                     -> cấu hình ban đầu (P/U mỗi feed, pha ngẫu nhiên, cổng ngẫu nhiên)
mỗi bước t:  observation cục bộ o_l (mỗi AP) ──actor──> action latent ──map──> (p, φ, cổng)
             ──env──> H_eff = A_kᴴ H̄ B(φ) Q1 -> d_{k,i} = Σ_l H_eff P_l f_{l,i} -> MMSE SINR -> R_k -> reward
sau T = 20 bước: episode kết thúc (terminal) -> reset kịch bản mới
```

---

## 2. Cấu trúc thư mục

| Đường dẫn | Vai trò |
|---|---|
| `configs/default.yaml` | Cấu hình dùng cho benchmark | *
| `configs/smoke.yaml` | Cấu hình rất nhỏ, chỉ dùng cho test |
| `src/simfas/config.py` | Dataclass cấu hình, kiểm tra hợp lệ (LU ≥ K, 1 ≤ n_k ≤ N_p…), tính kích thước đầu vào |
| `src/simfas/geometry.py` | Vị trí AP/feed/UE, khoảng cách, chọn AP điều khiển FAS (AP gần nhất) | * 
| `src/simfas/fas.py` | Tọa độ cổng (Eq. 1), chọn cổng, multi-hot, cổng ngẫu nhiên, top-k | *
| `src/simfas/channels.py` | Tương quan j0 (Eq. 2–3), path loss, shadowing, kênh H̄ (Eq. 8) | *
| `src/simfas/sim_transfer.py` | Hệ số Rayleigh–Sommerfeld (Eq. 4–5), ma trận Q1/Qinter, cascade B_l (Eq. 6–7) | *
| `src/simfas/association.py` | AGA: gán feed cho UE (Sec. III-B) | *
| `src/simfas/phy.py` | Kênh hiệu dụng (Eq. 9), d_{k,i} (Eq. 15), MMSE SINR (Eq. 17–20) | *
| `src/simfas/reward.py` | SE, phần hụt QoS, reward có phạt, các chỉ số | *
| `src/simfas/env.py` | Môi trường chạy song song E bản (batched): reset, step, observation, global state | * 
| `src/simfas/action_maps.py` | Ánh xạ latent → công suất/pha/cổng | *
| `src/simfas/mappo/` | Actor lai (Gaussian + Plackett–Luce), critic, trainer MAPPO/PPO | * 
| `src/simfas/offpolicy/` | Trainer DDPG/TD3, replay buffer | *
| `src/simfas/training.py` | Vòng lặp train dùng chung, log `episodes.csv`/`metrics.csv`, lưu model tốt nhất | *
| `src/simfas/evaluate.py` | Đánh giá trên các kịch bản giữ riêng (held-out), nạp model để inference | *
| `src/simfas/baselines.py` | Random, và policy giữ nguyên cấu hình ban đầu | *
| `scripts/train.py` | Train một thuật toán với một seed | * 
| `scripts/inference.py` | Chạy model đã lưu | *
| `scripts/plot_sumrate.py` | **Hình hội tụ**: sum rate theo steps, 5 đường | *
| `scripts/plot_convergence.py` | Hình 4 panel từ eval held-out (phụ) |
| `scripts/run_benchmark.sh`, `scripts/launch_detached.sh` | Chạy benchmark, tách khỏi phiên SSH |

---

## 3. Mô hình hệ thống và tham số mặc định

Cột **Nguồn** cho biết giá trị lấy từ paper/spec hay do tôi chọn. "Giả định" nghĩa là paper chưa cố định giá trị này, có thể đổi trong YAML.

### 3.1 Mạng và phần cứng
| Ký hiệu | YAML | Giá trị | Ý nghĩa | Nguồn |
|---|---|---|---|---|
| L | `system.L` | 3 | Số AP có SIM | spec (minh họa) |
| U | `system.U` | 4 | Số feed (anten) mỗi AP | spec |
| K | `system.K` | 4 | Số UE | spec |
| M | `system.M` | 2 | Số lớp SIM | spec |
| N | `system.N` | 16 (lưới 4×4) | Số meta-atom mỗi lớp | spec |
| N_x × N_y = N_p | `Nx, Ny` | 4 × 4 = 16 | Số cổng ứng viên của FAS | spec |
| n_k | `nk` | [2, 2, 2, 2] | Số cổng bật đồng thời mỗi UE | spec |
| W_x, W_y | `Wx, Wy` | 1 λ × 1 λ | Kích thước khẩu độ FAS → khoảng cách cổng λ/3 | spec |
| f_c | `carrier_ghz` | 28 GHz (λ = 1.07 cm) | Tần số sóng mang | spec |
| P_l^max | `p_max_dbm` | 20 dBm = 0.1 W, mỗi AP | Ngân sách công suất | spec |
| σ² | `noise_dbm` | −90 dBm = 10⁻¹² W | Nhiễu trên mỗi RF chain đang bật | spec |
| R_k^min | `min_se` | 0.5 bps/Hz | QoS tối thiểu | spec |
| λ_QoS | `env.qos_penalty` | 5.0 | Hệ số phạt QoS trong reward | spec |

Ràng buộc kiểm tra khi nạp cấu hình: **L·U ≥ K** (ở đây 12 ≥ 4, nên mỗi UE nhận trung bình 3 feed) và 1 ≤ n_k ≤ N_p.

### 3.2 Hình học (giả định)
| YAML | Giá trị | Ý nghĩa |
|---|---|---|
| `area_m` | 100 × 100 m | Vùng triển khai. 3 AP đặt theo lưới đều |
| `ap_height_m`, `ue_height_m` | 10 m, 1.5 m | Độ cao |
| `ue_margin_m` | 5 m | UE phân bố đều, cách biên 5 m |
| `env.reset_geometry_each_episode` | true | Mỗi episode bốc vị trí UE mới |

### 3.3 Kênh truyền (Eq. 2, 8)
- **Large-scale (giả định):** PL(dB) = 32.4 + 20·log₁₀(28) + 31.9·log₁₀(d) (mô hình UMi NLOS rút gọn của 3GPP TR 38.901), cộng shadowing log-normal 8.2 dB độc lập cho từng cặp AP–UE. Có thể chọn `three_slope` (Ngo 2017) thay thế.
- **Small-scale:** H̄_{l,k} = √β_{l,k} · J^{1/2} G R^{1/2}, với G ~ CN(0,1).
  - J là tương quan giữa các cổng FAS, dùng hàm Bessel cầu j0(2πd/λ) theo đúng Eq. 2.
  - R là tương quan giữa các atom ở lớp ra của SIM. Paper chưa cho công thức, nên mặc định **R = I** (có tùy chọn `sim_corr_model: j0`).
- **Kênh cố định trong cả episode** (T = 20 bước), đúng câu "fixed channel realization" trong paper. Hết episode thì bốc kênh mới.
- **CSI hoàn hảo** từ lớp ra của SIM tới mọi cổng ứng viên; không mô hình chi phí ước lượng kênh.

### 3.4 SIM (Eq. 4–7)
| YAML | Giá trị | Ý nghĩa |
|---|---|---|
| `meta_atom_spacing_lambda` | 0.5 | Khoảng cách atom d_a = λ/2, diện tích S = d_a² |
| `sim_interlayer_spacing_lambda` | 1.0 | Khoảng cách giữa hai lớp |
| `feed_to_first_layer_spacing_lambda` | 1.0 | Khoảng cách từ feed tới lớp 1 |
| `feed_spacing_lambda` | 0.5 | Các feed xếp thành ULA λ/2 dọc trục x (Eq. 5) |

- Q1 có kích thước 16×4 (feed → lớp 1), Qinter 16×16 (lớp → lớp), tính bằng Rayleigh–Sommerfeld. Mọi AP dùng chung hình học SIM.
- B_l = Φ_M Q Φ_{M−1} … Q Φ_1, và T_l = B_l Q1.

### 3.5 AGA: gán feed cho UE (Sec. III-B)
1. Lặp lại thao tác: chọn cặp (feed còn trống, UE chưa được phục vụ) có khoảng cách 3-D nhỏ nhất trên toàn mạng, cho đến khi mọi UE có ít nhất 1 feed.
2. Theo Algorithm 1 (dòng 9–14) của Shi et al. (arXiv:2409.12870): ở mỗi AP, xếp UE theo khoảng cách tới AP (trung bình qua các feed). u′ feed còn trống, theo thứ tự feed, lần lượt gán cho u′ UE đứng đầu, mỗi feed một UE (quay vòng nếu u′ > K). Feed của một AP vì thế được rải cho nhiều UE thay vì dồn hết cho UE gần nhất.

Ở bước 1, khi khoảng cách bằng nhau, thứ tự ưu tiên là (khoảng cách, l, u, k). Ma trận F giữ cố định trong cả episode.

**AP điều khiển FAS** của UE k là AP gần UE đó nhất **trong số các AP đang phục vụ UE đó** (có feed gán cho UE theo F). Mỗi AP chỉ đưa ra cổng cho các UE mà nó điều khiển; đề xuất cho UE khác không được bốc (−1) và không tính vào log-prob. Nếu nhiều AP cùng gần nhất, UE được giao cho AP đang quản lý ít UE nhất (tính các UE không hòa trước, rồi xét lần lượt các UE hòa theo chỉ số k); nếu vẫn hòa thì chọn AP có chỉ số nhỏ. Mọi AP vẫn cùng truyền (tín hiệu của AP vẫn gây nhiễu tới UE nó không phục vụ).

### 3.6 Truyền và thu (Eq. 9–22)
- H_eff[l,k] = A_kᴴ H̄_{l,k} B_l Q1, kích thước n_k × U.
- d_{k,i} = Σ_l H_eff[l,k] · diag(√p_l) · f_{l,i}. Các đóng góp từ các AP được **cộng phức** (cộng kết hợp).
- MMSE: γ_k = d_kkᴴ C_k⁻¹ d_kk, với C_k = Σ_{i≠k} d_{k,i} d_{k,i}ᴴ + σ² I. Khi d_kk = 0 thì γ = 0.
- R_k = log₂(1 + γ_k). **Sum rate** = Σ_k R_k.
- **Reward** = Σ_k R_k − λ_QoS · Σ_k [R_k^min − R_k]⁺. Đây là bài toán xấp xỉ bằng hàm phạt (P2), nên **không đảm bảo** mọi UE đạt QoS.

### 3.7 Mức tín hiệu điển hình (đã đo với cấu hình mặc định)
| Cấu hình | Sum rate cuối episode (bps/Hz) |
|---|---|
| Random (công suất Dirichlet, pha và cổng ngẫu nhiên) | ≈ 3.9 |
| Cấu hình ban đầu (P/U mỗi feed, pha và cổng ngẫu nhiên) | ≈ 4.2–5.0 |

Khi đặt pha ngẫu nhiên, SINR trung vị khoảng −5 dB và khoảng 45–50% UE đạt R_min. Đây là vùng mà tối ưu pha, công suất và cổng có tác động rõ.

---

## 4. Môi trường RL

### 4.1 Episode
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `env.episode_steps` (T) | 20 | Số bước tối ưu lặp trên **một** kênh cố định |
| Kết thúc | Terminal sau đúng T bước (bootstrap = 0) | Môi trường tự reset sang kịch bản mới |
| Cấu hình ban đầu | Công suất P_max/U mỗi feed, pha đều trong [0, 2π), cổng ngẫu nhiên | γ và ξ của bước "−1" là của cấu hình này |
| Số ngẫu nhiên | Sinh trên CPU rồi mới chuyển sang GPU | Kịch bản giống hệt nhau giữa CPU và CUDA |

> **Cập nhật 2026-10-30/31 (theo lựa chọn của bạn, bật trong `default.yaml`, `bench_L6U2K4.yaml` và `bench_L6U2K4_perhead.yaml`):** CSI chỉ gồm phần thực và ảo của H̄·√(P_max/σ²) (`channel_encoding: raw_snr`); pha SIM dạng φ/2π (`phase_encoding: raw`) ở cả observation và state; observation bỏ ξ (`obs_qos: false`); state bỏ D (`state_distance: false`). Khối cổng đang bật hiển thị cổng của **mọi UE mà AP phục vụ** (`obs_ports_scope: served`: lựa chọn cổng của một UE được chia sẻ cho mọi AP phục vụ nó), nên khối cổng không còn ngầm cho biết AP điều khiển UE nào, và mặt nạ điều khiển được bật lại (`obs_ctrl_mask: true`; từ 2026-10-31 `config.validate` báo lỗi nếu dùng `served` mà tắt mặt nạ). Thứ tự các khối do `config._blocks` quy định. Cũng bỏ cờ kết nối (`obs_conn_flag: false`), tắt độ lợi cổng sau SIM (`port_gain_obs: false`, tắt lại từ 2026-10-30) và dùng head cổng `flat` (model gốc). Với `bench_L6U2K4`: observation **3328** chiều [CSI 3200 | F_l 8 | p_l 2 | φ_l/2π 50 | cổng 64 (cổng: served) | mặt nạ điều khiển 4], state 19,632 chiều [CSI 19,200 | F 48 | p 12 | φ/2π 300 | cổng 64 | log(1+γ) 4 | ξ 4]; t/T đã bỏ từ 2026-11-04 (`state_time: false`, trước đó state 19,633 chiều). Với `default.yaml` (L = 3, U = 4): observation 3338 chiều, state 9882 chiều. Các bảng dưới mô tả layout cũ (giá trị mặc định trong code; `bench_L3U5K10.yaml` vẫn dùng layout này).

### 4.2 Observation của mỗi AP agent (cục bộ, Eq. 27/28): 2276 chiều
| Thành phần | Kích thước | Mã hóa |
|---|---|---|
| CSI của AP l tới các UE **mà AP này phục vụ** (có ≥ 1 feed trong F_l), mọi cổng | 4 × 529 = 2116 | Bằng 0 với UE mà AP không phục vụ. Với mỗi liên kết (l,k): phần thực và ảo của H̄/rms (2·16·16 = 512), SNR liên kết theo dB/10 (1), độ lợi từng cổng so với trung bình theo dB/10 (16) |
| F_l (gán feed–UE) | 4 × 4 = 16 | 0/1 |
| Công suất hiện tại p_l / P_max | 4 | |
| Pha hiện tại | 2 × 2·16 = 64 | cos, sin |
| Cổng đang bật của các UE mà AP này điều khiển | 4 × 16 = 64 | multi-hot, bằng 0 nếu AP không điều khiển UE đó |
| Mặt nạ điều khiển | 4 | 0/1 |
| Phần hụt QoS ξ_{t−1} / R_min | 4 | Chỉ của các UE mà AP phục vụ, bằng 0 với UE khác |
| Cờ kết nối | 4 | 0/1: AP có phục vụ UE k không (`env.obs_ue_scope: connected`) |

AP **không** thấy CSI của AP khác (`test_actor_obs_is_local_but_state_is_global`), và **không biết** CSI hay QoS của UE mà nó không phục vụ (`test_actor_obs_hides_unconnected_ues`), đúng với cell-free hướng người dùng: AP chỉ ước lượng kênh của các UE trong cụm của nó. Đặt `env.obs_ue_scope: all` để quay lại cách cũ (2272 chiều, thấy mọi UE) làm ablation. Vì sao mã hóa CSI theo hướng + độ lợi: độ lợi các UE chênh nhau hơn 30 dB, nên nếu chuẩn hóa thẳng H̄ thì thông tin của UE ở xa bị xóa. Cách mã hóa này không làm mất thông tin. Tùy chọn cũ `raw_snr` vẫn giữ để làm ablation.

### 4.3 Global state cho critic (Eq. 34, chỉ dùng khi train): 6685 chiều
D/D_max (12) | CSI của mọi AP (12 × 529 = 6348) | F (48) | công suất (12) | cos/sin pha (192) | cổng của mọi UE (64) | log(1+γ_{t−1}) (4) | ξ_{t−1} (4) | **t/T (1)**.

Thành phần t/T không có trong Eq. 34; ban đầu được thêm vào vì episode có độ dài cố định. Từ 2026-11-04 nó được tắt trong `default.yaml`, `bench_L6U2K4.yaml` và `bench_L6U2K4_perhead.yaml` (`env.state_time: false`): với δ = 0 thì V(s_t) = E[r_t | s_t], và actor MAPPO không thấy t. Mặc định trong code và `bench_L3U5K10.yaml` vẫn giữ t/T.

### 4.4 Chuẩn hóa
- Khối CSI được chuẩn hóa bằng mean/std chạy theo từng đặc trưng, chỉ cập nhật trong lúc train, đông cứng khi eval, và lưu kèm model. Sau đó cắt về [−10, 10].
- Các đặc trưng còn lại đã bị chặn sẵn (cos/sin, 0/1, chia cho P_max…).

---

## 5. Không gian hành động

### 5.1 MAPPO/PPO (policy ngẫu nhiên)
| Nhánh | Latent | Ánh xạ sang hành động vật lý |
|---|---|---|
| Công suất | z^p ∈ ℝ^{U+1} = ℝ⁵ (Gaussian) | p_{l,u} = P_max · softmax(z^p)_u với u = 1..U. **Phần thứ U+1 là công suất bỏ không**, nên Σ p ≤ P_max luôn đúng (`env.power_map: softmax`, mặc định trong code, Eq. 30) |
| Công suất, `env.power_map: linear` (các YAML benchmark, từ 2026-10-29) | z^p ∈ ℝ^U (Gaussian) | p_{l,u} = P_max · (clip(z^p_u, −1, 1) + 1)/2 cho từng feed; nếu tổng của AP vượt P_max thì nhân cả vector với P_max/Σ_u p_{l,u}. Vẫn được bỏ không công suất, Σ p ≤ P_max luôn đúng |
| Pha SIM | z^φ ∈ ℝ^{M·N} = ℝ³² (Gaussian) | φ = 2π · sigmoid(z^φ) |
| Cổng FAS | logits ∈ ℝ^{N_p} cho mỗi UE được điều khiển | Chọn tuần tự n_k cổng **không lặp lại** (Plackett–Luce): mỗi lượt che các cổng đã chọn |

- Log-prob chung (Eq. 32) = log N(z^p) + log N(z^φ) + Σ log P(cổng theo đúng thứ tự đã bốc).
- PPO tính tỉ số trên **latent đã lưu** và **chuỗi cổng đã lưu**, không bốc lại.
- MAPPO: mỗi agent có 5 + 32 chiều Gaussian, cộng tối đa 4 UE × 2 lượt chọn cổng. PPO tập trung: 15 + 96 chiều Gaussian, cộng 4 × 2 lượt chọn cổng.

### 5.2 DDPG/TD3 (policy tất định)
- Actor xuất a ∈ [−1, 1]^175, gồm 15 (công suất) + 96 (pha) + 64 (điểm của các cổng).
- Công suất: softmax(3·a) trên U+1 phần (`power_map: softmax`); với `power_map: linear` (các YAML benchmark) mức của từng feed lấy thẳng từ a: p = P_max(a+1)/2, co lại về P_max nếu tổng của AP vượt ngân sách.
- Pha: φ = π(a + 1).
- Cổng: chọn n_k cổng có điểm cao nhất. Critic học trên vector điểm liên tục.

---

## 6. Các thuật toán và mạng nơ-ron

### 6.1 MAPPO (đề xuất, CTDE)
| Thành phần | Chi tiết |
|---|---|
| Agent | 3 AP, **dùng chung một actor** (`ppo.shared_actor: true`; đặt false để mỗi AP có actor riêng) |
| Actor | MLP 2276 → 256 → 256 (tanh), 3 head: mean công suất (5), mean pha (32), logits cổng. log-std là tham số học được, khởi tạo −0.5 (std ≈ 0.61) |
| Head cổng (`ppo.fas_head: per_ue`) | Logits 16 cổng của UE k = MLP nhỏ dùng chung cho mọi UE, đầu vào [đầu ra thân mạng (256) ‖ đặc trưng của riêng UE k: CSI liên kết (l,k) (529), F_l[:, k] (4), cổng đang bật của k (16), cờ điều khiển, ξ_k, cờ kết nối] → 128 (tanh) → 16. Vị trí đặc trưng do `config.ue_feature_index` chỉ ra. PPO tập trung dùng cùng cách, với đặc trưng của UE k lấy từ global state. `flat` = head tuyến tính 4×16 cũ (ablation) |
| Số tham số actor | 673,674 |
| Critic | Tập trung, V(s) với MLP 6685 → 256 → 256 → 1 (tanh), **1,777,665 tham số**. Chỉ dùng khi train |
| Advantage | Một GAE cho mỗi transition, mọi agent dùng chung advantage (chuẩn hóa theo minibatch). Loss trung bình qua các agent |

### 6.2 PPO (tập trung, một agent)
Actor quan sát **global state** (6685 chiều) và xuất hành động cho cả 3 AP: 15 chiều công suất, 96 chiều pha, cổng cho cả 4 UE. Có **1,822,494 tham số**. Critic và hyperparameter giống MAPPO.

### 6.3 Hyperparameter của MAPPO/PPO (`ppo.*`)
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `discount` (δ) | **0.0** | Xem mục 8. Paper để δ ∈ [0,1); `config.validate` báo lỗi nếu δ ngoài [0, 1) (cả `offpolicy.discount`) |
| `gae_lambda` | 0.95 | Không còn tác dụng khi δ = 0. Phải thuộc [0, 1] |
| `clip` (ε) | 0.2 | Phải > 0 (Eq. 37) |
| `ppo_epochs` | 5 | Tối đa |
| `target_kl` | **0.02** | Dừng các epoch còn lại khi KL trung bình của epoch vượt ngưỡng này |
| `num_envs` × `rollout_steps` | 32 × 64 = 2048 transition mỗi lần update | |
| `minibatch_size` | 512 | Phải ≥ 2. Nếu minibatch cuối chỉ còn 1 mẫu (num_envs × rollout_steps chia minibatch_size dư 1) thì nó được gộp vào minibatch trước (độ lệch chuẩn của advantage không xác định với 1 mẫu); các cách chia khác giữ nguyên. 2048/512 không có phần dư |
| `actor_lr`, `critic_lr` | 3e-4, 3e-4 (Adam, eps 1e-5) | `lr_decay: none` |
| `entropy_coef_power / phase / fas` | 0 / 0 / 0.01 | Hệ số entropy riêng cho từng head |
| `value_coef`, `value_norm` | 0.5, true | Critic học trên return đã chuẩn hóa |
| `max_grad_norm` | 0.5 | |
| `init_log_std` | −0.5 | |
| `obs_clip` | 10 | |
| Ablation | `learn_power`, `learn_phase`, `fas_mode: learned / fixed_random / strongest / strongest_new / strongest_local` | Xem mục 6.6 |

### 6.4 DDPG và TD3 (một agent tập trung trên global state)
| Thành phần | Chi tiết |
|---|---|
| Actor | MLP 6685 → 256 → 256 → 175 (ReLU), tanh ở đầu ra. Lớp cuối khởi tạo đều trong ±3e-3. **1,822,383 tham số** |
| Critic | Q(s, a): MLP (6685 + 175) → 256 → 256 → 1 (ReLU), **1,822,465 tham số**. TD3 có 2 critic |
| Replay | 100,000 transition, lưu state thô; state kế tiếp là hàng kế tiếp của cùng env |

| Tham số (`offpolicy.*`) | Giá trị |
|---|---|
| `num_envs` | 4 |
| `discount` | **0.0** |
| `actor_lr`, `critic_lr` | 1e-4, 3e-4 |
| `batch_size` | 256 |
| `buffer_size` | 100,000 |
| `warmup_steps` | 5,000 (hành động ngẫu nhiên đều) |
| `gradient_steps` | 4 cho mỗi bước vector, tức 1 bước gradient cho mỗi transition (UTD = 1) |
| `tau` | 0.005 |
| `expl_noise` | 0.1 (Gaussian) |
| `target_noise`, `noise_clip` | 0.2, 0.5 (chỉ TD3) |
| `policy_delay` | 2 (TD3); DDPG = 1 |
| `power_scale` | 3.0 |
| `reward_scale` | 0.1 |
| `max_grad_norm` | 10 |

### 6.5 Random
- **Random:** mỗi bước chọn công suất theo Dirichlet(1) trên U+1 phần, pha đều, cổng ngẫu nhiên không lặp. Không học. Trong benchmark, Random cũng chạy đủ 1M steps để có đường cong.

### 6.6 Các tùy chọn thêm sau 2026-10-23 (cập nhật 2026-10-31)
Mọi tùy chọn đều có giá trị mặc định trong code giữ hành vi cũ, được kiểm tra trong `config.validate` và ghi trong `docs/ASSUMPTIONS.md`.

| Khóa YAML | Mặc định trong code | Trong YAML benchmark | Ý nghĩa |
|---|---|---|---|
| `env.power_map` | `softmax` | `linear` (`default`, `bench_L6U2K4`, `_perhead`) | Ánh xạ công suất, xem mục 5.1 và 5.2 |
| `env.obs_ports_scope` | `controlled` | `served` | AP thấy cổng đang bật của các UE nó điều khiển (`controlled`, Eq. 28) hay của mọi UE nó phục vụ (`served`). `served` bắt buộc `env.obs_ctrl_mask: true` |
| `ppo.fas_mode: strongest_new` | (`learned`) | biến thể `mappo_spn`, `ppo_spn` | Mỗi UE bật n_k cổng có công suất tín hiệu mong muốn lớn nhất, đo qua p, φ **mới** của bước t (đo sau khi cấu hình SIM). Cổng không học |
| `ppo.fas_mode: strongest_local` | (`learned`) | biến thể `mappo_spl` | AP điều khiển FAS chọn cổng chỉ từ phần đóng góp **của chính nó** (CSI của nó, p và φ mới của nó ở bước t), không cần thông tin của AP khác |
| `ppo.head_stop_freeze` | `false` | `true` (`_perhead`) | Khi dùng `head_target_kl`: một head đã dừng thì thân mạng dùng chung và các head đã dừng không được cập nhật nữa trong lần update đó (gradient đặt None, nên Adam không kéo chúng đi) |
| `ppo.std_schedule`, `ppo.final_std` | `none`, 0.05 | `none` (bench, từ 2026-10-30) | `linear`: std của Gaussian công suất/pha không học, đặt trước mỗi rollout, giảm tuyến tính exp(init_log_std) → final_std theo số step. Bật trong bench ngày 2026-10-29, tắt lại ngày 2026-10-30 |
| `ppo.fas_stats_active_only` | `false` | không bật (bạn quyết định) | `true`: surrogate của head cổng (khi `per_head_clip`), entropy FAS, `approx_kl_fas`, `clipfrac_fas` và phép so `head_target_kl` của head cổng lấy trung bình chỉ trên các agent có ít nhất 1 lượt chọn cổng. `false` (hành vi cũ): trung bình trên mọi agent, nên AP không điều khiển UE nào (46.9% agent trong bench) góp tỉ số 1 và KL 0, làm KL cổng loãng khoảng 2 lần |

Kiểm tra phạm vi trong `config.validate` (từ 2026-10-31): δ ∈ [0, 1) cho `ppo` và `offpolicy`, `gae_lambda` ∈ [0, 1], `clip` > 0, `ppo_epochs`, `rollout_steps`, `num_envs`, `offpolicy.num_envs` ≥ 1, `minibatch_size` ≥ 2 và `num_envs × rollout_steps` ≥ 2. Mọi YAML hiện có đều qua.

---

## 7. Huấn luyện, đánh giá, lưu model

| Mục | Giá trị / hành vi |
|---|---|
| `training.total_env_steps` | 1,000,000 steps = 50,000 episode (T = 20) |
| `training.eval_interval` | Mỗi 10,000 steps (và cả ở step 0) |
| `training.eval_episodes` | 128 episode giữ riêng (seed + 1,000,003), **giống hệt nhau cho mọi thuật toán** |
| Chỉ số đánh giá | Cấu hình **sau bước T** (cuối episode), policy tất định |
| `training.best_metric` | `final_reward`: reward có phạt ở cuối episode |
| `best_model.pt` | Model có `final_reward` eval cao nhất |
| `final_model.pt` | Model ở cuối quá trình train |

### File kết quả của mỗi run (`runs/bench/<algo>/seed1/`)
| File | Nội dung |
|---|---|
| `episodes.csv` | Một dòng cho mỗi episode train: `episode, env_steps, ep_sum_se` (sum rate trung bình T bước, **dùng để vẽ hình hội tụ**), `ep_reward, ep_qos_satisfied, return, final_sum_se, final_reward, final_qos_all, final_shortfall` |
| `metrics.csv` | Một dòng cho mỗi vòng lặp: loss, KL, clipfrac, std, và các cột `eval_*` |
| `best_model.pt`, `final_model.pt` | Trọng số actor, thống kê chuẩn hóa, config, kết quả eval |
| `final.json` | Kết quả eval của model tốt nhất (`best_eval`). Run Random lưu thêm kết quả của cấu hình ban đầu |
| `config.yaml`, `progress.json` | Cấu hình đã dùng; trạng thái và ETA |

---

## 8. Quá trình dò và lý do các quyết định

Dò với seed 1, mỗi lượt khoảng 100k–300k steps. Số liệu chỉ để so sánh tương đối.

| Vấn đề phát hiện | Bằng chứng | Quyết định |
|---|---|---|
| PPO cập nhật quá mạnh | KL ≈ 0.05–0.1, clipfrac ≈ 0.35–0.48, return lúc train giảm dần | `target_kl = 0.02`, 32 env, minibatch 512 |
| Kịch bản phụ thuộc thiết bị | Cùng seed nhưng eval ở bước 0 cho 6.15 trên GPU, 4.83 trên CPU | Sinh mọi số ngẫu nhiên của môi trường trên CPU |
| CSI thô khó học | MAPPO gần như không cải thiện so với lúc khởi tạo | Mã hóa hướng + độ lợi (dB) + độ lợi từng cổng |
| **δ = 0.99 không học được** | Eval đứng yên ≈ 5.0 sau 300k steps | **δ = 0**: eval lên 5.44 và vẫn đang tăng (δ = 0.5 đạt 5.26; PPO với δ = 0 đạt 5.57) |

**Vì sao δ = 0 hợp lý:** kênh cố định trong episode và hành động là giá trị tuyệt đối, nên r_t chỉ phụ thuộc a_t. Các reward tương lai chỉ thêm nhiễu khám phá vào advantage. Paper cần ghi rõ giá trị δ này và giải thích vai trò của các bước lặp: phản hồi γ, ξ giúp các agent chỉ quan sát cục bộ điều chỉnh dần qua các bước.

---

## 9. Benchmark đang chạy

| Mục | Giá trị |
|---|---|
| Lệnh | `STEPS=1000000 SEEDS="1" OUT=runs/bench GPU_ON=0 GPU_OFF=0 bash scripts/launch_detached.sh` |
| Các run | MAPPO, PPO, DDPG, TD3, Random; 1 seed; tất cả trên RTX 5090; tách khỏi phiên SSH (`setsid nohup`) |
| Tốc độ | MAPPO/PPO khoảng 1.1k steps/s (≈ 15 phút cho 1M). DDPG/TD3 khoảng 95 steps/s (≈ 3 giờ cho 1M) |
| Hình | `runs/bench/figures/convergence_sumrate.{png,pdf,csv}`, tự vẽ lại mỗi 5 phút |
| Cách đọc hình | Trục y là `ep_sum_se` của các episode train (policy đang khám phá), làm mượt 1000 episode (≈ 20k steps). Mọi thuật toán thấy cùng một chuỗi kịch bản, nên những chỗ cùng tụt là do kịch bản khó |

---

## 10. Cách dùng

```bash
cd /ESSD2/ttmanh/manh_2/FA_SIM/sim_fas_cf
conda activate tuanmanh

python -m pytest -q                                          # 129 test

# train 1 thuật toán (mappo | ppo | ddpg | td3 | random)
python scripts/train.py --algo mappo --seed 1 --steps 1000000 --out runs/bench
python scripts/train.py --algo mappo --set ppo.discount=0.5 env.qos_penalty=10   # đổi bất kỳ khóa YAML nào

# vẽ hình hội tụ
python scripts/plot_sumrate.py --root runs/bench              # --smooth 2000 để mượt hơn, --band để thêm dải ±std

# inference với model tốt nhất
python scripts/inference.py --model runs/bench/mappo/seed1/best_model.pt --episodes 200 --dump-episode

# theo dõi
cat runs/bench/*/seed1/progress.json
tail runs/bench/logs/mappo_seed1.log
```

Lưu ý khi inference:
- Actor của MAPPO chỉ dùng thông tin cục bộ, nên chạy được với số AP L khác lúc train.
- Đổi U, K, M, N hoặc lưới cổng thì kích thước mạng thay đổi, phải train lại.

### Inference khi đổi tham số (`scripts/sweep_inference.py`)
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/sweep_inference.py --root runs/bench --device cuda
```
- Dùng `best_model.pt` (train ở cấu hình mặc định) và đánh giá khi đổi **một** tham số: P_max, n_k, khẩu độ FAS, kích thước vùng, L, R_min.
- Mỗi điểm đánh giá trên 128 episode giữ riêng, cùng bộ kịch bản cho mọi phương pháp.
- Kết quả: `runs/bench/inference_sweeps/sweep_<tham số>.{png,pdf}` và `sweep_results.csv`. Đường chấm dọc trên hình đánh dấu cấu hình lúc train.
- Đổi L làm thay đổi global state, nên chỉ MAPPO và Random dùng được. Đổi U, K, M, N hoặc lưới cổng thì phải train lại.
- **Chạy DDPG/TD3 trên GPU**: khoảng 80–90% điểm cổng của actor bị bão hòa ở ±1, nên top-k phá thế hòa khác nhau giữa CPU và GPU. Trên GPU thì kết quả khớp với lúc train.

---

## 11. Giới hạn cần lưu ý khi viết paper

- SE là giá trị lý tưởng: CSI hoàn hảo, không tính chi phí đo kênh hay chọn cổng.
- QoS là ràng buộc mềm, nên phải báo cáo riêng tỉ lệ UE đạt QoS và mức hụt.
- Đường cong hội tụ là sum rate **lúc train**, với policy đang khám phá. Số liệu để so sánh cuối cùng nên lấy từ `inference.py` trên `best_model.pt`, với policy tất định trên kịch bản giữ riêng.
- Benchmark hiện tại chỉ có **1 seed**, nên chưa có dải sai số giữa các seed.
- MAPPO và PPO dùng δ = 0; paper cần ghi rõ giá trị này.
- Mục III-D (Q-PPO) trong paper còn trống, nên không có phần Q-PPO trong code.

---

## 12. Sai khác so với bản thảo (đã ghi nhận, không phải lỗi)

Theo bản review MAPPO ngày 2026-10-31 (`reviews/review_mappo_2026-10-31.md`). Đây là quyết định của bạn hoặc lựa chọn cài đặt đã ghi lại; bảng đầy đủ ở `docs/ASSUMPTIONS.md`, mục 4. Không sửa trong code hay YAML khi chưa có đồng ý của bạn.

| # | Bản thảo | Đang dùng (`default`, `bench_L6U2K4`, `bench_L6U2K4_perhead`) |
|---|---|---|
| 1 | Eq. 30: softmax U+1 phần, có phần bỏ không | `power_map: linear` (mục 5.1) |
| 2 | Eq. 35: δ ∈ [0, 1) | δ = 0 cho mọi thuật toán (mục 8) |
| 3 | Eq. 28: CSI tới **mọi** K UE | `obs_ue_scope: connected`: chỉ CSI của UE mà AP phục vụ. Lượt dò của chính repo nghiêng về `all` (12.67 so với 12.10), nhưng giữ `connected` theo lựa chọn |
| 4 | Eq. 28, Sec. III-E: ξ_{t−1} có trong observation | `obs_qos: false`: không có ξ trong observation của actor (chỉ có trong state) |
| 5 | Eq. 28: cổng của các UE được điều khiển | `obs_ports_scope: served` cộng mặt nạ điều khiển |
| 6 | AP điều khiển FAS = AP gần nhất | AP gần nhất **trong các AP phục vụ UE**; khác với AP gần nhất cho khoảng 1.6% UE |
| 7 | Eq. 34: state có D | Không có D, không có t/T (từ 2026-11-04), pha φ/2π, CSI `raw_snr`; head cổng của actor là một lớp tuyến tính K·N_p (`fas_head: flat`) |
| 8 | Eq. 31–32 | Likelihood của chuỗi Plackett–Luce **có thứ tự**; entropy FAS = tổng entropy có điều kiện dọc theo chuỗi đã bốc (kỳ vọng của nó bằng đúng entropy của chuỗi, `test_fas_entropy_surrogate_vs_exact_sequence_entropy`) |
| 9 | Eq. 36–38 | Một advantage chung cho mọi agent, chuẩn hóa theo minibatch |
| 10 | Eq. 37 | Clip trên tỉ số chung và dừng sớm theo KL chung (`target_kl: 0.02`) trong bench; cơ chế theo từng head chỉ dùng trong `_perhead` và biến thể `mappo_phc` |
| 11 | Sec. II–III | CSI lý tưởng; cấu hình ban đầu (P_max/U, pha và cổng ngẫu nhiên); T = 20. Lỗi phía paper: Eq. 33 lặp lại Eq. 32, đúng ra phải là reward Eq. 25 |
