import math
import numpy as np
import torch
#AP, UE positions and distances (all in meters)

def ap_positions(L: int, area) -> np.ndarray:
    #Chia bản đồ thành các hàng --> Chia các ô --> Đặt mỗi AP vào một ô 
    ax, ay = float(area[0]), float(area[1])
    n_rows = max(1, round(math.sqrt(L))) # Số hàng AP 
    counts = np.full(n_rows, L // n_rows, dtype=int) # Chia đều AP cho các hàng, Ex: L = 3 --> counts = [1, 1] 
    order = np.argsort(np.abs(np.arange(n_rows) - (n_rows - 1) / 2)) # Quyết định hàng nào nhận AP dư trước 
    # (n_rows - 1) / 2 : vị trí của hàng giữa 
    counts[order[: L % n_rows]] += 1 # Hàng được ưu tiên đầu tiên - có khoảng cách ngắn nhất đến centre
    pos = []
    for r, n in enumerate(counts):
        y = (r + 0.5) * ay / n_rows # Chia chiều cao thành n_rows dải bằng nhau, đặt hàng tâm dải 
        pos += [[(i + 0.5) * ax / n, y] for i in range(n)] # Chia chiều rộng thành n ô, mỗi AP đặt tâm ô 
    return np.asarray(pos, dtype=np.float64)

def planar_grid(nx: int, ny: int, spacing) -> np.ndarray: #Lưới 2D có tâm ở gốc 
    # Tạo nx x ny điểm cách đều, tâm lưới nằm ở (0, 0)
    sx, sy = (spacing, spacing) if np.isscalar(spacing) else spacing
    cx = (np.arange(nx) - (nx - 1) / 2) * sx # Xét tâm x = 0, chia tọa độ theo trục x
    cy = (np.arange(ny) - (ny - 1) / 2) * sy # Xét tâm y = 0, chia tọa độ theo trục y 
    xx, yy = np.meshgrid(cx, cy, indexing="xy") # Tạo ma trận (ny, nx)
    return np.stack([xx.ravel(), yy.ravel()], axis=1) # Duỗi ma trận theo hàng --> Lưới atoms

def feed_offsets(U: int, spacing: float) -> np.ndarray:
    return (np.arange(1, U + 1) - (U + 1) / 2) * spacing # Vị trí 4 antennas của các APs

def sample_ue_positions(E: int, K: int, area, margin: float, generator, device) -> torch.Tensor:
    # Vị trí UE ngẫu nhiên 
    # Bốc K UE cho mỗi kịch bản trong E kịch bản. Các UE độc lập, phân bố đều 
    lo = torch.tensor([margin, margin], dtype=torch.float64, device=device) #low : biên dưới - margin = 5 
    hi = torch.tensor([area[0] - margin, area[1] - margin], dtype=torch.float64, device=device) #high : biên trên - margin = 5
    # torch.rand --> cho số đều trong [0, 1), shape (E, K, 2)
    # Số sinh ra trên CPU xong chuyển qua cho GPU (.to(device)) --> Các thiết bị thấy cùng 1 kịch bản khi sinh số trên CPU 
    u = torch.rand((E, K, 2), generator=generator, device=generator.device, dtype=torch.float64).to(device)
    return lo + (hi - lo) * u #Cơ giãn tuyến tính từ [0, 1) sang [5, 95)

def ap_ue_distances(ap_xy: torch.Tensor, ue_xy: torch.Tensor, h_ap: float, h_ue: float) -> torch.Tensor:
    #Tính mọi cặp (AP, UE) của mọi kịch bản cùng lúc - không cần vòng lặp 
    dxy = ap_xy[None, :, None, :] - ue_xy[:, None, :, :]
    return torch.sqrt((dxy ** 2).sum(-1) + (h_ap - h_ue) ** 2) #Kết quả chạy D[0] (hàng là AP, cột là UE)

def feed_ue_distances(ap_xy: torch.Tensor, offsets: torch.Tensor, ue_xy: torch.Tensor, h_ap: float, h_ue: float):
    feed_x = ap_xy[:, None, 0] + offsets[None, :]                                
    dx = feed_x[None, :, :, None] - ue_xy[:, None, None, :, 0]
    dy = ap_xy[None, :, None, None, 1] - ue_xy[:, None, None, :, 1]
    return torch.sqrt(dx ** 2 + dy ** 2 + (h_ap - h_ue) ** 2)

def nearest_ap_control(D: torch.Tensor, serve: torch.Tensor | None = None):
    #nearest_ap_control(D): AP gần UE nhất chọn cổng FAS cho UE đó.
    #serve (E, L, K): AP l có feed phục vụ UE k (F.any(2)) -> chỉ xét các AP này (mỗi UE có >= 1 AP theo AGA).
    #Phá thế hòa: UE có nhiều AP cùng gần nhất -> giao cho AP đang quản lý ít UE nhất (hòa tiếp thì AP chỉ số nhỏ).
    if serve is not None:
        D = torch.where(serve, D, torch.full_like(D, float("inf")))
    E, L, K = D.shape # Kịch bản, AP, UE 
    is_min = D == D.min(dim=1, keepdim=True).values #(E, L, K) các AP gần nhất của mỗi UE                  
    n_cand = is_min.sum(1) #(E, K) số AP hòa nhau                                            
    idx = torch.arange(L, device=D.device).view(1, L, 1).expand_as(D) #Tensor chỉ số AP --> Đánh chỉ số AP
    ctrl = torch.where(is_min, idx, torch.full_like(idx, L)).min(dim=1).values # Ô nào là AP gần nhất thì giữ chỉ số AP/ còn lại đánh là L
    load = (is_min & (n_cand == 1)[:, None, :]).sum(-1)               #(E, L) số UE mỗi AP đã chắc chắn quản lý
    ar = torch.arange(E, device=D.device)
    for k in range(K): # Vòng lặp xét UE hòa                                                
        tied = n_cand[:, k] > 1
        key = torch.where(is_min[:, :, k], load * L + idx[:, :, 0], torch.full_like(load, L * (K + 1))) # Tải x L + chỉ số AP --> Đảm bảo tải chi phối hơn số AP
        pick = key.min(dim=1).values % L                              # ưu tiên tải nhỏ nhất, rồi chỉ số nhỏ nhất
        ctrl[:, k] = torch.where(tied, pick, ctrl[:, k]) # Đổi sang AP hợp lý hơn 
        load[ar, pick] += tied.long() #Cập nhật lại tải 
    mask = ctrl[:, None, :] == torch.arange(L, device=D.device).view(1, L, 1) # Đổi sang mask one-hot 
    return ctrl, mask
