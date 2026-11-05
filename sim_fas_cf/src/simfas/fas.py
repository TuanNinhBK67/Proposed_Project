import numpy as np
import torch

def port_coordinates(Nx: int, Ny: int, Wx: float, Wy: float, wavelength: float, centered: bool = True) -> np.ndarray:
    #Tọa độ vật lý của 16 cổng trên mặt FAS --> Tính ma trận J 
    ax = np.arange(Nx) / (Nx - 1) * Wx * wavelength if Nx > 1 else np.zeros(1) #Eq (2) - (3)
    ay = np.arange(Ny) / (Ny - 1) * Wy * wavelength if Ny > 1 else np.zeros(1)
    if centered:
        ax, ay = ax - ax.mean(), ay - ay.mean() # Rời lưới về tâm khẩu độ --> Hình học đẹp hơn
    xx, yy = np.meshgrid(ax, ay, indexing="xy") # Tạo lưới 2-D rồi duỗi theo hàng 
    return np.stack([xx.ravel(), yy.ravel()], axis=1)

def slot_valid_mask(nk_list, n_max: int, device="cpu") -> torch.Tensor:
    # Tạo mặt nạ (K, n_max) cho biết ô nào thật  --> Mặt nạ ô thật/ ô thừa 
    nk = torch.as_tensor(nk_list, device=device) # Số cổng mỗi UE bật 
    return torch.arange(n_max, device=device)[None, :] < nk[:, None] # So sánh chỉ số ô với n_k: ô j là thật khi j <n_k

def validate_ports(ports: torch.Tensor, slot_valid: torch.Tensor, Np: int) -> None:
    # Kiểm tra một cấu hình cổng có hợp lệ không - báo lỗi nếu sai --> Kiểm tra action hợp lệ 
    if ports.dtype not in (torch.int64, torch.int32):
        raise TypeError("port indices must be integer tensors") #Kiểm tra kiểu dữ liệu số nguyên 
    sv = slot_valid.expand_as(ports) # Mở rộng thành (E, K, n_max) để khớp với ports 
    v = ports[sv] # Lấy ra chỉ số các ô thật 
    if v.numel() and (v.min() < 0 or v.max() >= Np): # Kiểm tra về phạm vi 
        raise ValueError("port indices must lie in 0..Np-1")
    if (ports[~sv] != -1).any(): # Ô thừa bằng -1 
        raise ValueError("unused port slots must hold -1")
    if (multi_hot(ports, slot_valid, Np) > 1).any(): # Đếm số lần mỗi cổng suất hiện. Nếu > 1 --> sai
        raise ValueError("active ports of a UE must be distinct")

def multi_hot(ports: torch.Tensor, slot_valid: torch.Tensor, Np: int) -> torch.Tensor:
    # Đổi danh sách chỉ số cổng (K, n_max) thành vector 0/1 dài Np cho mỗi UE --> Mã hóa cổng cho mạng nơ-ron
    # Biểu diễn các ô như một loạt công tắc đèn: 0 - tắt, 1 - bật -- VD: (32, 4, 2) --> (32, 4, 16)
    oh = torch.nn.functional.one_hot(ports.clamp_min(0), Np).to(torch.float32) # order-free encoding 
    return (oh * slot_valid.expand_as(ports)[..., None].float()).sum(-2) # Kích thước (K, Np) - dãy 0/1 dài Np

def selection_matrix(ports_k: torch.Tensor, Np: int) -> torch.Tensor:
    # Tạo ma trận chọn cổng A_k đúng như ký hiệu trong paper - Np x nk
    return torch.nn.functional.one_hot(ports_k, Np).T.to(torch.float64)

def gather_selected(Hbar: torch.Tensor, ports: torch.Tensor, slot_valid: torch.Tensor) -> torch.Tensor:
    E, L, K, Np, N = Hbar.shape # (E, L, K, Np, N) : Np cổng x N meta-atom  --> Tạo kênh A . H(SIM-FAS)
    n = ports.shape[-1] # n_max 
    idx = ports.clamp_min(0)[:, None, :, :, None].expand(E, L, K, n, N) # Chuẩn bị số cho torch.gather
    sel = torch.gather(Hbar, 3, idx) # Lấy theo chiều 3 --> (E, L, K, n, N)
    return sel * slot_valid[None, None, :, :, None].to(sel.real.dtype) # 

def random_ports(E: int, nk_list, Np: int, n_max: int, generator, device) -> torch.Tensor:
    # Bốc ngẫu nhiên n_k cổng không lặp bằng cách cho mỗi cổng một điểm ngẫu nhiên đều rồi lấy top-n_k
    # Cấu hình ngẫu nhiên 
    K = len(nk_list)
    scores = torch.rand((E, K, Np), generator=generator, device=generator.device)
    return topk_ports(scores, nk_list, n_max).to(device)

def topk_ports(scores: torch.Tensor, nk_list, n_max: int) -> torch.Tensor:
    # Dành riêng cho thuật toán DDPG, TD3 
    # Cho trước một điểm số cho mỗi cổng --> Giữ n_k cổng tốt nhất 
    idx = torch.topk(scores, n_max, dim=-1).indices # Lấy n_max cổng điểm cao nhất cho mọi UE, theo thứ tự giảm dần 
    sv = slot_valid_mask(nk_list, n_max, scores.device) 
    return torch.where(sv.expand_as(idx), idx, torch.full_like(idx, -1))

def merge_port_proposals(proposals: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
    # Gộp đề xuất của các AP theo ctrl (Toàn bộ AP đề xuất port cho UE --> UE nghe đề xuất của nearest-AP)
    # Dùng cho MAPPO 
    E, L, K, n = proposals.shape # (E, L, K, n_max) : đề xuất của AP 1 cho UE k
    idx = ctrl[:, None, :, None].expand(E, 1, K, n) # (E, K) -> (E, 1, K, n)
    return torch.gather(proposals, 1, idx)[:, 0] # Lấy theo chiều AP, bỏ chiều 1 (E, K, n)
