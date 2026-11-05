import torch

def aga(D: torch.Tensor) -> torch.Tensor:
    E, L, U, K = D.shape 
    if L * U < K: # Kiểm tra yêu cầu LU >= K 
        raise ValueError(f"AGA needs L*U >= K, got L*U={L * U} < K={K}")
    flat = D.reshape(E, L * U, K) # Làm phẳng AP và AP'antenna thành 1 danh sách 
    ar = torch.arange(E, device=D.device) # Đánh chỉ số theo kịch bản 
    F = torch.zeros(E, L * U, K, dtype=torch.bool, device=D.device) # Kết quả phép gán association
    feed_free = torch.ones(E, L * U, dtype=torch.bool, device=D.device) # Xem antenna nào còn trống 
    ue_unserved = torch.ones(E, K, dtype=torch.bool, device=D.device) # UE nào chưa có antenna 
    inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device) # đánh dấu loại cặp đã dùng 
    for _ in range(K): # Gán gặp UE-AP 
        cand = torch.where(feed_free[:, :, None] & ue_unserved[:, None, :], flat, inf) # Xét các cặp hợp lệ 
        idx = cand.reshape(E, -1).argmin(-1) # Tìm cặp nhỏ nhất trên toàn bảng 
        f, k = idx // K, idx % K # Tìm ra hàng cột, để ghép cặp 
        F[ar, f, k] = True # Ghép cặp 
        feed_free[ar, f] = False
        ue_unserved[ar, k] = False
    # stage 2: free feeds -> top-u' UEs of their AP    
    free = feed_free.reshape(E, L, U) # Xem AP'antenna nào rảnh 
    order = torch.argsort(D.mean(2), dim=-1, stable=True) # Xếp hạng UE theo khoảng cách tới từng AP 
    rank = (free.long().cumsum(-1) - 1).clamp_min(0) % K # i-th free feed of AP l -> i-th UE
    target = torch.gather(order, -1, rank) # (E, L, U) - Đổi hạng thành UE cụ thể 
    F |= (free[..., None] & torch.nn.functional.one_hot(target, K).bool()).reshape(E, L * U, K)
    return F.reshape(E, L, U, K)

def check_association(F: torch.Tensor) -> None: # Kiểm tra hợp lệ 
    """Raises if F violates the exclusive-assignment / coverage constraints."""
    if not (F.sum(-1) == 1).all(): # Cộng theo chiều UE, mỗi antenna phục vụ bnh UE
        raise ValueError("every feed must serve exactly one UE")
    if not (F.sum((1, 2)) >= 1).all(): # Cộng theo antenna và AP, mỗi UE có bnh antenna phục vụ 
        raise ValueError("every UE must be served by at least one feed")
