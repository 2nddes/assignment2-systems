import torch
import itertools
import gc
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW



def standard_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

# 初始化编译版本的 attention
# 注意：第一次传入新形状时会自动触发底层的 Triton 编译
compiled_attention = torch.compile(standard_attention)

def measure_performance(attention_fn, d_model, seq_len, device, mode_name):
    batch_size = 8
    try:
        # 清理状态，保证环境纯净
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 初始化输入参数
        q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        targets = torch.randint(0, d_model, (batch_size, seq_len), device=device)

        optimizer = AdamW([q, k, v], lr=1e-3)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # --- Warm-up (预热) ---
        # 对于 torch.compile，这一步会触发动态编译，耗时较长但不计入最终成绩
        for _ in range(5):
            out = attention_fn(q, k, v)
            loss = cross_entropy(out.view(-1, d_model), targets.view(-1))
            loss.backward()
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # --- 测量前向传播 (100次) ---
        start_event.record()
        for _ in range(100):
            out = attention_fn(q, k, v)
        end_event.record()
        torch.cuda.synchronize()
        fwd_time_ms = start_event.elapsed_time(end_event) / 100

        # --- 测量反向传播前的显存占用 ---
        out = attention_fn(q, k, v)
        loss = cross_entropy(out.view(-1, d_model), targets.view(-1))
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)

        # --- 测量反向传播 (100次) ---
        bwd_total_time_ms = 0.0
        for _ in range(100):
            out = attention_fn(q, k, v)
            loss = cross_entropy(out.view(-1, d_model), targets.view(-1))
            
            start_event.record()
            loss.backward()
            end_event.record()
            
            torch.cuda.synchronize()
            bwd_total_time_ms += start_event.elapsed_time(end_event)
            optimizer.zero_grad()

        bwd_time_ms = bwd_total_time_ms / 100

        print(f"{d_model:<10} | {seq_len:<10} | {mode_name:<10} | {fwd_time_ms:<15.2f} | {bwd_time_ms:<15.2f} | {mem_allocated:<15.2f}")

    except torch.cuda.OutOfMemoryError:
        print(f"{d_model:<10} | {seq_len:<10} | {mode_name:<10} | {'OOM':<15} | {'OOM':<15} | {'OOM':<15}")
        torch.cuda.empty_cache()
    except Exception as e:
        # torch.compile 在遇到极端情况时可能抛出非 OOM 异常
        print(f"{d_model:<10} | {seq_len:<10} | {mode_name:<10} | {'Error':<15} | {'Error':<15} | {'Error':<15}")
        torch.cuda.empty_cache()

def run_benchmark():
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("此脚本需要 CUDA 设备。")

    print(f"{'d_model':<10} | {'seq_len':<10} | {'Mode':<10} | {'Fwd Time (ms)':<15} | {'Bwd Time (ms)':<15} | {'Memory (MiB)':<15}")
    print("-" * 88)

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        # 1. 运行未编译的 Eager 模式
        measure_performance(standard_attention, d_model, seq_len, device, "Eager")
        
        # 2. 运行编译后的 Compiled 模式
        measure_performance(compiled_attention, d_model, seq_len, device, "Compiled")
        
        # 分隔线，方便阅读同一配置下的对比
        print("-" * 88)

if __name__ == "__main__":
    # 禁用 TF32 确保精度一致性和更准确的基准测试
    torch.backends.cuda.matmul.allow_tf32 = False
    run_benchmark()