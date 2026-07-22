from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

import torch as t
import argparse
import timeit


def run_benchmark(num_layers, num_heads, d_ff, batch_size, d_model, seq_len, vocab_size, warmup, steps, mode):

    device = "cuda" if t.cuda.is_available() else "cpu"
    model = BasicsTransformerLM(
        vocab_size=vocab_size, 
        context_length=seq_len, 
        d_model=d_model, 
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    optimizer = AdamW(model.parameters())
    model.to(device)

    x = t.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=t.int32, device=device)
    target = t.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=t.int32, device=device)

    def step():
        if mode == "forward":
            out = model(x)
        elif mode == "backward":
            out = model(x)
            loss = cross_entropy(out, target)
            loss.backward()
        elif mode == "optimize":
            out = model(x)
            loss = cross_entropy(out, target)
            loss.backward()
            optimizer.step()
        else: 
            raise ValueError(f"Unknown mode: {mode}")
# 1. 测试前环境清理
    t.cuda.empty_cache()           # 清理之前的显存缓存
    t.cuda.reset_peak_memory_stats() # 重置显存峰值统计
    
    # 2. 预热阶段 (Warmup)
    for _ in range(warmup):
        step()
    t.cuda.synchronize() # 确保预热完全结束
    
    # 3. 初始化 CUDA 原生计时器
    start_event = t.cuda.Event(enable_timing=True)
    end_event = t.cuda.Event(enable_timing=True)
    
    # 4. 正式计时 (核心执行区)
    start_event.record() # 记录起点
    
    for _ in range(steps):
        step()        # 疯狂下发任务，不进行任何同步阻塞
        
    end_event.record()   # 记录终点
    
    # 5. 统一同步，等待所有任务完成
    t.cuda.synchronize()
    
    # 6. 计算详细 Benchmark 数据
    # elapsed_time 直接返回毫秒 (ms)
    total_time_ms = start_event.elapsed_time(end_event) 
    avg_time_ms = total_time_ms / steps
    
    # 计算吞吐量 (每秒处理的样本数)
    total_samples = steps * batch_size
    throughput = total_samples / (total_time_ms / 1000.0) 
    
    # 获取显存峰值占用 (转换为 MB)
    max_memory_mb = t.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 7. 打印报告
    print("="*30)
    print("🏆 GPU Benchmark Report")
    print("="*30)
    print(f"Total Steps : {steps}")
    print(f"Batch Size  : {batch_size}")
    print("-" * 30)
    print(f"⏱️ Total Time : {total_time_ms:.2f} ms")
    print(f"⚡ Avg Latency: {avg_time_ms:.2f} ms / step")
    print(f"🚀 Throughput : {throughput:.2f} samples / sec")
    print(f"💾 Peak Memory: {max_memory_mb:.2f} MB")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mode", type=str, required=True, choices=[
        "forward", "backward", "optimize"
    ])

    args = parser.parse_args()

    run_benchmark(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        seq_len=args.seq_len,
        warmup=args.warmup,
        steps=args.steps,
        mode=args.mode
    )
