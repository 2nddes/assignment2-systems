from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

import torch as t
import argparse
import timeit


def run_benchmark(num_layers, num_heads, d_ff, batch_size, d_model, seq_len, vocab_size, warmup, steps, mode):
    model = BasicsTransformerLM(
        vocab_size=vocab_size, 
        context_length=seq_len, 
        d_model=d_model, 
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    optimizer = AdamW(model.parameters())

    x = t.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=t.int32, device="cuda" if t.cuda.is_available() else "cpu")
    target = t.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=t.int32, device="cuda" if t.cuda.is_available() else "cpu")

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

    for _ in range(warmup):
        step()
        t.cuda.synchronize()

    start_time = timeit.default_timer()

    for _ in range(steps):
        step()
        t.cuda.synchronize()

    end_time = timeit.default_timer()

    time_cost = end_time - start_time
    avg = time_cost * 1000 / steps

    return avg

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

    result = run_benchmark(
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

    print(f"{result}")