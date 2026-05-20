import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add services/asl to path
sys.path.insert(0, str(Path(__file__).parent))
from models import create_model, count_parameters

def main():
    print("=" * 60)
    print("ASL Self-Attention edge Optimization & Quantization Benchmark")
    print("=" * 60)
    
    checkpoint_path = r"services/asl/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return
        
    device = "cpu"
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("config", {}).get("model_type", "attention")
    
    # 1. Initialize PyTorch model and load weights
    model_fp32 = create_model(model_type)
    model_fp32.load_state_dict(checkpoint["model_state_dict"])
    model_fp32.eval()
    
    # 2. PyTorch Dynamic Quantization (INT8)
    print("\n[Optimization] Performing PyTorch Dynamic Quantization (INT8)...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear}, # Target linear layers for weight quantization
        dtype=torch.qint8
    )
    model_int8.eval()
    
    # 3. Export to ONNX format
    onnx_path = r"services/asl/checkpoints/best_model.onnx"
    onnx_exported = False
    size_onnx = 0.0
    try:
        import onnx
        print(f"[Optimization] Exporting model to ONNX: {onnx_path}...")
        dummy_input = torch.randn(1, 63)
        torch.onnx.export(
            model_fp32,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        size_onnx = os.path.getsize(onnx_path) / (1024 * 1024)
        onnx_exported = True
    except Exception as e:
        print(f"[INFO] ONNX library is not installed or export failed. Bypassing ONNX export. ({e})")
    
    # Check sizes
    size_fp32 = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    # Save quantized PyTorch checkpoint to check size
    quant_checkpoint_path = r"services/asl/checkpoints/best_model_quant.pt"
    torch.save(model_int8.state_dict(), quant_checkpoint_path)
    size_int8 = os.path.getsize(quant_checkpoint_path) / (1024 * 1024)
    
    # Clean up temporary quantized checkpoint file
    if os.path.exists(quant_checkpoint_path):
        os.remove(quant_checkpoint_path)
        
    print(f"  PyTorch FP32 Model Size: {size_fp32:.2f} MB")
    print(f"  PyTorch INT8 Model Size: {size_int8:.2f} MB")
    if onnx_exported:
        print(f"  ONNX FP32 Model Size:    {size_onnx:.2f} MB")
    
    # 4. Latency Benchmarks
    num_samples = 1000
    inputs = [torch.randn(1, 63) for _ in range(num_samples)]
    
    # Warmup
    print("\nRunning latency benchmarks (1000 iterations)...")
    for _ in range(50):
        _ = model_fp32(inputs[0])
        _ = model_int8(inputs[0])
        
    # Benchmark PyTorch FP32
    t0 = time.perf_counter()
    with torch.no_grad():
        for x in inputs:
            _ = model_fp32(x)
    t_fp32 = (time.perf_counter() - t0) * 1000.0 / num_samples
    fps_fp32 = 1000.0 / t_fp32
    
    # Benchmark PyTorch INT8
    t0 = time.perf_counter()
    with torch.no_grad():
        for x in inputs:
            _ = model_int8(x)
    t_int8 = (time.perf_counter() - t0) * 1000.0 / num_samples
    fps_int8 = 1000.0 / t_int8
    
    # Benchmark ONNX Runtime if available
    t_onnx = None
    fps_onnx = None
    try:
        import onnxruntime as ort
        print("[INFO] ONNX Runtime detected. Benchmarking ONNX model...")
        ort_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Warmup
        input_name = ort_sess.get_inputs()[0].name
        for _ in range(50):
            _ = ort_sess.run(None, {input_name: inputs[0].numpy()})
            
        t0 = time.perf_counter()
        for x in inputs:
            _ = ort_sess.run(None, {input_name: x.numpy()})
        t_onnx = (time.perf_counter() - t0) * 1000.0 / num_samples
        fps_onnx = 1000.0 / t_onnx
    except ImportError:
        print("[INFO] onnxruntime not found. Skipping ONNX runtime benchmark.")
        
    # 5. Output Summary Table
    print("\n" + "=" * 60)
    print(" EDGE DEPLOYMENT SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Format/Engine':<20} | {'Model Size':<10} | {'Latency (avg)':<15} | {'Throughput (FPS)':<16} | {'Speedup':<8}")
    print("-" * 78)
    print(f"{'PyTorch FP32':<20} | {size_fp32:.2f} MB   | {t_fp32:.3f} ms      | {fps_fp32:.2f} FPS       | {'1.00x':<8}")
    print(f"{'PyTorch INT8 (Quant)':<20} | {size_int8:.2f} MB   | {t_int8:.3f} ms      | {fps_int8:.2f} FPS       | {t_fp32/t_int8:.2f}x")
    
    if t_onnx is not None:
        print(f"{'ONNX Runtime FP32':<20} | {size_onnx:.2f} MB   | {t_onnx:.3f} ms      | {fps_onnx:.2f} FPS       | {t_fp32/t_onnx:.2f}x")
        
    print("=" * 60)
    
    # Save benchmark metrics to CSV
    csv_path = r"C:/Users/Pc/.gemini/antigravity/brain/331ab3b0-b2e1-44dd-b5fe-05c56a606d80/artifacts/asl_edge_benchmarks.csv"
    with open(csv_path, "w") as f:
        f.write("Engine,Size_MB,Latency_ms,FPS,Speedup\n")
        f.write(f"PyTorch_FP32,{size_fp32:.4f},{t_fp32:.4f},{fps_fp32:.2f},1.0\n")
        f.write(f"PyTorch_INT8,{size_int8:.4f},{t_int8:.4f},{fps_int8:.2f},{t_fp32/t_int8:.4f}\n")
        if t_onnx is not None:
            f.write(f"ONNX_Runtime_FP32,{size_onnx:.4f},{t_onnx:.4f},{fps_onnx:.2f},{t_fp32/t_onnx:.4f}\n")
    print(f"\n[SUCCESS] Edge benchmark metrics saved to {csv_path}!")

if __name__ == "__main__":
    main()
