"""
Export trained model to ONNX format.
"""
import torch
from pathlib import Path

from src.encoder import TemporalEncoder
from src.model import Predictor, PredictorForONNX


def export_to_onnx(
    encoder_path='models/encoder_pretrained.pt',
    predictor_path='models/predictor_final.pt',
    output_path='models/predictor_final.onnx',
    device='cpu'
):
    """
    Export trained predictor to ONNX format.
    
    Args:
        encoder_path: path to pretrained encoder
        predictor_path: path to trained predictor
        output_path: where to save ONNX model
        device: 'cpu' or 'cuda'
    """
    device = torch.device(device)
    
    # Load encoder
    print("Loading encoder...")
    encoder = TemporalEncoder(d_in=64, d_hidden=64, d_gru=96)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    
    # Load predictor
    print("Loading predictor...")
    predictor = Predictor(encoder, freeze_encoder=False)
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor.eval()
    
    # Wrap for ONNX (ensures last timestep output)
    model = PredictorForONNX(predictor).to(device)
    model.eval()
    
    # Dummy input (1 sequence, full 1000 steps, 64 features)
    dummy_input = torch.randn(1, 1000, 64, device=device)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        pred, conf = model(dummy_input)
        print(f"  Prediction shape: {pred.shape}")  # Should be (1, 2)
        print(f"  Confidence shape: {conf.shape}")  # Should be (1, 1)
        print(f"  Sample prediction: {pred[0].cpu().numpy()}")
        print(f"  Sample confidence: {conf[0].cpu().numpy()}")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}...")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['x'],
        output_names=['pred', 'conf'],
        dynamic_axes={
            'x': {0: 'batch', 1: 'time'},
            'pred': {0: 'batch'},
            'conf': {0: 'batch'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ ONNX model saved to {output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider']
    )
    
    # Test inference
    dummy_np = dummy_input.cpu().numpy()
    onnx_pred, onnx_conf = session.run(None, {'x': dummy_np})
    
    print(f"  ONNX prediction shape: {onnx_pred.shape}")
    print(f"  ONNX confidence shape: {onnx_conf.shape}")
    print(f"  ONNX sample prediction: {onnx_pred[0]}")
    print(f"  ONNX sample confidence: {onnx_conf[0]}")
    
    # Check consistency
    torch_pred = pred.cpu().numpy()
    torch_conf = conf.cpu().numpy()
    
    pred_diff = abs(onnx_pred - torch_pred).max()
    conf_diff = abs(onnx_conf - torch_conf).max()
    
    print(f"\n  Max prediction difference: {pred_diff:.2e}")
    print(f"  Max confidence difference: {conf_diff:.2e}")
    
    if pred_diff < 1e-5 and conf_diff < 1e-5:
        print("  ✓ ONNX model matches PyTorch model")
    else:
        print("  ⚠ Warning: ONNX model may have numerical differences")
    
    return session


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='models/encoder_pretrained.pt')
    parser.add_argument('--predictor', type=str, default='models/predictor_final.pt')
    parser.add_argument('--output', type=str, default='models/predictor_final.onnx')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    export_to_onnx(
        encoder_path=args.encoder,
        predictor_path=args.predictor,
        output_path=args.output,
        device=args.device
    )