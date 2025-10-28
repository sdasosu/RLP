from torchinfo import summary as torch_summary

def summary(model, input, depth=1, device='cpu'):
    """
    Print model summary using torchinfo.
    
    Args:
        model: PyTorch model
        input: Input tensor for the model
        depth: Depth of nested modules to display (default: 1)
        device: Device to run summary on (default: 'cpu')
    """
    model = model.to(device)
    input = input.to(device)
    
    print("\nModel Summary:")
    print("=" * 80)
    torch_summary(
        model, 
        input_data=input,
        depth=depth,
        device=device,
        col_names=["input_size", "output_size", "num_params", "trainable"]
    )
    print("=" * 80)
    print()
