import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import quantizer

@torch.no_grad()
def apply_quantization_to_model(model, quant_func_or_profile):
    """
    Applies quantization to a PyTorch model's weight parameters.

    Supports:
    1. Uniform precision: if a function is passed, it is applied to all weight layers.
    2. Mixed precision: if a dict (layer precision profile) is passed, applies per-layer quantization.

    Args:
        model (torch.nn.Module): Model to quantize.
        quant_func_or_profile (function or dict): Either a quantization function (uniform) 
                                                  or a dict mapping layer names to precision ('INT4', 'INT8', 'FP32').
    """
    # Define available quantization functions
    quant_map = {
        'INT4': partial(quantizer.quantize_to_adaptivfloat, total_bits=4, exponent_bits=2),
        'INT8': partial(quantizer.quantize_to_adaptivfloat, total_bits=8, exponent_bits=3),
        'FP32': lambda x: x  # No quantization
    }

    # --- CASE 1: Uniform precision ---
    if callable(quant_func_or_profile):
        quant_func = quant_func_or_profile
        print(f"Applying UNIFORM quantization using {quant_func.__name__}...")
        for name, param in model.named_parameters():
            if 'weight' in name:  # Only quantize weights
                param.data.copy_(quant_func(param.data))

    # --- CASE 2: Mixed-precision ---
    elif isinstance(quant_func_or_profile, dict):
        profile = quant_func_or_profile
        print("Applying MIXED-PRECISION quantization using provided profile...")
        for name, param in model.named_parameters():
            if 'weight' in name:
                precision = profile.get(name, 'FP32')  # Default to FP32 if layer not in profile
                if precision in quant_map:
                    param.data.copy_(quant_map[precision](param.data))
                else:
                    print(f"Warning: Unknown precision '{precision}' for layer {name}. Skipping.")

    else:
        raise TypeError("Expected a quantization function or a dict profile for apply_quantization_to_model")

    print("Quantization complete.")


@torch.no_grad()
def simple_int8_quantizer(fp32_tensor, total_bits=8, **kwargs):
    """
    Standard uniform INT8 quantizer (simulated).

    Args:
        fp32_tensor (torch.Tensor): Tensor to quantize.
        total_bits (int): Number of bits (default 8 for INT8).

    Returns:
        torch.Tensor: De-quantized tensor (simulated quantization).
    """
    q_min = -128
    q_max = 127

    min_val, max_val = fp32_tensor.min(), fp32_tensor.max()

    # Avoid division by zero
    if max_val == min_val:
        return fp32_tensor

    scale = (max_val - min_val) / (q_max - q_min)
    if scale < 1e-10:
        return fp32_tensor

    zero_point_fp = q_min - (min_val / scale)
    zero_point = torch.round(zero_point_fp).clamp(q_min, q_max)

    # Quantize and dequantize
    quantized_tensor = torch.round(fp32_tensor / scale + zero_point).clamp(q_min, q_max)
    dequantized_tensor = (quantized_tensor - zero_point) * scale

    return dequantized_tensor


@torch.no_grad()
def evaluate_resnet(model, dataloader, device):
    """
    Evaluates a ResNet model (or similar) on a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Data loader for evaluation.
        device (torch.device): Device to run the model on.

    Returns:
        float: Accuracy (%) on the dataset.
    """
    model.eval()
    correct = 0
    total = 0
    print("Evaluating ResNet...")
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


@torch.no_grad()
def evaluate_bert(model, tokenizer, sentence, device):
    """
    Performs a qualitative test on a BERT model (Masked LM).

    Args:
        model (transformers.PreTrainedModel): The BERT model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for input sentences.
        sentence (str): Input sentence containing a [MASK] token.
        device (torch.device): Device to run the model on.

    Returns:
        str: Predicted token for the [MASK] position.
    """
    model.eval()
    try:
        mask_token_id = tokenizer.mask_token_id
        tokenized_input = tokenizer(sentence, return_tensors="pt").to(device)
        mask_token_index = torch.where(tokenized_input.input_ids == mask_token_id)[1]

        outputs = model(**tokenized_input)
        predictions = outputs.logits

        predicted_index = torch.argmax(predictions[0, mask_token_index]).item()
        predicted_token = tokenizer.decode([predicted_index])
        return predicted_token

    except Exception as e:
        print(f"Error during BERT evaluation: {e}")
        return "[EVALUATION FAILED]"
