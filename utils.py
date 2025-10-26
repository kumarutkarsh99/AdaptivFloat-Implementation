# utils.py
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial

@torch.no_grad()
def apply_quantization_to_model(model, quant_func):
    """
    Applies a given quantization function to all 'weight' parameters in a model.
    """
    print(f"Applying quantization with {quant_func.__name__}...")
    for name, param in model.named_parameters():
        if 'weight' in name: # We only quantize weights, not biases
            quantized_param = quant_func(param.data)
            param.data.copy_(quantized_param)
    print("Quantization complete.")

@torch.no_grad()
def simple_int8_quantizer(fp32_tensor, total_bits=8, **kwargs):
    """
    This is the "bad" uniform quantizer (like standard INT8)
    that the paper compares against.
    """
    q_min = -128
    q_max = 127

    # Calculate scale and zero point
    min_val, max_val = fp32_tensor.min(), fp32_tensor.max()

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return fp32_tensor 

    scale = (max_val - min_val) / (q_max - q_min)

    # Ensure scale is not too small
    if scale < 1e-10:
        return fp32_tensor

    zero_point_fp = q_min - (min_val / scale)
    zero_point = torch.round(zero_point_fp).clamp(q_min, q_max)

    # Quantize: (float / scale) + zero_point
    quantized_tensor = torch.round(fp32_tensor / scale + zero_point).clamp(q_min, q_max)

    # De-quantize (Simulated Quantization): (quant - zero_point) * scale
    dequantized_tensor = (quantized_tensor - zero_point) * scale

    return dequantized_tensor

@torch.no_grad()
def evaluate_resnet(model, dataloader, device):
    """
    Evaluates a ResNet model on a test dataloader (e.g., CIFAR-10).
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
    Performs a qualitative test on a BERT model for Masked LM.
    """
    model.eval()
    # Find the [MASK] token
    try:
        mask_token_id = tokenizer.mask_token_id
        tokenized_input = tokenizer(sentence, return_tensors="pt").to(device)
        mask_token_index = torch.where(tokenized_input.input_ids == mask_token_id)[1]

        # Get model predictions
        outputs = model(**tokenized_input)
        predictions = outputs.logits

        # Get the predicted token at the [MASK] position
        predicted_index = torch.argmax(predictions[0, mask_token_index]).item()
        predicted_token = tokenizer.decode([predicted_index])
        return predicted_token
    except Exception as e:
        print(f"Error during BERT evaluation: {e}")
        return "[EVALUATION FAILED]"