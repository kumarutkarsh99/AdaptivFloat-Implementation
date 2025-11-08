import torch

@torch.no_grad()
def quantize_to_adaptivfloat(fp32_tensor, total_bits, exponent_bits):
    """
    Simulates AdaptivFloat quantization for a given FP32 tensor.

    Args:
        fp32_tensor (torch.Tensor): The tensor to quantize.
        total_bits (int): Total number of bits for each value.
        exponent_bits (int): Number of bits allocated for the exponent.

    Returns:
        torch.Tensor: Quantized tensor (same shape as input).
    """
    # --- 1. Determine mantissa and sign bits ---
    sign_bits = 1
    if total_bits <= (exponent_bits + sign_bits):
        raise ValueError("total_bits must be greater than exponent_bits + 1 (for sign)")

    mantissa_bits = total_bits - sign_bits - exponent_bits

    # --- 2. Compute exponent bias based on max tensor value ---
    max_abs_val = torch.max(torch.abs(fp32_tensor))
    if max_abs_val == 0:
        return fp32_tensor  # all-zero tensor, no quantization needed

    # Maximum exponent needed to represent the largest tensor value
    max_exponent_needed = torch.floor(torch.log2(max_abs_val))

    # Largest storable exponent with the given number of bits
    max_exponent_storable = (2**exponent_bits) - 1

    # Exponent bias (shift) for AdaptivFloat
    exponent_bias = max_exponent_needed - max_exponent_storable

    # --- 3. Determine quantization range ---
    min_normal_exponent = 0 + exponent_bias      # minimum representable exponent
    max_exponent = max_exponent_storable + exponent_bias  # maximum representable exponent

    # Smallest normal value representable
    AF_min_normal = 2.0 ** min_normal_exponent

    # Largest value representable (max mantissa * 2^max_exponent)
    max_mantissa = 2.0 - (2.0**(-mantissa_bits))
    AF_max = max_mantissa * (2.0**max_exponent)

    # --- 4. Apply quantization ---
    quantized_tensor = fp32_tensor.clone()
    sign = torch.sign(quantized_tensor)
    abs_tensor = torch.abs(quantized_tensor)

    # 4a. Identify very small values that will become zero
    zero_threshold = AF_min_normal / 2.0
    zero_mask = abs_tensor < zero_threshold

    # 4b. Clamp values above max representable value
    clamp_mask = abs_tensor > AF_max
    abs_tensor[clamp_mask] = AF_max

    # 4c. Quantize normal values (neither zero nor clamped)
    normal_mask = ~zero_mask & ~clamp_mask
    if normal_mask.any():
        vals_to_quantize = abs_tensor[normal_mask]

        # Compute exponent for each value
        exponent_of_values = torch.floor(torch.log2(vals_to_quantize))

        # Extract mantissa (fractional part in [1.0, 2.0))
        mantissa = vals_to_quantize / (2.0 ** exponent_of_values)

        # Quantize the fractional part of mantissa
        mantissa_fraction = mantissa - 1.0  # range [0.0, 1.0)
        num_quant_steps = 2.0 ** mantissa_bits
        quantized_fraction = torch.round(mantissa_fraction * num_quant_steps) / num_quant_steps

        # Reconstruct quantized mantissa and absolute value
        quantized_mantissa = 1.0 + quantized_fraction
        quantized_abs_val = quantized_mantissa * (2.0 ** exponent_of_values)

        # Update tensor with quantized values
        abs_tensor[normal_mask] = quantized_abs_val

    # 4d. Set very small values to zero
    abs_tensor[zero_mask] = 0.0

    # 4e. Reapply original sign
    final_tensor = sign * abs_tensor

    return final_tensor
