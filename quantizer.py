# quantizer.py
import torch

@torch.no_grad()
def quantize_to_adaptivfloat(fp32_tensor, total_bits, exponent_bits):
    """
    Implements the AdaptivFloat algorithm (simulated quantization).
    Based on Section 3 of the paper.
    """
    # --- 1. Calculate Format Parameters ---
    sign_bits = 1
    if total_bits <= (exponent_bits + sign_bits):
        raise ValueError("total_bits must be > exponent_bits + 1 (for sign bit)")
    mantissa_bits = total_bits - sign_bits - exponent_bits

    # --- 2. Find Exponent Bias (The "Shift") ---

    # Find the max absolute value in the tensor
    max_abs_val = torch.max(torch.abs(fp32_tensor))

    if max_abs_val == 0:
        return fp32_tensor # Tensor is all zeros, no quantization needed

    # Find the exponent of the max value (scale of the data)
    # log2(max_val) gives us the exponent
    max_exponent_needed = torch.floor(torch.log2(max_abs_val))

    # Find the largest exponent we can *store* with our bits
    # e.g., 3 exp bits -> max_storable = 2^3 - 1 = 7
    max_exponent_storable = (2**exponent_bits) - 1

    # The BIAS is the "shift" we need
    # This is the core of AdaptivFloat (Section 3.1)
    exponent_bias = max_exponent_needed - max_exponent_storable

    # --- 3. Define the "Adaptiv" Quantization Range ---

    # Smallest storable exponent is 0. With bias, this is our min normal exp
    min_normal_exponent = 0 + exponent_bias

    # Largest storable exponent. With bias, this is our max exp
    max_exponent = max_exponent_storable + exponent_bias

    # Smallest *normal* value we can represent (mantissa=1.0)
    # AF_min_normal = 1.0 * (2.0 ** min_normal_exponent)
    AF_min_normal = 2.0 ** min_normal_exponent

    # Largest value we can represent
    # Max mantissa is (2.0 - epsilon)
    max_mantissa = 2.0 - (2.0**(-mantissa_bits))
    AF_max = max_mantissa * (2.0**max_exponent)

    # --- 4. Quantize the Tensor (The "Snap-to-Grid") ---

    quantized_tensor = fp32_tensor.clone()
    sign = torch.sign(quantized_tensor)
    abs_tensor = torch.abs(quantized_tensor)

    # 4a. Handle Zeros (The "Sacrifice" from Sec 3.1 & 3.2)
    # We round values smaller than the halfway point to our min value to 0.
    zero_threshold = AF_min_normal / 2.0
    zero_mask = abs_tensor < zero_threshold

    # 4b. Handle Clamping
    # Clamp all values larger than our max representable value
    clamp_mask = abs_tensor > AF_max
    abs_tensor[clamp_mask] = AF_max

    # 4c. Quantize "Normal" Values
    # Everything that is not a zero or clamped
    normal_mask = ~zero_mask & ~clamp_mask

    if normal_mask.any(): # Only quantize if there are normal numbers
        vals_to_quantize = abs_tensor[normal_mask]

        # Find the exponent for *every* value
        exponent_of_values = torch.floor(torch.log2(vals_to_quantize))

        # Calculate the mantissa (will be [1.0, 2.0))
        mantissa = vals_to_quantize / (2.0**exponent_of_values)

        # Quantize the *fractional* part of the mantissa
        mantissa_fraction = mantissa - 1.0 # -> [0.0, 1.0)

        # Calculate number of steps for the mantissa
        num_quant_steps = 2.0**mantissa_bits

        # The core quantization step:
        quantized_fraction = torch.round(mantissa_fraction * num_quant_steps) / num_quant_steps

        # Reconstruct the quantized mantissa
        quantized_mantissa = 1.0 + quantized_fraction

        # Reconstruct the final absolute value
        quantized_abs_val = quantized_mantissa * (2.0**exponent_of_values)

        # Put the quantized values back into the tensor
        abs_tensor[normal_mask] = quantized_abs_val

    # 4d. Put Zeros back
    abs_tensor[zero_mask] = 0.0

    # 4e. Re-apply the sign
    final_tensor = sign * abs_tensor

    return final_tensor