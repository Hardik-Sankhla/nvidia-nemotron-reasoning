"""LoRA adapter validation and Kaggle submission packaging helpers."""

import os
import zipfile


def validate_lora(path):
    required_files = ["adapter_config.json"]

    for file_name in required_files:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            raise ValueError(f"Missing {file_name} in {path}")

    return True


def create_submission_zip(lora_path, output="submission.zip"):
    validate_lora(lora_path)

    with zipfile.ZipFile(output, "w") as zipf:
        for root, _, files in os.walk(lora_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # keep relative structure inside zip
                arcname = os.path.relpath(file_path, lora_path)
                zipf.write(file_path, arcname=arcname)

    return output


def ensure_vllm_compatibility(adapter_dir):
    """Basic compatibility guard used before evaluation/submission."""
    return validate_lora(adapter_dir)


def export_lora_adapter(state_dict, out_dir):
    """Compatibility placeholder for future custom save workflows."""
    os.makedirs(out_dir, exist_ok=True)
    # A real implementation can persist state_dict here if needed.
    return out_dir
