import os
import glob
from pyexpat import model
import numpy as np
import torch
import sys
import importlib
import inspect
import pandas as pd
from get_NN_dataset import CustomDataset, generate_dataset
from testtttttttt import BinaryCNN
SUPPORTED_EXTS = [".pt", ".pth"]

def find_model_file(name, search_dir="."):
    if os.path.exists(name):
        return os.path.abspath(name)
    pattern = os.path.join(search_dir, f"{name}*")
    candidates = []
    for ext in SUPPORTED_EXTS:
        candidates += glob.glob(pattern + ext)
    candidates += glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No model file found for '{name}' in '{search_dir}'")
    return os.path.abspath(sorted(candidates)[0])

def load_torch_model(path):
    loaded = torch.load(path, map_location="cpu")
    # If it's a state_dict (dict of tensors) we can't reconstruct the model class here.
    if isinstance(loaded, dict) and not hasattr(loaded, "forward"):
        return ("state_dict", loaded)
    # assume it's a full model object
    try:
        loaded.eval()
    except Exception:
        pass
    return ("torch", loaded)

def summarize_state_dict(sd, max_items=30):
    """Print a short summary of state_dict keys and shapes (returns list of (k, tensor))."""
    items = list(sd.items())
    print("state_dict contains", len(items), "entries. Sample:")
    for k, v in items[:max_items]:
        try:
            print("  ", k, tuple(v.shape))
        except Exception:
            print("  ", k, type(v))
    return items

def try_run_test_script(obj, is_state_dict=False):
    """
    Try to import testtttttttt and call an evaluation function.
    - obj: model instance or state_dict
    - is_state_dict: if True prefer evaluate_state_dict / evaluate_sd functions
    Returns True if a call was made successfully.
    """
    mod_name = "testtttttttt"
    try:
        # import from same directory as this file
        sys.path.insert(0, os.path.dirname(__file__))
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
    except Exception as e:
        print("No test script importable (testtttttttt.py):", e)
        return False

    candidates = []
    if is_state_dict:
        candidates = ["evaluate_state_dict", "evaluate_sd", "evaluate_state", "evaluate"]
    else:
        candidates = ["evaluate_model", "evaluate", "make_confusion", "compute_confusion", "run_evaluation", "eval_model"]
    # try candidate functions
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    fn()
                else:
                    fn(obj)
                print(f"Called {mod_name}.{name}(...)")
                return True
            except Exception as e:
                print(f"Error when calling {mod_name}.{name}:", e)
                return False
    # fallback: try mod.main
    if hasattr(mod, "main"):
        fn = getattr(mod, "main")
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                fn()
            else:
                fn(obj)
            print(f"Called {mod_name}.main(...)")
            return True
        except Exception as e:
            print("Error when calling testtttttttt.main:", e)
            return False

    print("testtttttttt present but no suitable evaluation function found.")
    return False

def try_infer_sequential_from_state_dict(sd):
    """
    Attempt to build a simple torch.nn.Sequential model from the state_dict.
    Supports:
      - pure fully-connected chain (all weight tensors 2D)
      - conv1d -> fc pattern where conv weight is 3D and fc weight is 2D,
        and fc_in == conv_out_channels * conv_output_length
    Returns (model, param_tensor_list, inferred_input_shape) or (None, None, None).
    """
    items = list(sd.items())
    # weight entries in appearance order
    weight_entries = [(k, v) for k, v in items if k.endswith(".weight")]
    if not weight_entries:
        return None, None, None

    # 1) Pure Linear stack (existing behavior)
    if all(v.dim() == 2 for _, v in weight_entries):
        layers = []
        first_in = None
        for i, (k, w) in enumerate(weight_entries):
            out_f, in_f = int(w.size(0)), int(w.size(1))
            if first_in is None:
                first_in = in_f
            layers.append(torch.nn.Linear(in_f, out_f))
            if i != len(weight_entries) - 1:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers)
        tensor_list = []
        for k, w in weight_entries:
            tensor_list.append(w)
            bias_key = k[:-7] + ".bias"
            if bias_key in sd:
                tensor_list.append(sd[bias_key])
            else:
                tensor_list.append(torch.zeros(w.size(0)))
        inferred_shape = (1, first_in) if first_in is not None else None
        return model, tensor_list, inferred_shape

    # 2) conv1d -> fc pattern: look for a 3D conv weight and a 2D fc weight
    conv_candidates = [(k, v) for k, v in items if v.dim() == 3 and k.endswith(".weight")]
    fc_candidates = [(k, v) for k, v in items if v.dim() == 2 and k.endswith(".weight")]

    # try to find a conv + fc pair that makes sense
    for conv_key, conv_w in conv_candidates:
        for fc_key, fc_w in fc_candidates:
            out_ch, in_ch, ksize = int(conv_w.size(0)), int(conv_w.size(1)), int(conv_w.size(2))
            fc_out, fc_in = int(fc_w.size(0)), int(fc_w.size(1))
            # check if fc_in is divisible by conv out channels -> indicates flatten length
            if fc_in % out_ch != 0:
                continue
            conv_out_len = fc_in // out_ch
            # Use "same" padding for odd kernel sizes (padding = ksize//2) so conv preserves length.
            padding = ksize // 2
            # with same padding and stride=1, conv_out_len == input_len
            input_len = conv_out_len
            # build minimal model: Conv1d(padding=same) -> ReLU -> Flatten -> Linear
            model = torch.nn.Sequential(
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding=padding),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding=padding),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(out_ch * conv_out_len, fc_out)
            )
            # prepare tensor list in the same order as model.parameters()
            tensor_list = []
            # conv weight and bias
            tensor_list.append(conv_w)
            conv_bias_key = conv_key[:-7] + ".bias"
            if conv_bias_key in sd:
                tensor_list.append(sd[conv_bias_key])
            else:
                tensor_list.append(torch.zeros(out_ch))
            # fc weight and bias
            tensor_list.append(fc_w)
            fc_bias_key = fc_key[:-7] + ".bias"
            if fc_bias_key in sd:
                tensor_list.append(sd[fc_bias_key])
            else:
                tensor_list.append(torch.zeros(fc_out))
            inferred_shape = (1, in_ch, input_len)
            conv_info = {"in_ch": in_ch, "out_ch": out_ch, "ksize": ksize, "padding": padding, "conv_out_len": conv_out_len, "input_len": input_len}
            return model, tensor_list, inferred_shape, conv_info

    # no recognizable pattern
    return None, None, None, None

def load_tensors_into_model_by_order(model, tensor_list):
    """
    Copy tensors from tensor_list into model.parameters() in order.
    Returns True on complete copy, False otherwise.
    """
    params = list(model.parameters())
    n = min(len(params), len(tensor_list))
    for p, t in zip(params[:n], tensor_list[:n]):
        try:
            p.data.copy_(t.to(p.data.dtype))
        except Exception as e:
            print("Failed to copy parameter:", e)
            return False
    if len(params) != len(tensor_list):
        print("Warning: parameter count mismatch (model:", len(params), "tensors:", len(tensor_list), ")")
    return True

def run_demo(model, input_shape=(1, 8)):
    x = np.random.randn(*input_shape).astype(np.float32)
    t = torch.from_numpy(x)
    model.eval()
    with torch.no_grad():
        out = model(t)
    try:
        out_np = out.cpu().numpy()
    except Exception:
        out_np = np.asarray(out)
    return out_np

def validate_model(model, input_shape, test_input=None):
    """
    Do a single deterministic forward pass to verify the model was loaded correctly.
    - test_input: optional numpy array with same shape as input_shape to use instead of zeros.
    Returns a dict: { ok: bool, error: str?, output_shape: tuple?, sample: list? }
    """
    # prepare deterministic input (zeros) unless a test_input is provided
    if test_input is None:
        x = np.zeros(input_shape, dtype=np.float32)
    else:
        x = np.asarray(test_input, dtype=np.float32)
        if tuple(x.shape) != tuple(input_shape):
            return {"ok": False, "error": "test_input shape mismatch", "expected_shape": tuple(input_shape), "given_shape": tuple(x.shape)}
    t = torch.from_numpy(x)
    model.eval()
    try:
        with torch.no_grad():
            out = model(t)
    except Exception as e:
        return {"ok": False, "error": f"forward error: {e}"}
    try:
        out_np = out.cpu().numpy()
    except Exception:
        out_np = np.asarray(out)
    if not np.isfinite(out_np).all():
        return {"ok": False, "error": "output contains NaN or Inf", "output_shape": out_np.shape}
    return {"ok": True, "output_shape": out_np.shape, "sample": out_np.flatten()[:10].tolist()}

def get_input_shape_from_model(model, default=(1, 8)):
    """
    Attempt to extract an input shape from the loaded model by checking common attributes.
    If found, ensure there's a leading batch dimension (1). Otherwise try to infer from
    the first layer's weights (Conv/Linear). Fall back to default.
    """
    candidates = [
        "input_shape", "input_size", "example_input", "example_input_array",
        "dummy_input", "dummy_input_array", "input_dims", "in_shape", "input_dim"
    ]
    for attr in candidates:
        val = getattr(model, attr, None)
        if val is None:
            continue
        # Tensor or torch.Size
        if isinstance(val, torch.Size) or isinstance(val, tuple) or isinstance(val, list):
            shape = tuple(val)
        elif isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            shape = tuple(val.shape)
        elif isinstance(val, int):
            shape = (1, val)
        else:
            # unknown type -- skip
            continue
        # ensure batch dim
        if len(shape) == 0:
            continue
        if shape[0] != 1:
            shape = (1,) + tuple(shape)
        return shape

    # Inspect first module with a weight parameter to infer input channels/features
    for name, m in model.named_modules():
        # skip the top-level container if it has no parameters
        params = list(m.parameters())
        if not params:
            continue
        w = params[0]
        ws = tuple(w.shape)
        # Linear: weight shape (out_features, in_features)
        if isinstance(m, torch.nn.Linear) and len(ws) >= 2:
            in_features = ws[1]
            return (1, in_features)
        # Conv2d: weight shape (out_channels, in_channels, kH, kW)
        if isinstance(m, torch.nn.Conv2d) and len(ws) >= 2:
            in_ch = ws[1]
            # height/width unknown — choose a reasonable default (32x32)
            return (1, in_ch, 32, 32)
        # Conv1d: weight shape (out_channels, in_channels, kW)
        if isinstance(m, torch.nn.Conv1d) and len(ws) >= 2:
            in_ch = ws[1]
            return (1, in_ch, 128)
        # Generic fallback based on weight dimensions
        if len(ws) >= 2:
            in_ch = ws[1]
            if len(ws) == 4:
                return (1, in_ch, 32, 32)
            if len(ws) == 3:
                return (1, in_ch, 128)
            return (1, in_ch)

    # no hint found
    return default

def compute_confusion_from_dataset(model, dataset_path="10_Million_samples_LP_DEG_SAC.pt", train_test_ratio=0.8, test_batch=256, conv_info=None):
    """
    Load dataset file (same format as in testtttttttt.py), split train/test, run model on test set,
    and print per-bit confusion matrix (TP, TN, FP, FN) and a summary for bit 0.
    """
    if not os.path.exists(dataset_path):
        print("Dataset file not found:", dataset_path)
        return
    try:
        # dataset here is a pickled object (CustomDataset). Match testtttttttt.py and allow loading objects:
        ds = torch.load(dataset_path, weights_only=False, map_location="cpu")
    except Exception as e:
        print("Failed to load dataset:", e)
        return
    # try to ensure data/labels are tensors
    try:
        ds.data = torch.stack(ds.data)
        ds.labels = torch.stack(ds.labels)
    except Exception:
        # if already tensors or custom dataset, ignore
        pass
    total_size = len(ds)
    train_size = int(total_size * train_test_ratio)
    test_size = total_size - train_size
    # split and create test loader
    _, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print("using device:", device)

    # infer bits from a single batch
    bits = None
    confusion = None
    # if conv_info provided, prepare tail (skip conv) for cases where dataset contains conv outputs
    model_children = list(model.children())
    tail_module = None
    if conv_info is not None:
        # expect model layout: Conv1d, ReLU, Flatten, Linear
        # tail should be Flatten+Linear to accept pre-computed conv outputs
        if len(model_children) >= 4:
            tail_module = torch.nn.Sequential(*model_children[2:]).to(device)

    with torch.no_grad():
        for x, y in test_dl:
            # x may have shapes:
            #  - (batch, input_len)        : raw 1D signal -> needs unsqueeze to (batch,1,input_len)
            #  - (batch, in_ch*input_len) : flattened -> reshape to (batch,in_ch,input_len)
            #  - (batch, out_ch, conv_out_len) : already conv outputs -> apply tail only
            x_cpu = x
            # decide which input to feed
            if conv_info is not None and x_cpu.ndim == 3 and x_cpu.shape[1] == conv_info["out_ch"] and x_cpu.shape[2] == conv_info["conv_out_len"]:
                # dataset already contains conv outputs: run tail only
                logits = tail_module(x_cpu.to(device))
            else:
                # # adapt to expected (batch, in_ch, input_len) if possible
                # in_ch = conv_info["in_ch"] if conv_info is not None else None
                # input_len = conv_info["input_len"] if conv_info is not None else None
                # x_in = x_cpu
                # if x_in.ndim == 2:
                #     if input_len is not None and x_in.shape[1] == input_len:
                #         x_in = x_in.unsqueeze(1)
                #     elif in_ch is not None and input_len is not None and x_in.shape[1] == in_ch * input_len:
                #         x_in = x_in.view(x_in.shape[0], in_ch, input_len)
                #     else:
                #         # best-effort: unsqueeze to give channel dim
                #         x_in = x_in.unsqueeze(1)
                # elif x_in.ndim == 3:
                #     if in_ch is not None and input_len is not None:
                #         if not (x_in.shape[1] == in_ch and x_in.shape[2] == input_len):
                #             # try to reshape if total elements match
                #             total_per_sample = x_in.numel() // x_in.shape[0]
                #             if total_per_sample == in_ch * input_len:
                #                 x_in = x_in.view(x_in.shape[0], in_ch, input_len)
                logits = model(x_cpu.to(device))

            pred = torch.where(logits.cpu() > 0, 1.0, -1.0)
            y_cpu = y.cpu()
            if bits is None:
                bits = pred.shape[1]
                confusion = np.zeros((bits, 4), dtype=int)  # TP,TN,FP,FN
            for bit in range(bits):
                confusion[bit, 0] += ((pred[:, bit] == 1) & (y_cpu[:, bit] == 1)).sum().item()   # TP
                confusion[bit, 1] += ((pred[:, bit] == -1) & (y_cpu[:, bit] == -1)).sum().item()  # TN
                confusion[bit, 2] += ((pred[:, bit] == 1) & (y_cpu[:, bit] == -1)).sum().item()   # FP
                confusion[bit, 3] += ((pred[:, bit] == -1) & (y_cpu[:, bit] == 1)).sum().item()   # FN

    if confusion is None:
        print("No test data / could not compute confusion.")
        return
    df = pd.DataFrame(confusion, columns=["TP", "TN", "FP", "FN"])
    df.index.name = "Bit"
    print("Confusion matrix per bit:")
    print(df)

    # print metrics for bit 0 like in your script
    bit = 0
    TP, TN, FP, FN = confusion[bit]
    Total = confusion[bit].sum()
    FN_percent = 100 * FN / Total if Total > 0 else 0.0
    FP_percent = 100 * FP / Total if Total > 0 else 0.0
    TN_percent = 100 * TN / Total if Total > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    print()
    print(f"Bit 0: TN % = {TN_percent:.2f}%, FN % = {FN_percent:.2f}%, FP % = {FP_percent:.2f}%")
    print(f"Recall = {100*recall:.2f}%, Precision = {100*precision:.2f}%, F1 = {f1:.4f}")

def old_main():
    # Edit these variables (or set the MODEL_NAME env var) to point to the model you want to load.
    MODEL_NAME = os.environ.get("MODEL_NAME", "BNN_Model_50_10_Million_samples_LP_DEG_SAC")   # name or path (e.g. "mymodel", "mymodel.pt")
    SEARCH_DIR = "C:\\Users\\USER\\Documents\\binyamin\\studies\\סמסטר ח\\ארכיטקטורות מתקדמות\\code\\new code\\Lightweight-Neural-SBox-evaluation\\models"                                     # directory to search if name is not a path
    INPUT_SHAPE = (1, 8)                                 # change to match your model input

    try:
        path = find_model_file(MODEL_NAME, SEARCH_DIR)
    except FileNotFoundError as e:
        print("Model file not found:", e)
        return

    print("Using model file:", path)

    framework, loaded = load_torch_model(path)
    print("Detected:", framework)
    if framework == "state_dict":
        sd = loaded
        print("The file is a PyTorch state_dict.")
        # show a short summary
        summarize_state_dict(sd, max_items=40)
        # try to infer a simple sequential model
        model_guess, tensor_list, inferred_shape, conv_info = try_infer_sequential_from_state_dict(sd)
        if model_guess is None:
            print("Could not auto-infer a simple Sequential model from the state_dict.")
            # try to let user's test script handle the raw state_dict (if available)
            ran = try_run_test_script(sd, is_state_dict=True)
            if ran:
                print("Test script handled the state_dict.")
                return
            print("Recreate your model class and call model.load_state_dict(state_dict) before inference.")
            print("Example:\n  model = MyModel(...)\n  model.load_state_dict(state_dict)\n  run_demo(model, INPUT_SHAPE)")
            return
        print("Auto-inferred a Sequential Linear model with", len(list(model_guess.parameters())), "parameters.")
        ok = load_tensors_into_model_by_order(model_guess, tensor_list)
        if not ok:
            print("Failed to load parameters into the inferred model. Recreate your model class and load the state_dict manually.")
            return
        # proceed using the inferred model
        loaded = model_guess
        framework = "torch (inferred from state_dict)"
        print("Model reconstructed from state_dict. Proceeding with validation and demo.")
        # prefer the inferred input shape if available
        if inferred_shape is not None:
            detected_shape = inferred_shape
        else:
            detected_shape = INPUT_SHAPE
    else:
        # non-state_dict path: leave detected_shape to be computed below
        detected_shape = None

    # Try to get input shape from the model itself if available (only if not already set)
    if detected_shape is None:
        detected_shape = get_input_shape_from_model(loaded, INPUT_SHAPE)
    if detected_shape != INPUT_SHAPE:
        print("Overriding INPUT_SHAPE with model-reported shape:", detected_shape)

    # Validate model with a deterministic forward pass (zeros). Report concise result.
    v = validate_model(loaded, detected_shape)
    if not v.get("ok", False):
        print("Model validation FAILED:", v.get("error"))
        return
    print("Model validation OK — output shape:", v.get("output_shape"), "sample:", v.get("sample"))

    # Compute confusion table on your dataset using the loaded model
    dataset_file = os.environ.get("DATASET_FILE", "10_Million_samples_LP_DEG_SAC.pt")
    compute_confusion_from_dataset(loaded, dataset_path=dataset_file, conv_info=conv_info)
    return

def main():
    MODEL_NAME = os.environ.get("MODEL_NAME", "BNN_Model_50_10_Million_samples_LP_DEG_SAC")   # name or path (e.g. "mymodel", "mymodel.pt")
    SEARCH_DIR = "C:\\Users\\USER\\Documents\\binyamin\\studies\\סמסטר ח\\ארכיטקטורות מתקדמות\\code\\new code\\Lightweight-Neural-SBox-evaluation\\models"                                     # directory to search if name is not a path
    PATH  = find_model_file(MODEL_NAME, SEARCH_DIR)
    
    print("loading model:")
    model = BinaryCNN()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    
    dataset_file = os.environ.get("DATASET_FILE", "10_Million_samples_LP_DEG_SAC.pt")
    compute_confusion_from_dataset(model, dataset_path=dataset_file)

if __name__ == "__main__":
    main()