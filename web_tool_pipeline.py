import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import torch
from torchvision.transforms import Normalize, Resize, Compose
from torchvision.transforms import InterpolationMode
from supervised_phenotype_model import FusionModel
from torch.utils.data import DataLoader, TensorDataset
from train import load_clip, preprocess_text

# ---- Image + Text Preprocessing Functions ----

def getIndexOfLast(l, element):
    """ Get index of last occurrence of element """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i 

def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    
    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    return new_img

def preprocess_single_image_and_report(image_path: str, report_path: str, resolution: int = 320) -> Tuple[np.ndarray, str]:
    """
    Preprocess a single image and extract the 'IMPRESSION' section from a single report.
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_preprocessed = preprocess(img_pil, desired_size=resolution)
        img_array = np.array(img_preprocessed)
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess image: {image_path}, Error: {e}")

    try:
        with open(report_path, 'r') as f:
            s = f.read()
        s_split = s.split()
        if "IMPRESSION:" in s_split:
            begin = getIndexOfLast(s_split, "IMPRESSION:") + 1
            end = None
            end_cand1 = None
            end_cand2 = None

            if "RECOMMENDATION(S):" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION(S):")
            elif "RECOMMENDATION:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION:")
            elif "RECOMMENDATIONS:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATIONS:")

            if "NOTIFICATION:" in s_split:
                end_cand2 = s_split.index("NOTIFICATION:")
            elif "NOTIFICATIONS:" in s_split:
                end_cand2 = s_split.index("NOTIFICATIONS:")

            if end_cand1 and end_cand2:
                end = min(end_cand1, end_cand2)
            elif end_cand1:
                end = end_cand1
            elif end_cand2:
                end = end_cand2

            impression = " ".join(s_split[begin:end]) if end else " ".join(s_split[begin:])
        else:
            impression = 'NO IMPRESSION'
    except Exception as e:
        raise RuntimeError(f"Failed to process report: {report_path}, Error: {e}")

    return img_array, impression

# ---- Embedding Extraction for Single Image + Text ----

def get_single_embeddings(
    image_array: np.ndarray,
    impression_text: str,
    model,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate image and text embeddings from a preprocessed image array and impression string.
    """
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=0)  # [1, H, W]
    if image_array.shape[0] == 1:
        image_array = np.repeat(image_array, 3, axis=0)  # [3, H, W]

    img_tensor = torch.from_numpy(image_array).float()

    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
    ])
    img_tensor = transform(img_tensor).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    txt_tensor = preprocess_text([impression_text], model).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(img_tensor)
        text_embedding = model.encode_text(txt_tensor)

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    return image_embedding.cpu(), text_embedding.cpu()

# ---- Main Execution ----

def run_main(image_path: str, report_path: str):
    model_path = "checkpoints_train/pt-imp/checkpoint_4000.pt"

    img_array, impression = preprocess_single_image_and_report(image_path, report_path)
    print("Preprocessed image shape:", img_array.shape)
    print("Impression:", impression)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clip(model_path=model_path, pretrained=True, context_length=77)

    image_emb, text_emb = get_single_embeddings(img_array, impression, model, device)

    semi_model = FusionModel()
    semi_model.load_state_dict(
        torch.load("checkpoints_retrain/final_model_semi_supervised.pth", map_location=device),
        strict=False  # or weights_only=True if needed
    )
    semi_model.to(device)
    semi_model.eval()

    dummy_labels = torch.zeros((1, 14))
    dataset = TensorDataset(image_emb, text_emb, dummy_labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    PHENOTYPES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                  'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                  'Pneumothorax', 'Support Devices']

    pred_labels = []
    with torch.no_grad():
        for img_emb_batch, txt_emb_batch, _ in loader:
            img_emb_batch = img_emb_batch.to(device).float()
            txt_emb_batch = txt_emb_batch.to(device).float()

            outputs = semi_model(txt_emb_batch, img_emb_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().cpu().numpy()

            pred_indices = np.where(preds[0] == 1)[0].tolist()
            pred_labels = [PHENOTYPES[i] for i in pred_indices]

    return pred_labels
