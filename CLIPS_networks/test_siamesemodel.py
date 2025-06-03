import torch
import torch.nn.functional as F
import os
from train_siamese_custom import SiamesePairClassifier

def normalize(vec):
    return [(x - 1) / 4 for x in vec]

def test_siamese(anchor, test_vectors):
    # optimize for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the trained model
    # ensure the model class is defined in train_siamese_custom.py
    model = SiamesePairClassifier().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "siamese_pair_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # normalize and prepare anchor tensor (batch size 1)
    anchor_norm = normalize(anchor)
    print(f"Anchor vector (raw): {anchor}")
    print(f"Anchor vector (normalized): {anchor_norm}\n")
    x1 = torch.tensor([anchor_norm], dtype=torch.float32).to(device)

    # iterate over test vectors
    for i, test_vec in enumerate(test_vectors):
        test_norm = normalize(test_vec)
        x2 = torch.tensor([test_norm], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(x1, x2)
            probs = F.softmax(logits, dim=1)
            p_same = probs[0, 0].item()
            p_diff = probs[0, 1].item()

        print(f"Test Vector #{i+1}: {test_vec}")
        print(f"  P(same) = {p_same:.4f}, P(different) = {p_diff:.4f}")

        if p_same > p_diff:
            print("  Model prediction: SAME concept (consistent)\n")
        else:
            print("  Model prediction: DIFFERENT concept (inconsistent)\n")

## main execution
if __name__ == "__main__":
    anchor = [5, 5, 4, 4]
    test_vectors = [
        [2, 1, 2, 1],  # consistent with anchor
        [4, 5, 4, 5],  # inconsistent with anchor
        [1, 1, 2, 2],  # other anchor
        [5, 5, 4, 4],  # exact same as anchor
    ]
    test_siamese(anchor, test_vectors)
