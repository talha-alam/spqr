import torch
import clip

class ClipSimModel_Infer(torch.nn.Module):
    def __init__(self, args, device, prompts):
        super(ClipSimModel_Infer, self).__init__()
        self.device = device

        if self.device == 'cpu':
            raise ValueError('This script requires a CUDA-enabled GPU.')
        
        # Load the base CLIP model specified by the arguments
        self.MMM, self.preprocess = clip.load(args.language_model.split('_')[1], self.device, jit=False)
        self.MMM.to(self.device).eval()

        # Load the pre-tuned prompts directly from the .p file
        self.text_features = prompts
        print(f"Successfully loaded tuned prompts for Q16 model. Shape: {self.text_features.shape}")

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        
        # Normalize features for cosine similarity
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        return similarity.squeeze()

def eval_model(args, x, model, device='cuda'):
    labels = ['safe', 'nsfw']
    x = x.to(device)
    logits = model(x)
    probs = logits.softmax(dim=-1)
    
    prediction_score, pred_label_idx = torch.topk(probs.float(), 1)
    pred_label_idx = pred_label_idx.squeeze_()
    
    # Returns the index of the predicted label (0 for 'safe', 1 for 'nsfw')
    return pred_label_idx.cpu().detach().numpy().item()