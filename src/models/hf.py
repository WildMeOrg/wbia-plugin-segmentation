from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn

class HfTransformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            args.model_path,
            id2label=args.id2label,
            label2id=args.label2id
        )

        self.image_processor = AutoImageProcessor.from_pretrained(args.model_path)

    def forward(self, img, mask):
        img_mask_processed = self.image_processor(img, mask)
        pred_mask = self.model(img_mask_processed['pixel_values'])

        return pred_mask.logits, img_mask_processed['labels']
