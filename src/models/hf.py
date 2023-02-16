from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import torch

class HfTransformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.device = args.device
        self.img_height, self.img_width = args.img_height, args.img_width

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            args.model_path,
            id2label=args.id2label,
            label2id=args.label2id
        )

        self.image_processor = AutoImageProcessor.from_pretrained(args.model_path)

    def forward(self, img, mask):
        img = [x for x in img]
        mask = [x for x in mask]
        img_mask_processed = self.image_processor(images=img, segmentation_maps=mask, return_tensors="pt")
        img_processed = img_mask_processed['pixel_values'].to(self.device)
        pred_mask = self.model(img_processed)

        logits = nn.functional.interpolate(
            pred_mask.logits,
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=False,
        )

        return logits, img_mask_processed['labels'].to(device=self.device, dtype=torch.long)
