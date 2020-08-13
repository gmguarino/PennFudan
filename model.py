import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN


def get_fasterrcnn_resnet50():
    # Pre-trained model on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model_instance_segmentation(num_classes):
    # Pre-trained model on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask =  model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNN(model.features, in_features_mask, hidden_layer, num_classes)
    
    return model


