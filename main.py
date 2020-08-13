import torch
import math

from model import get_model_instance_segmentation, get_fasterrcnn_resnet50
from dataset import PennFudan
from utils import collate_fn, get_tranforms

def main():
    dataset = PennFudan('ImageSegmentation\data\PennFudanPed', get_tranforms(train=True))
    dataset_test = PennFudan('ImageSegmentation\data\PennFudanPed', get_tranforms(train=False))
    
    # Splitting the data
    indexes = torch.randperm(dataset.__len__()).tolist()
    dataset = torch.utils.data.Subset(dataset, indexes[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indexes[-50:])

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_fasterrcnn_resnet50()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        counter = 0
        avg_loss = 0
        for obj in train_loader:
            images, targets = obj
            counter += 1
            # moving data to gpu for training
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print('losses', losses)
            loss_value = losses.item()
            avg_loss += loss_value
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if counter != 0 and not counter%10:
                print('loss: ', avg_loss / 50)

if __name__ == '__main__':
    main()