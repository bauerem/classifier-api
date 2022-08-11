from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import argparse



parser = argparse.ArgumentParser(description="Image classification model that takes image path and returns top prediction(s) with respect to imagenet classes.")
parser.add_argument('-m', '--model_path', type=str, default="./mobilenet_v3_small.pth", help="Path to pytorch model class trained to classify on imagenet.",  metavar='')
parser.add_argument('-i', '--image_path', type=str, default="./temp.jpg", help="Path to image to be classified.", metavar='')
parser.add_argument('-k', '--nr_predictions', type=int, default=5)
args = parser.parse_args()

"""
# Do following only to download model if not saved locally.
#torch.hub.list('pytorch/vision:v0.13.0')

model = torch.hub.load('pytorch/vision:v0.13.0', 'mobilenet_v3_small', weights=True)

torch.save(model, 'mobilenet_v3_small.pth')
"""

if __name__=="__main__":
    test_img = Image.open(args.image_path)
    test_img_data = np.asarray(test_img)

    #plt.imshow(test_img_data)
    #plt.show()

    # model expects 224x224 3-color image
    transform = transforms.Compose([
     transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ])

    transformed_img = transform(test_img)

    #plt.imshow(np.moveaxis(np.asarray(transformed_img),0,-1))
    #plt.show()

    # standard ImageNet normalization
    transform_normalize = transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )

    input_img = transform_normalize(transformed_img)

    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    labels_path = 'imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    model = torch.load(args.model_path) #'mobilenet_v3_small.pth')
    model.eval()

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, args.nr_predictions)

    labels = [idx_to_labels[str(k)][1] for k in pred_label_idx[0].tolist()]
    print(labels)
    probabilities = prediction_score[0].tolist()
    print(probabilities)
    retJson = {k: v for (k,v) in zip(labels, probabilities)}
    print(retJson)
    #pred_label_idx.squeeze_()
    #predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    #print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')

    #retJson = {}
    with open("text.txt", 'w') as f:
        json.dump(retJson, f)
