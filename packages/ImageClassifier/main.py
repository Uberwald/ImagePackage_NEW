import torch
import clip
from PIL import Image
import os

# загружаем модель
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x64", device=device)

# список классов, можно свободно добавлять любые
labels_prod = ['frozen foods', 'perfumery and cosmetic products',
          'dishes', 'dairy products', 'household goods', 'water, juices, drinks']
labels_not_prod = []
labels_excep = []
labels = labels_prod + labels_not_prod + labels_excep

text = clip.tokenize(labels).to(device)

# присваиваем этим классам номер
classes_num = list(range(1, len(labels) + 1))
class_dict = dict(zip(labels, classes_num))


def image_recogn(image_path):
    if os.path.exists(image_path) is False:
        print(f"Image file not exist: {image_path}. Execution abort.")
        return

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = dict(zip(labels, probs[0]))
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # Если процент уверенности слишком мал, класс = 0
    if sorted_items[0][1] < 0.2:
        product_class = 0

    # Определение класса продукта
    product_class = class_dict[sorted_items[0][0]]
    return product_class
