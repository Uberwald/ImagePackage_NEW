import torch
import clip
from PIL import Image
import os

# загружаем модель
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x64", device=device)

# список классов, можно свободно добавлять любые
labels_prod = ['frozen foods', 'dairy products', 'water, juices, drinks']
labels_not_prod = ['dishes', 'household goods']
labels_excep = ['perfumery and cosmetic products', 'household chemicals']
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

    # Определение класса продукта
    if sorted_items[0][0] in labels_prod:
      product_class = 0
    elif sorted_items[0][0] in labels_not_prod:
      product_class = 1
    elif sorted_items[0][0] in labels_excep:
      product_class = 2

    # Если процент уверенности слишком мал, класс = 3
    if sorted_items[0][1] < 0.2:
        product_class = 3
    answer = [product_class, sorted_items[0][0]]

    return tuple(answer)

def help():
    print('0 - продовольственные товары\n1 - непродовольственные товары\n2 - товары исключения'
    '\n3 - товар не распознан')
