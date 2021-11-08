import streamlit as st
import tempfile

import numpy as np
from PIL import Image
import pandas as pd

from bat.attacks import SimBA
from bat.apis.deepapi import VGG16Cifar10

from tqdm import tqdm

# Initialize the UI
st.header("Black-box Attack Toolbox (BAT)")

logo = Image.open('bat_dark.png')
st.sidebar.image(logo, use_column_width=True)

attack = st.sidebar.radio(
    "What's your favorite attack",
    ('Distributed', 'Non-Distributed'))

# Upload the Image
f = st.sidebar.file_uploader("Please Select the image to be attacked", type=['jpg', 'jpeg'])

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(f.read())
    img = Image.open(tfile).convert('RGB')

    st.sidebar.write('Please wait for the magic to happen!')

    left_column, right_column = st.columns(2)

    left_column.image(img.resize((32, 32)), caption = "Input Image")

    x = np.asarray(img.resize((32, 32))) / 255.0

    # Initialize the Cloud API Model
    model = VGG16Cifar10("https://deep.wuhanstudio.uk" + "/vgg16_cifar10")

    # Get Preditction
    y_pred = model.predict(np.array([x]))[0]

    # Print result
    model.print(y_pred)
    print()
    print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

    left_column.text('Predictions ' + str(np.argmax(y_pred)) + ' ' + str(model.get_class_name(np.argmax(y_pred))))

    print("Initiating Attack")

    # SimBA Attack
    simba = SimBA(model)

    # x_adv = simba.attack(np.array(x))
    # x_adv = simba. attack_dist(x, epsilon=0.1, max_it=1000, batch=50, max_workers=10)
    
    x_adv, y_pred, y_list, perm = simba.init(x)

    max_it = 1000

    if attack == 'Distributed':
        batch = 50
        max_workers = 10
        pbar = tqdm(range(0, max_it, batch), desc="Dist SimBA")
    else:
        pbar = tqdm(range(0, max_it), desc="SimBA")

    probs = []
    l2_norm = []

    adv_placeholder = st.empty()
    with right_column:
        img_placeholder = st.empty()

    for i in pbar:

        if attack == 'Distributed':
            x_adv, y_adv, y_list = simba.batch(x_adv, y_pred, y_list, perm, i, 0.1, max_workers, batch)
        else:
            x_adv, y_adv, y_list = simba.step(x_adv, y_pred, y_list, perm, i, 0.1)
        with right_column:
            img_placeholder.image(Image.fromarray(np.uint8(x_adv * 255.0)).resize((32, 32)), caption = "Output Image")

        pbar.set_postfix({'origin prob': y_list[-1], 'l2 norm': np.sqrt(np.power(x_adv - x, 2).sum())})

        probs.append(y_list[-1])
        l2_norm.append(np.sqrt(np.power(x_adv - x, 2).sum()))

        with adv_placeholder.container():
            df = pd.DataFrame.from_dict({'Probability of ' + model.get_class_name(np.argmax(y_pred)): probs, 'L2 Norm': l2_norm}, orient='index')
            df
            st.line_chart(df.T)

        # Early break
        if y_adv is not None:
            if(np.argmax(y_adv) != np.argmax(y_pred)):
                break

    # Get Preditction
    y_adv = model.predict(np.array([x_adv]))[0]

    # Print result
    model.print(y_adv)
    print()
    print('Prediction', np.argmax(y_adv), model.get_class_name(np.argmax(y_adv)))

    right_column.text('Predictions ' + str(np.argmax(y_adv)) + ' ' + str(model.get_class_name(np.argmax(y_adv))))
