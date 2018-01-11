# GANs_for_data_augmentation

Data is a highly important factor in Deep Learning. The presence of a large, labeled and balanced dataset is
often essential for a model to perform optimally at a particular task. However, finding such data in the real
world is rare, and creating these datasets is a tedious task.

The aim of this project is to overcome one of the aspects of difficulties with datasets: unbalanced data
across different classes. In more detail, Generative Adversarial Networks (GANs) have been used on an
existing unbalanced dataset to create synthetic data of the lesser represented classes. To my best knowledge,
GANs have never been used for this purpose before. 

Using Deep Convolutional GANs after pretraining the Discriminator on the dataset gave promising results
which surpassed the baseline method and almost matched the original dataset.

Code for all experiments have been detailed in this repository. The report may be found at: https://drive.google.com/file/d/19u5PwBdRXN6555aIIx0LgTeysS_bwTT5/view?usp=sharing
