"""Build the midterm report (Markdown) from the results in the repo.

Usage: python doc/make_report.py . doc/BaoCao_GiuaKy.md
"""

import json
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
R = ROOT / "results"
FIG = R / "figures"

MODELS = [
    ("simple_cnn", "Simple CNN"),
    ("simple_cnn_aug", "Simple CNN + Aug"),
    ("complex_cnn_aug", "Complex CNN + Aug"),
    ("resnet50_tl", "ResNet-50 TL + Aug"),
    ("mobilenetv3_tl", "MobileNetV3 TL + Aug"),
]
M = {k: json.loads((R / k / "metrics_test.json").read_text()) for k, _ in MODELS}
META = {k: json.loads((R / k / "meta.json").read_text()) for k, _ in MODELS}
CLEAN = json.loads((ROOT / "data" / "cleaning_summary.json").read_text())

md: list[str] = []
fig_no = [0]
tab_no = [0]
FIG_REL = "../results/figures"


def H(text, level=1):
    md.append(f"{'#' * level} {text}\n")


def P(text="", bold_prefix=None):
    md.append((f"**{bold_prefix.strip()}** " if bold_prefix else "") + text + "\n")


def B(items, style="bullet"):
    for k, it in enumerate(items, 1):
        mark = f"{k}." if style == "number" else "-"
        md.append(f"{mark} **{it[0].strip()}** {it[1]}" if isinstance(it, tuple) else f"{mark} {it}")
    md.append("")


def EQ(text):
    md.append(f"`{text}`\n")


def T(header, rows, caption, bold_rows=()):
    tab_no[0] += 1
    md.append(f"**Table {tab_no[0]}.** *{caption}*\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for i, row in enumerate(rows):
        cells = [f"**{v}**" if i in bold_rows and str(v) else str(v) for v in row]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")


def F(name, caption):
    fig_no[0] += 1
    md.append(f"![Figure {fig_no[0]}]({FIG_REL}/{name})\n")
    md.append(f"**Figure {fig_no[0]}.** *{caption}*\n")


def f4(x):
    return f"{x:.4f}"


# ------------------------------------------------------------------ cover
md.append("**DEEP NEURAL NETWORKS – MIDTERM REPORT**\n")
md.append("# Automated Chest X-ray Classification for Respiratory Diseases: COVID-19, Pneumonia and Normal\n")
P("[group number]", bold_prefix="Group:")
T(
    ["No.", "Student ID", "Full name"],
    [
        ["1", "11247345", "Lê Duy Quyền"],
        ["2", "11247347", "Phan Thị Anh Quỳnh"],
        ["3", "11247287", "Chử Vũ Thảo Hiền"],
        ["4", "11247344", "Nguyễn Minh Quang"],
    ],
    "Group members",
)
P("October 2026")
md.append("---\n")

# ------------------------------------------------------------------ Part 1
H("Part 1: Theory – Convolutional Neural Networks")

H("1.1 Why convolutional networks for images", 2)
P(
    "A digital image is a tensor of size C × H × W, where C is the number of channels (1 for a grayscale X-ray, "
    "3 for an RGB photo) and H × W is the spatial resolution. A fully connected network (multilayer perceptron, MLP) "
    "has to flatten this tensor into a vector. For a 224 × 224 grayscale image this gives 50,176 inputs, so a single "
    "hidden layer of 1,000 neurons already needs about 50 million weights. Flattening also throws away the spatial "
    "layout: the network no longer knows which pixels are neighbours, and a pattern that moves a few pixels looks "
    "like a completely different input."
)
P(
    "Convolutional Neural Networks (CNNs) solve both problems with three ideas. Local connectivity: each neuron only "
    "looks at a small window of the input (its receptive field). Weight sharing: the same small filter is applied at "
    "every position of the image, so the number of parameters does not depend on the image size. Hierarchy: early "
    "layers detect simple patterns such as edges and textures, and deeper layers combine them into larger structures "
    "(in our case, rib outlines, lung borders or areas of opacity). Because the same filter slides over the whole "
    "image, a feature is detected wherever it appears (translation equivariance)."
)

H("1.2 Convolutional layer", 2)
P(
    "A convolutional layer holds C_out filters, each of size K × K × C_in. Each filter slides over the input and, at "
    "every position, computes a weighted sum of the pixels under it plus a bias:"
)
EQ("y[i, j] = Σ_c Σ_m Σ_n  x[c, i·S + m, j·S + n] · w[c, m, n] + b")
P(
    "The result of one filter is a feature map; stacking the C_out feature maps gives the output tensor. Two "
    "hyperparameters control the geometry: the stride S (step between two positions) and the padding P (zeros added "
    "around the border). For an input of width W the output width is"
)
EQ("W_out = ⌊(W − K + 2P) / S⌋ + 1")
P(
    "and the number of trainable parameters is (K · K · C_in + 1) · C_out. For example, the first layer of our simple "
    "CNN uses 16 filters of size 5 × 5 on a 1-channel image with padding 2 and stride 1, so it keeps the 224 × 224 "
    "size and has (5 · 5 · 1 + 1) · 16 = 416 parameters. Stacking layers grows the receptive field: two 3 × 3 layers "
    "see a 5 × 5 region, three of them a 7 × 7 region, with fewer parameters than one large filter. A 1 × 1 "
    "convolution does not look at neighbours at all; it mixes channels and is used to reduce or expand the channel "
    "dimension cheaply."
)

H("1.3 Activation functions", 2)
P(
    "Without a non-linear function between layers, a stack of convolutions would collapse into one linear operation. "
    "Modern CNNs use the Rectified Linear Unit, ReLU(x) = max(0, x). It is cheap to compute and its gradient is 1 "
    "for positive inputs, which avoids the vanishing gradient problem of sigmoid and tanh in deep networks. "
    "MobileNetV3 uses a variant called hard-swish, h-swish(x) = x · ReLU6(x + 3) / 6, a smooth approximation that "
    "works well on mobile hardware."
)

H("1.4 Pooling layers", 2)
P(
    "A pooling layer reduces the spatial size of feature maps without any trainable parameters. Max pooling keeps "
    "the largest value in each window, average pooling keeps the mean. A 2 × 2 window with stride 2 halves the width "
    "and height, which cuts computation in the following layers, enlarges the receptive field and makes the network "
    "less sensitive to small shifts. Global average pooling (GAP) averages each whole feature map to a single number; "
    "it is used before the classifier in GoogLeNet, ResNet and MobileNet instead of a large flatten + dense layer, "
    "which removes millions of parameters."
)

H("1.5 Fully connected layers, softmax and loss function", 2)
P(
    "After the convolution and pooling stages, the feature maps are flattened (or globally pooled) and passed to one "
    "or more fully connected (FC) layers. The last FC layer has one output per class, giving scores (logits) z_1 ... z_K. "
    "The softmax function turns them into probabilities:"
)
EQ("p_k = exp(z_k) / Σ_j exp(z_j)")
P(
    "Training minimises the cross-entropy loss between the predicted distribution and the true label y. When the "
    "classes are unbalanced, a weight w_y can be given to each class so that errors on a rare class cost more:"
)
EQ("L = − w_y · log p_y")

H("1.6 Training a CNN", 2)
P(
    "The gradients of the loss with respect to all weights are computed by backpropagation (the chain rule applied "
    "layer by layer from the output back to the input). The weights are then updated with mini-batch gradient "
    "descent: at each step a batch of B images is used to estimate the gradient. Plain SGD updates w ← w − η · ∇L, "
    "where η is the learning rate. Adam keeps running averages of the gradient and of its square and uses them to "
    "give each weight its own adaptive step size; AdamW is Adam with weight decay applied directly to the weights, "
    "which regularises better than adding an L2 term to the loss. The learning rate is often reduced during "
    "training, for example by a factor when the validation loss stops improving (ReduceLROnPlateau). One pass over "
    "the whole training set is called an epoch."
)

H("1.7 Regularisation and techniques against overfitting", 2)
B(
    [
        ("Batch Normalization (BN): ", "normalises the output of a layer over the mini-batch, x̂ = (x − μ_B) / √(σ²_B + ε), "
         "then rescales it with learned γ and β. BN stabilises and speeds up training and allows higher learning rates."),
        ("Dropout: ", "during training, each neuron is switched off with probability p, so the network cannot rely on "
         "a few specific neurons. At test time all neurons are used."),
        ("Weight decay: ", "pulls weights towards zero and keeps the model simple."),
        ("Data augmentation: ", "creates new training images on the fly by applying label-preserving transformations "
         "(rotation, shift, zoom, brightness changes). The model sees a slightly different version of each image every "
         "epoch, which acts as extra data."),
        ("Early stopping: ", "the validation loss is monitored after each epoch; training stops when it has not improved "
         "for a number of epochs (the patience), and the best checkpoint is kept."),
    ]
)

H("1.8 Well-known CNN architectures", 2)
B(
    [
        ("LeNet-5 (LeCun et al., 1998): ", "the first successful CNN, used for handwritten digits. It alternates "
         "convolution and pooling layers and ends with fully connected layers. Our simple CNN follows this pattern."),
        ("AlexNet (2012) and VGG (2014): ", "deeper networks trained on ImageNet with ReLU and dropout. VGG showed that "
         "stacks of small 3 × 3 filters work better than a few large filters."),
        ("GoogLeNet / Inception (Szegedy et al., 2015): ", "introduced the Inception block, in which the input goes "
         "through four parallel branches (1 × 1 convolution, 1 × 1 then 3 × 3, 1 × 1 then 5 × 5, and 3 × 3 max pooling "
         "then 1 × 1) whose outputs are concatenated along the channel axis. The block captures patterns at several "
         "scales at once, and the 1 × 1 convolutions keep the cost low."),
        ("ResNet (He et al., 2016): ", "very deep plain networks become harder to train (the degradation problem). "
         "A residual block learns F(x) and outputs y = ReLU(F(x) + x); the identity shortcut lets the gradient flow "
         "directly to earlier layers, which made networks with 50 to 152 layers trainable. ResNet-50 uses "
         "bottleneck blocks (1 × 1, 3 × 3, 1 × 1 convolutions)."),
        ("MobileNet (Howard et al., 2017–2019): ", "built for phones. It replaces a standard convolution with a "
         "depthwise convolution (one K × K filter per channel) followed by a 1 × 1 pointwise convolution, which cuts "
         "the computation by roughly a factor of K². MobileNetV2 added inverted residual blocks with linear "
         "bottlenecks, and MobileNetV3 added squeeze-and-excitation attention and the h-swish activation."),
    ]
)

H("1.9 Transfer learning and fine-tuning", 2)
P(
    "Training a deep CNN from scratch needs a large labelled dataset, which is rarely available in medical imaging. "
    "Transfer learning reuses a network that was already trained on a large dataset such as ImageNet (1.2 million "
    "images, 1,000 classes). Its early layers have learned general filters (edges, corners, textures) that are useful "
    "for almost any image. Two strategies are common. In feature extraction, the pretrained layers are frozen and only "
    "a new classifier head is trained. In fine-tuning, some of the deeper pretrained layers are unfrozen as well and "
    "trained with a small learning rate, so that the high-level features adapt to the new domain without destroying "
    "what was learned before. In practice the two are combined: the head is trained first, then the last stages are "
    "fine-tuned."
)

H("1.10 Evaluation metrics", 2)
P("For each class c, with TP, FP and FN the true positives, false positives and false negatives of that class:")
B(
    [
        ("Accuracy ", "= number of correct predictions / total number of samples."),
        ("Precision_c ", "= TP / (TP + FP): of the images predicted as class c, the fraction that really belongs to c."),
        ("Recall_c ", "= TP / (TP + FN): of the images of class c, the fraction the model finds. In medicine this is "
         "the sensitivity; a low recall for a disease means missed patients."),
        ("F1_c ", "= 2 · Precision · Recall / (Precision + Recall), the harmonic mean of the two."),
        ("Macro average: ", "the per-class values are averaged with equal weight, so a small class counts as much as a "
         "large one."),
        ("Confusion matrix: ", "a K × K table where entry (i, j) counts the images of true class i predicted as class j. "
         "It shows which classes are confused with which."),
    ]
)

H("1.11 Explainability with Grad-CAM", 2)
P(
    "A CNN gives a prediction but not a reason. Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., "
    "2017) produces a heatmap of the image regions that most influenced the score y^c of a class c. Let A^k be the k-th "
    "feature map of the last convolutional layer. The importance of each map is the spatial average of the gradient:"
)
EQ("α_k^c = (1 / Z) Σ_i Σ_j ∂y^c / ∂A^k_ij")
P("The heatmap is the weighted sum of the maps, keeping only positive evidence, then upsampled to the image size:")
EQ("L^c = ReLU( Σ_k α_k^c · A^k )")
P(
    "Overlaying L^c on the X-ray shows whether the network looks at the lungs or at something unrelated, such as text "
    "markers or the image border."
)

# ------------------------------------------------------------------ Part 2
H("Part 2: Practical Application")
H("I. Problem Description", 2)
P("Automated Chest X-ray Classification for Respiratory Diseases (COVID-19, Pneumonia, Normal).", bold_prefix="Title: ")
P("", bold_prefix="Research objectives:")
B(
    [
        "Build CNN models that classify a frontal chest X-ray into one of three classes: COVID-19, Pneumonia or Normal.",
        "Compare three families of models required by the course: a simple CNN designed by the group, a more complex "
        "CNN built from Inception and residual blocks, and pretrained networks adapted with transfer learning and "
        "fine-tuning.",
        "Go beyond overall accuracy: find which classes are confused, which class is hardest, and which image regions "
        "the network actually uses (Grad-CAM), and discuss the limitations of the models and the data.",
    ]
)
P("one chest X-ray image (PNG), converted to grayscale and resized to 224 × 224 pixels.", bold_prefix="Input: ")
P(
    "one of the three labels {COVID-19, Normal, Pneumonia} together with the softmax probability of each class; "
    "optionally a Grad-CAM heatmap explaining the decision.",
    bold_prefix="Output: ",
)
P("", bold_prefix="Summary of the tasks performed:")
B(
    [
        "Downloaded the Kaggle dataset and audited it; removed corrupted files, exact duplicates and near-duplicates "
        "to avoid leakage between the training and test sets.",
        "Split the cleaned data into training, validation and test sets (70 / 15 / 15, stratified by class).",
        "Normalised the images and applied data augmentation to the training set.",
        "Designed and trained a simple CNN, including a search over learning rate and batch size.",
        "Designed and trained a complex CNN with Inception and residual blocks.",
        "Fine-tuned two ImageNet models, ResNet-50 and MobileNetV3-Large.",
        "Evaluated all models on the test set with accuracy, precision, recall, F1-score and confusion matrices, and "
        "visualised their decisions with Grad-CAM.",
    ],
    style="number",
)

H("II. Dataset Description", 2)
P("https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images", bold_prefix="Download link: ")
raw = CLEAN["raw"]
P(
    f"The dataset contains {sum(raw.values()):,} chest X-ray images in PNG format, organised in three folders: "
    f"COVID ({raw['COVID']:,} images), NORMAL ({raw['NORMAL']:,}) and PNEUMONIA ({raw['PNEUMONIA']:,}). The objects "
    "in the images are medical images, namely frontal (posterior-anterior or anterior-posterior) radiographs of the "
    "chest. Most images are 232 × 232 pixels (4,228 images) and the rest are 256 × 256 pixels (1,000 images). The "
    "files are stored as RGB (one file as RGBA), but the three channels are identical, so the content is grayscale."
)
P(
    "Looking at the samples (Figure 1) shows that the classes do not come from the same source. The Normal and "
    "Pneumonia images are mostly paediatric radiographs with the same acquisition style and an \"R\" marker, while "
    "the COVID-19 images are adult radiographs collected from several other sources, often with ECG leads, tubes and "
    "different text labels. This matters for the interpretation of the results (Section IV.4)."
)
F("samples.png", "Random training images from each class.")

H("III. CNN Model Design", 2)
H("1) Data preprocessing: cleaning, normalisation and augmentation", 3)
P(bold_prefix="Cleaning. ", text=
    "While inspecting the data we found many repeated images. If the same image ends up in both the training and "
    "the test set, the test score measures memorisation instead of generalisation (data leakage). We therefore "
    "cleaned the dataset in four steps:"
)
B(
    [
        ("Corrupted files: ", "every file was opened and verified; none were broken."),
        ("Exact duplicates: ", f"files with the same MD5 hash were reduced to one copy ({CLEAN['exact_dups']} files removed)."),
        ("Near-duplicates: ", "the same radiograph also appears with slightly different contrast, crop or size. "
         "We computed a 16 × 16 difference hash (dHash) for each image and grouped images whose hashes differ in at "
         "most 4 of 256 bits. We checked sample pairs visually and they were indeed the same radiograph. One image per "
         f"group was kept ({CLEAN['near_dups']} images removed). The COVID class had by far the most copies."),
        ("Label conflicts: ", f"groups that contained the same image under different labels were dropped entirely "
         f"({CLEAN['label_conflicts']} images)."),
    ],
)
c = CLEAN["clean"]
T(
    ["Class", "Original", "After cleaning", "Removed"],
    [[k, f"{raw[k]:,}", f"{c[k]:,}", f"{raw[k] - c[k]:,}"] for k in ["COVID", "NORMAL", "PNEUMONIA"]]
    + [["Total", f"{sum(raw.values()):,}", f"{sum(c.values()):,}", f"{sum(raw.values()) - sum(c.values()):,}"]],
    "Number of images before and after cleaning.",
    bold_rows=(3,),
)
F("class_distribution.png", "Class distribution before and after cleaning.")
P(bold_prefix="Normalisation. ", text=
    "All images were converted to one grayscale channel, resized to 224 × 224 with bilinear interpolation and scaled "
    "to [0, 1]. For the two networks designed by the group, the pixels were then standardised as (x − 0.5) / 0.25. "
    "The pretrained networks expect three-channel input, so the grayscale channel was copied three times and "
    "normalised with the ImageNet mean and standard deviation."
)
P(bold_prefix="Augmentation. ", text=
    "Only the training images were augmented, randomly and differently at every epoch: rotation within ±10°, "
    "translation up to 5 % of the image size, scaling between 0.9 and 1.1, and brightness and contrast changes of "
    "±20 % (Figure 3). We deliberately did not use horizontal flipping: on a chest X-ray the heart lies on the left "
    "side, so a mirrored image is anatomically wrong."
)
F("augmentation.png", "Examples of random augmentation applied to two training images.")
P(bold_prefix="Class imbalance. ", text=
    "After cleaning, COVID is the smallest class (about 26 % of the images). To keep the model from neglecting it, "
    "the cross-entropy loss uses inverse-frequency class weights computed on the training set."
)
P(bold_prefix="Data split. ", text=
    "The cleaned images were split per class (stratified) with a fixed random seed (42), so every subset keeps the "
    "class proportions. The exact file list of each subset is saved in data/splits.csv."
)
sp = CLEAN["split"]
T(
    ["Subset", "Purpose", "COVID", "NORMAL", "PNEUMONIA", "Total"],
    [
        [s.capitalize() if s != "val" else "Validation", purpose,
         sp[s]["COVID"], sp[s]["NORMAL"], f"{sp[s]['PNEUMONIA']:,}", f"{sum(sp[s].values()):,}"]
        for s, purpose in [("train", "train the weights (70 %)"), ("val", "tune and select models (15 %)"),
                           ("test", "final evaluation (15 %)")]
    ],
    "Number of samples in the training, validation and test sets.",
)

H("2) Simple CNN", 3)
P(
    "The first model follows the LeNet idea: three convolution–ReLU–max-pooling stages followed by two fully "
    "connected layers. Larger 5 × 5 filters are used in the first two layers to capture broad structures of the "
    "chest; an adaptive average pooling layer reduces the 28 × 28 maps to 7 × 7 before flattening, which keeps the "
    "first dense layer small. Figure 4 shows the architecture and Table 4 lists every layer."
)
F("arch_simple_cnn.png", "Architecture of the simple CNN (compressed notation, input at the bottom).")
T(
    ["Layer", "Configuration", "Output shape", "Parameters"],
    [
        ["Input", "grayscale image", "1 × 224 × 224", "0"],
        ["Conv1 + ReLU", "16 filters 5 × 5, stride 1, pad 2", "16 × 224 × 224", "416"],
        ["MaxPool1", "2 × 2, stride 2", "16 × 112 × 112", "0"],
        ["Conv2 + ReLU", "32 filters 5 × 5, stride 1, pad 2", "32 × 112 × 112", "12,832"],
        ["MaxPool2", "2 × 2, stride 2", "32 × 56 × 56", "0"],
        ["Conv3 + ReLU", "64 filters 3 × 3, stride 1, pad 1", "64 × 56 × 56", "18,496"],
        ["MaxPool3", "2 × 2, stride 2", "64 × 28 × 28", "0"],
        ["AdaptiveAvgPool", "to 7 × 7, then flatten", "3,136", "0"],
        ["FC1 + ReLU + Dropout", "128 units, dropout 0.5", "128", "401,536"],
        ["FC2 + Softmax", "3 units", "3", "387"],
        ["Total", "", "", "433,667"],
    ],
    "Layers of the simple CNN.",
    bold_rows=(10,),
)
P(bold_prefix="Training setup and hyperparameter tuning. ", text=
    "All models were trained with AdamW (weight decay 10⁻⁴), weighted cross-entropy, mixed precision on an NVIDIA "
    "RTX 3050 Laptop GPU (6 GB), ReduceLROnPlateau (factor 0.3, patience 3) and early stopping on the validation loss "
    "(patience 8, best checkpoint restored). For the simple CNN we searched the learning rate and the batch size "
    "(augmentation on, at most 25 epochs) and compared the settings on the validation set only, so the test set "
    "stayed untouched (Table 5)."
)
tune_rows = []
for lr in ("0.003", "0.001", "0.0003"):
    for bs in (32, 64):
        d = R / f"tune_simple_lr{lr}_bs{bs}"
        if not d.exists():
            d = R / f"tune_simple_lr{float(lr):g}_bs{bs}"
        m = json.loads((d / "metrics_val.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        tune_rows.append([lr, bs, f4(m["accuracy"]), f4(m["f1_macro"]), meta["best_epoch"]])
T(
    ["Learning rate", "Batch size", "Val. accuracy", "Val. macro-F1", "Best epoch"],
    tune_rows,
    "Learning-rate and batch-size search for the simple CNN (validation set).",
    bold_rows=(0,),
)
P(
    "The combination learning rate 0.003 and batch size 32 gave the best validation F1 (0.9586) and was used for the "
    "final simple CNN. Smaller learning rates converged more slowly and had not caught up within the epoch budget, "
    "and batch size 32 was better than 64 for every learning rate, probably because the noisier gradient of smaller "
    "batches acts as a regulariser on such a small dataset. The simple CNN was then trained twice, without and with "
    "augmentation, to measure the effect of augmentation."
)

H("3) Complex CNN with Inception and residual blocks", 3)
P(
    "The second model is deeper and built from blocks, combining the two ideas from GoogLeNet and ResNet (Figure 5). "
    "A stem (7 × 7 convolution with stride 2, max pooling, 1 × 1 and 3 × 3 convolutions, max pooling) reduces the "
    "image to 96 × 28 × 28. Two Inception blocks then look at the feature maps at several scales in parallel. "
    "Block A has 32 (1 × 1), 64 (1 × 1 → 3 × 3), 16 (1 × 1 → 5 × 5) and 16 (pool → 1 × 1) output channels, concatenated "
    "to 128; block B has 64, 96, 32 and 32 channels, concatenated to 224. After another max pooling, four residual "
    "blocks (two with 256 channels, two with 384 channels, the third with stride 2) deepen the network while the "
    "identity shortcuts keep it easy to optimise. A 1 × 1 convolution is used on the shortcut when the number of "
    "channels or the resolution changes. Every convolution is followed by batch normalisation and ReLU. The classifier "
    "uses global average pooling, dropout 0.4, a 128-unit dense layer, dropout 0.3 and the 3-unit output layer. The "
    "learning rate was 0.001 with batch size 32."
)
F("arch_complex_cnn.png", "Architecture of the complex CNN (left) with the Inception block and the residual block used inside it (right).")
T(
    ["Stage", "Content", "Output shape"],
    [
        ["Input", "grayscale image", "1 × 224 × 224"],
        ["Stem", "Conv 7 × 7/2 (32) + BN + ReLU, MaxPool 3 × 3/2", "32 × 56 × 56"],
        ["", "Conv 1 × 1 (64), Conv 3 × 3 (96), MaxPool 3 × 3/2", "96 × 28 × 28"],
        ["Inception A", "branches 32 / 64 / 16 / 16, concatenated", "128 × 28 × 28"],
        ["Inception B", "branches 64 / 96 / 32 / 32, concatenated", "224 × 28 × 28"],
        ["Pooling", "MaxPool 3 × 3/2", "224 × 14 × 14"],
        ["Residual stage 1", "2 residual blocks, 256 channels", "256 × 14 × 14"],
        ["Residual stage 2", "2 residual blocks, 384 channels (first with stride 2)", "384 × 7 × 7"],
        ["Head", "Global average pooling, Dropout 0.4, FC 128, Dropout 0.3, FC 3", "3"],
        ["Total parameters", "", "7,555,059"],
    ],
    "Stages of the complex CNN.",
    bold_rows=(9,),
)

H("4) Transfer learning and fine-tuning", 3)
rn, mb = META["resnet50_tl"], META["mobilenetv3_tl"]
P(
    "We adapted two networks pretrained on ImageNet: ResNet-50 (a deep residual network, "
    f"{rn['params_total']/1e6:.1f} M parameters) and MobileNetV3-Large (a light network for mobile devices, "
    f"{mb['params_total']/1e6:.1f} M parameters). In both, the original 1,000-class output layer was replaced by a "
    "new 3-class layer (with dropout 0.3 for ResNet-50). Training followed two phases (Figures 6 and 7):"
)
B(
    [
        ("Phase 1 – feature extraction (5 epochs): ", "all pretrained layers were frozen and only the new head was "
         "trained with learning rate 0.001."),
        ("Phase 2 – fine-tuning: ", "the deepest stage was unfrozen (layer4 of ResNet-50; the last four blocks of "
         "MobileNetV3) and trained together with the head using a ten times smaller learning rate, 0.0001, for at "
         f"most 25 more epochs with early stopping. This leaves {rn['params_trainable_final']/1e6:.1f} M trainable "
         f"parameters in ResNet-50 and {mb['params_trainable_final']/1e6:.1f} M in MobileNetV3. The early layers stay "
         "frozen because their generic edge and texture filters transfer well and our training set is small."),
    ]
)
F("arch_resnet50_tl.png", "ResNet-50 with transfer learning: frozen and fine-tuned stages and the new classifier head.")
F("arch_mobilenetv3_tl.png", "MobileNetV3-Large with transfer learning.")

# ------------------------------------------------------------------ IV
H("IV. Experimental Results", 2)
H("1. Overall performance on the test set", 3)
P(
    "All five models were evaluated once on the held-out test set of 627 images. Table 7 reports accuracy and the "
    "macro-averaged precision, recall and F1-score, together with model size and training time."
)
rows = []
for k, name in MODELS:
    m, meta = M[k], META[k]
    rows.append([name, f4(m["accuracy"]), f4(m["precision_macro"]), f4(m["recall_macro"]), f4(m["f1_macro"]),
                 f"{meta['params_total']/1e6:.2f} M", f"{meta['train_seconds']:.0f} s"])
T(
    ["Model", "Accuracy", "Precision", "Recall", "F1-score", "Params", "Train time"],
    rows,
    "Test-set performance of all models (macro average over the three classes).",
    bold_rows=(4,),
)
F("training_curves.png", "Training and validation loss and accuracy per epoch. The dashed line marks the start of fine-tuning.")
P(
    "Transfer learning gave the best results. MobileNetV3 reached 97.61 % accuracy and a macro-F1 of 0.9781, and "
    "ResNet-50 was close behind with 97.29 % and 0.9752. The two networks designed from scratch stayed around 95.4–95.7 % "
    "accuracy, about two points lower. With only 2,923 training images, a network that starts from features learned "
    "on 1.2 million images has a clear advantage. MobileNetV3 was also the most efficient of the pretrained models: "
    "it has 5.6 times fewer parameters than ResNet-50 and trained in 278 s instead of 467 s."
)
P(
    "The complex CNN was only slightly better than the simple CNN (F1 0.9596 against 0.9566 with augmentation) "
    "despite having 17 times more parameters. Its validation loss also fluctuated strongly during the first 20 "
    "epochs (Figure 8), a sign that 7.6 million weights trained from random initialisation are a lot for this "
    "amount of data. Augmentation did not raise the simple CNN's test score (0.9566 with and 0.9577 without), but the "
    "curves show its effect on training: without augmentation the training loss kept falling while the validation "
    "loss stopped improving after about 10 epochs, and early stopping ended the run at epoch 18. With augmentation "
    "the gap between the two curves stayed small and the model kept improving until epoch 25."
)

H("2. Per-class results and confusion matrices", 3)
pc_rows = []
for k, name in [("simple_cnn", "Simple CNN"), ("complex_cnn_aug", "Complex CNN + Aug"), ("mobilenetv3_tl", "MobileNetV3 TL")]:
    for cls in ["COVID", "NORMAL", "PNEUMONIA"]:
        r_ = M[k]["report"][cls]
        pc_rows.append([name if cls == "COVID" else "", cls, f4(r_["precision"]), f4(r_["recall"]), f4(r_["f1-score"]), int(r_["support"])])
T(
    ["Model", "Class", "Precision", "Recall", "F1-score", "Support"],
    pc_rows,
    "Per-class test results of one model from each family.",
)
F("confusion_matrices.png", "Confusion matrices on the test set (rows: true class, columns: predicted class).")
P(bold_prefix="Does the model confuse COVID-19 with Pneumonia? ", text=
    "Almost never. COVID-19 recall is above 0.96 for every model, and ResNet-50 recognised all 161 COVID-19 test "
    "images. Only the complex CNN mistook COVID-19 for Pneumonia (5 images), and no model predicted a Pneumonia image "
    "as COVID-19 except the simple CNN without augmentation (3 images)."
)
P(bold_prefix="Which class is the hardest? ", text=
    "Normal versus Pneumonia. Nearly all remaining errors are between these two classes, and the most common one is "
    "a Pneumonia image predicted as Normal: 10 to 17 of the 252 Pneumonia test images depending on the model "
    "(10 for MobileNetV3, 11 for ResNet-50). In a clinical setting this is a false negative, the more dangerous kind "
    "of error, because a sick patient would be reported as healthy. Mild pneumonia can produce only faint opacities, "
    "and at 224 × 224 pixels some of this detail is lost, which may explain why these cases are missed."
)

H("3. Where does the CNN look? Grad-CAM analysis", 3)
F("gradcam_resnet50_tl.png", "Grad-CAM heatmaps of ResNet-50 on test images: three correct predictions per class and six errors (red titles). Red areas had the most influence on the prediction.")
P(
    "For most correct predictions the heatmaps fall inside the lung fields: Pneumonia images light up across both "
    "lungs, where diffuse opacities are expected, and COVID-19 images often highlight a localised region of one lung. "
    "However, the maps are not always convincing. In several images, including most of the misclassified ones, the "
    "network focuses on the abdomen below the diaphragm, the neck or the image border. In these cases the decision "
    "is based on regions that carry no information about lung disease."
)

H("4. Limitations", 3)
B(
    [
        ("Possible shortcut learning. ", "As noted in Section II, the Normal and Pneumonia images are paediatric "
         "radiographs from one source, while the COVID-19 images are adult radiographs from other sources. A network "
         "can separate COVID-19 from the other classes just by recognising the patient's age, body shape or the "
         "acquisition style, without learning anything about COVID-19 itself. The near-perfect COVID-19 scores and "
         "the off-lung Grad-CAM activations suggest that this happens at least partly. DeGrave et al. (2021) reported "
         "the same problem in many published COVID-19 detectors."),
        ("No patient identifiers. ", "We removed duplicate images, but different radiographs of the same patient could "
         "still fall into different subsets."),
        ("Single run. ", "Each model was trained once with one seed. With 627 test images, a difference of 0.3 % "
         "(two images), such as between ResNet-50 and MobileNetV3, is not statistically meaningful."),
        ("Not a diagnostic tool. ", "The models were evaluated on one public dataset only and have not been tested on "
         "data from other hospitals."),
    ]
)

# ------------------------------------------------------------------ V
H("V. Conclusion", 2)
P(
    "We built a complete pipeline for classifying chest X-rays into COVID-19, Pneumonia and Normal and covered all "
    "five requirements of the assignment. The data audit turned out to be an important step: 1,051 of the 5,228 "
    "images were duplicates, near-duplicates or label conflicts, and removing them prevented an optimistic, leaked "
    "test score. On the cleaned data, a simple LeNet-style CNN with 0.43 M parameters already reached 95.5 % test "
    "accuracy. A deeper network with Inception and residual blocks gave only a small gain, while transfer learning "
    "gave the largest improvement: fine-tuned MobileNetV3-Large reached 97.6 % accuracy and a macro-F1 of 0.978 with "
    "4.2 M parameters."
)
P(
    "The error analysis showed that COVID-19 is almost never confused with Pneumonia and that the hardest distinction "
    "is between Normal and Pneumonia, where missed pneumonia cases are the main error. Grad-CAM showed that the "
    "models mostly look at the lungs but sometimes rely on regions outside them, and the different sources of the "
    "classes mean that part of the high COVID-19 accuracy may come from shortcuts rather than disease patterns. "
    "Future work could segment the lungs before classification so that the network can only use lung tissue, use "
    "a dataset in which all classes come from the same hospitals and age groups, repeat the experiments with several "
    "seeds or k-fold cross-validation, and test the best model on an external dataset."
)

# ------------------------------------------------------------------ References
H("References", 2)
refs = [
    "Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, \"Gradient-based learning applied to document recognition,\" Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.",
    "A. Krizhevsky, I. Sutskever and G. E. Hinton, \"ImageNet classification with deep convolutional neural networks,\" in Advances in Neural Information Processing Systems (NeurIPS), 2012.",
    "K. Simonyan and A. Zisserman, \"Very deep convolutional networks for large-scale image recognition,\" in International Conference on Learning Representations (ICLR), 2015.",
    "C. Szegedy et al., \"Going deeper with convolutions,\" in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.",
    "K. He, X. Zhang, S. Ren and J. Sun, \"Deep residual learning for image recognition,\" in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.",
    "A. Howard et al., \"Searching for MobileNetV3,\" in IEEE/CVF International Conference on Computer Vision (ICCV), 2019.",
    "S. Ioffe and C. Szegedy, \"Batch normalization: Accelerating deep network training by reducing internal covariate shift,\" in International Conference on Machine Learning (ICML), 2015.",
    "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, \"Dropout: A simple way to prevent neural networks from overfitting,\" Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.",
    "I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in International Conference on Learning Representations (ICLR), 2019.",
    "R. R. Selvaraju et al., \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in IEEE International Conference on Computer Vision (ICCV), 2017.",
    "A. J. DeGrave, J. D. Janizek and S.-I. Lee, \"AI for radiographic COVID-19 detection selects shortcuts over signal,\" Nature Machine Intelligence, vol. 3, pp. 610–619, 2021.",
    "A. Zhang, Z. C. Lipton, M. Li and A. J. Smola, Dive into Deep Learning. Cambridge University Press, 2023. Available: https://d2l.ai",
    "S. Kumar, \"COVID-19 Pneumonia Normal Chest X-ray Images,\" Kaggle dataset. Available: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images",
    "A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
]
for i, ref in enumerate(refs, 1):
    md.append(f"{i}. {ref}")

OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
print("saved", OUT)
