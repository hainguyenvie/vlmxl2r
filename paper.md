## Open-Set Recognition in the Age of

## Vision-Language Models

```
Dimity Miller, Niko Sünderhauf, Alex Kenna, and Keita Mason
```
```
Queensland University of Technology, Brisbane, Australia⋆⋆
{d24.miller, niko.suenderhauf}@qut.edu.au
```
```
Abstract.Are vision-language models (VLMs) for open-vocabulary per-
ception inherently open-set models because they are trained on internet-
scale datasets? We answer this question with a clearno– VLMs in-
troduce closed-set assumptions via their finite query set, making them
vulnerable to open-set conditions. We systematically evaluate VLMs for
open-set recognition and find they frequently misclassify objects not con-
tained in their query set, leading to alarmingly low precision when tuned
for high recall and vice versa. We show that naively increasing the size
of the query set to contain more and more classes doesnotmitigate this
problem, but instead causes diminishing task performanceandopen-set
performance. We establish a revised definition of the open-set problem
for the age of VLMs, define a new benchmark and evaluation protocol
to facilitate standardised evaluation and research in this important area,
and evaluate promising baseline approaches based on predictive uncer-
tainty and dedicated negative embeddings on a range of open-vocabulary
VLM classifiers and object detectors.
```
### 1 Introduction

```
In 2013, Scheirer et al. [38] described the closed-set assumption ingrained in al-
most all vision models at the time: all test classes are known during training and
the model is never exposed to novel or unknown classes during testing. Real-
world applications challenge this assumption and test in open-set conditions –
encountering unexpected objects not in the training dataset – and causing po-
tentially dangerous overconfident misclassifications [4,10,27,39]. Open-set recog-
nition [38] was introduced to address this issue by evaluating the ability of vision
models to identify and reject open-set inputs as unknown.
Vision-language models (VLMs) have recently revolutionised the fields of im-
age classification [14,19,34] and object detection [15,49,51]. Trained on vast and
diverse internet-scale datasets, VLMs recognise an extensive variety of classes
andseeminglyare open-set by default [43]. Our paper challenges this view.
In the age of VLMs, we introduce an updated definition of the open-set prob-
lem – while previous definitions focused on a finitetrainingset [38], we show that
⋆⋆The authors acknowledge ongoing support from the QUT Centre for Robotics.
Evaluation code is publicly available atgithub.com/dimitymiller/openset_vlms
```
# arXiv:2403.16528v2 [cs.CV] 19 Jul 2024


2 D. Miller et al.

```
a photo of
a 92.8%dog.
```
```
Vision-Language
Model
(CLIP)
```
a photo ofa {object}. (^)
a photo ofa {object}. (^)
dog
I·T 2
I·T 1
I·T 3
I·T 4
I·T 5
I·T 6
I·T 7
a photo ofa horse.
94.9%
I·T 2
I·T 1
I·T 3
I·T 4
I·T 5
I·T 6
car
plane
cat
dog
bird
horse
ship
car
plane
cat
bird
horse
ship
Traditional open-set recognition
Vision-Language Model open-set recognition
Query set
containing true class
Training* (^) Testing
Open-set
Training
Querying Testing
Open-set
* Training = Querying
Query set
without true class
True positive
prediction
Open-set error
Fig. 1:Left: Traditional open-set recognition arises when testing classes outside a
model’s finite training class set. While VLMs are trained on internet-scale datasets
containing most conceivable object classes, they use a finite query set for classification.
Open-set recognition arises when test classes are not included in the query set. Right:
When testing an object present in the predefined query set, VLMs often correctly
classify the object. When testing an object that is not present in the query set (i.e.
an open-set object), VLMs often misclassify the object as a query class with high
confidence (i.e. an open-set error).
VLMs impose closed-set assumptions through a finitequeryset. As illustrated
in Fig. 1, VLMs for open-vocabulary image classification or object detection
are still evaluated under closed-set conditions: they compare image embeddings
with text embeddings from a predefined, finite query set of class labels. This in-
troduces a closed-set assumption that all classes encountered during testing are
included in this query set. Open-set conditions emerge when VLMs encounter
objects that are not included in the query set (unknown objects). As we show
in Section 5, even state-of-the-art VLMs heavily degrade in performance and
misclassify unknown objects as belonging to the query set with high confidence.
In theory, this problem disappears if a VLM’s query set containsevery pos-
sible class label in a vocabulary. However, we show that naively increasing the
query set to contain more and more class labels results in worse task perfor-
mance, where increasing numbers of misclassifications occur (see Sec. 5.4). The
serious concerns about deploying vision models in safety-critical applications
voiced in the pre-VLM era [2, 39] continues to be relevant and underscore the
urgent need for further research into the open-set problem for VLMs.
Our paper is the first to systematically investigate and quantify the open-set
vulnerabilities of VLM-based open-vocabulary object detectors and zero-shot
image classifiers. We re-define the open-set problem for VLMs (Section 3) and
compare promising baseline strategies, leveraging predictive uncertainty (Sec-
tion 5.1) and different approaches to inserting dedicatednegative(unknown)
class representations into the VLM pipeline in Section 5.2. We propose a new


```
Open-Set Recognition in the Age of Vision-Language Models 3
```
benchmark and evaluation protocol in Section 4 to foster further research and
evaluation of VLMs for open-set recognition.

### 2 Background

2.1 Vision-Language Models for Zero-shot Classification

A recently emerged foundation model is thevision-language model (VLM), which
learns a multi-modal feature space that can jointly represent visual features
and text [1, 19, 34, 47, 52] (and in some cases other modalities as well [14, 48]).
Large-scale contrastive, self-supervised learning on internet-scale datasets pro-
vides VLMs with a rich feature representation that can be adapted for many
downstream tasks, one of which is zero-shot or open vocabulary image classifi-
cation [1, 14, 19, 34, 47, 48, 52]. Despite not being trained for classification tasks
or datasets, VLMs generalise surprisingly well and achieve near state-of-the-art
supervised learning performance [47]. Alongside this, a distinguishing advantage
of VLMs is their adaptability to different datasets with different classes – the
set of classes can be changed at will depending on the dataset [19,34].

2.2 Vision-Language Models for Open Vocabulary Object Detection

Unlocked by the advances in VLMs for zero-shot classification, [51] proposed
open vocabulary object detection, where detectors must generalise to an arbi-
trary set of object classes at test time, even object classes that were not seen
during training [51]. This is related to our proposed open-set recognition, but
assumes that novel object classes are added to the query set, whereas we test
against object classes that are not present in the query set.
Existing open vocabulary detectors leverage VLM backbones to enable gen-
eralisation to new object classes, embedded within Faster R-CNN-style architec-
tures [11,12,15,20,22,32,42,44,51,55,56] or DETR-style architectures [45,49].
CLIP [34] is frequently utilised by these open vocabulary detectors, directly in-
cluded in the architecture via its text encoder [11,12,15,22,32,42,44,49,55,56]
or in its entirety [20,45], or used as a supervision signal during training [11,12,
15, 32, 42, 44, 49, 55]. We refer the reader to [43] for a comprehensive survey on
open vocabulary learning.

2.3 Open-set Recognition

Scheirer et al. [38] introduced open-set recognition to challenge the prevailing
closed-set assumptions of image recognition systems.Closed-setmodels only test
on samples from a predefined set of classes that were known during model train-
ing [38]. Inopen-setrecognition, samples can instead come from an expansive
range of classes that cannot be predefined, and therefore are unknown during
model training [38].Open-set recognition tests the ability of models to identify
and reject unknown samples that do not belong to one of the training classes.


4 D. Miller et al.

The open-set concept has been explored widely in computer vision, including
in image classification [4, 7, 29,31,41,46,54], object detection [10,16,27,28,57],
image segmentation [6,17,33] and other vision tasks [3,25,26,30,36].
Our paper is the first to systematically investigate the open-set problem for
VLMs, and our findings challenge the assumption that the open-set problem has
been solved for VLMs due to their exposure to internet-scale training data. The
following sections define the closed-set assumptions existing in current VLMs,
demonstrate that VLMs continue to be vulnerable to open-set conditions and
evaluate a number of baseline strategies for open-set recognition with VLMs.

### 3 Problem Definition

The core concept behind VLMs, mapping images and text into a joint embed-
ding space, makes them powerful foundations for image classification and ob-
ject detection. Classification is performed by comparing an image embedding
eIwith text embeddingsEq={eq 1 ,...,eqN}obtained from a set of query la-
belsQ={q 1 ,...,qN}which are class names such as “dog”^1. The similarities
between the image embedding and all text embeddings fromQare measured by
a similarity functionsq= sim(eI,Eq), e.g. using the cosine similarity. The text
embedding with the highest similarity to the image embedding, i.e.argmaxsq,
determines the predicted class label. The process is the same for object detec-
tion, witheInow representing the embedding of an image region corresponding
to an object proposal.
Open-set and Closed-set:While the pre-VLM definition of the closed-set
and open-set concepts focused on the challenges arising from a finitetraining
set [38], our updated definition emphasises the limitations of the finitequeryset:
The closed-set assumption in VLMs is that all classes seen during testing are
predefined prior to testing and included in the query set. An open-set situation
arises when the VLM encounters an object or image class that is not part of this
predefined query setQ.

3.1 Baseline Approaches to Open-Set Recognition for VLMs

In an open-set situation, the VLM should be able to identify an input not rep-
resented by the query set and reject it is asunknown, instead of misclassifying it
as one of the query classes fromQ. Similar to the pre-VLM open-set literature,
there are multiple ways of recognising an input as unknown [13]. We investi-
gate baseline methods using predictive uncertainty and different approaches of
negative embeddings.
Using Uncertainty:Different measures of predictive uncertaintyψ can be
used to reject inputs asunknown, for example based on the class similarities, e.g.

(^1) Most methods insert an additional step here and generate a set of prompts from each
query labelqi, such as {“A photo of a dog”, “A picture of a dog”,... } and compare
the embeddings of these prompts witheI. For clarity of notation, we omit this step.


```
Open-Set Recognition in the Age of Vision-Language Models 5
```
ψCosine= max(sq)orψSoftmax= max(softmax(sq)), or the entropy of the soft-
max similaritiesψEntropy=−H(softmax(sq)). Only inputs that pass a threshold
testψ≥θare classified according toargmaxsq. Otherwise, the input is declared
to be open-set and unknown.
Dedicated Negative Embeddings:Another approach is to create a set of
augmented query embeddingsE ̃q=Eq∪Enby addingMdedicatednegative
class embeddingsEn={en 1 ,...,enM}. The input image or detection can then
be rejected asunknownif the image embedding is most similar to one of those
negativeembeddingsenirather than a query class embeddingeqj. These negative
embeddings can be created in one of two ways: (1) by creatingnegative words,
which are then passed through the VLM text encoder to create the negative
embeddings, or (2) by directly creatingnegative embeddings.
We investigate two baseline options for negative embeddings: arandom
words method, where we generate random strings with random length be-
tween 2-8 characters (e.g. “braxl”); and arandom embeddingsmethod, where
we randomly draw M embedding vectors from a Gaussian distribution, i.e.
eni ∼ N(μ,Σ), using the mean and standard deviation from the query em-
bedding vectorsEq. We compare the effectiveness of these approaches, includ-
ing for different numbers ofM random negative queries, in our experiments.
Interestingly, many open vocabulary object detectors already use a negative
query approach, though typically with a only single embedding vector of all
zeroes [22,45,51,55,56].

### 4 Evaluation Protocol

4.1 Creating an Open-set Recognition Dataset for VLMs

We now introduce how to use existing image classification or object detection
datasets to evaluate open-set recognition. In short, we test all images twice: once
in a standard closed-set manner with all dataset classes included in the query
set (see Fig. 1 top-right), and once in an open-set manner with a query set
containing all dataset classes not present in the image (see Fig. 1 bottom-right).
Classification and object detection datasets have a pre-defined set ofKla-
belled target classesL. In image classification, there is a single corresponding
class labely∈Lper image. In object detection, there are multiple ground truth
objects in each image and each has a corresponding class label. For both tasks,
we can define a setYthat contains the ground truth class labels for each image:
Ycontains a single label in classification and multiple labels in object detection.
Testing Closed-set Recognition: We first test images in a standard manner
to establish the closed-set performance of the VLM and identify alltrue positive
(TP)predictions. This involves testing all images with a query set identical to
the dataset class list,Q=L. To identify the TP predictions, we follow standard
protocol for classification and object detection datasets – in classification, a
TP has a predicted class matching the ground-truth label; in object detection,
this involves a object-detection assignment process that requires correct object


6 D. Miller et al.

classification and localisation (see Supplementary Sec. S1). TPs represent correct
closed-set predictions that should not be rejected asunknown, and we collect the
uncertainty measuresψassociated with all TPs for later evaluation.
For object detection datasets, unlabelled objects are present in the back-
ground of the image [10]. These unlabelled objects are not in the query setQ,
and therefore meet our definition for open-set objects. Detections of these open-
set objects will present as false positives (FPs) in a closed-set test, however not
all FPs arise from open-set objects – FPs in object detection can also arise due
to poor object localisation, misclassification between query classes, or duplicate
detections of a query class (see [5]). Rather than attempting to categorise differ-
ent FP types to distinguish open-set errors, our approach definitively identifies
open-set errors for both object detection and image classification.
Testing Open-set Recognition: To evaluate open-set recognition, we test
each image in the dataset with a modified query set,Q ̃that contains all the
class labels notin the image ground-truth class list,Q ̃=L−Y. This ensures
the image only contains object classes that are not in the modified query set –
every single prediction must be anopen-set error (OSE)^2. In image classification,
this modifies the query set to haveK− 1 labels. In object detection, the modified
query set size depends on the number of unique object classes in the image. We
collect the uncertainty measuresψof all OSE predictions for evaluation.

4.2 Metrics

We use performance metrics that measure the ability of the VLM to reject OSEs
(with high uncertainty or with anegativeclass), while maintaining the closed-
set task performance.
Precision-Recall Curves: Open-set recognition can be formulated as a binary
classification task that uses uncertainty as the decision threshold. Correct closed-
set predictions (introduced as TPs in Sec. 4.1) and open-set errors (OSEs) are
considered as the positive and negative class respectively. We construct precision-
recall curves, which are well-suited to this problem as they are affected by the
size of the negative class (OSEs) [37]^3 as well as prediction uncertainty. High
precision indicates that little-to-none negatives (OSEs) remain after the uncer-
tainty thresholding. High recall indicates that most positives (TPs) remain after
the uncertainty thresholding. We summarise the performance of the precision-
recall curves with 3 core metrics: (1)Area under the PR curve (AuPR),
(2)Precision at 95 %Recall, and (3)Recall at 95 %Precision.
Task Metrics: We use the established task metrics used to evaluate classifica-
tion and detection – top-1 accuracy [35] and mean Average Precision (mAP) [23]

(^2) This assumes accurate labelling of ground-truth classes, though most datasets will
contain some label error. This approach is unsuitable for weakly-labelled datasets.
(^3) ROC curves do not have this property – we discuss this further in Sec. S3 of the
Supplementary Material where we report AuROC for the object detection experi-
ments.


```
Open-Set Recognition in the Age of Vision-Language Models 7
```
respectively. We evaluate mAP at an IoU threshold of 0. 5 – we are primarily in-
terested in the classification ability of the detector (rather than localisation) and
this is standard protocol in open vocabulary detection [15,22,45,49,51,55,56].
Absolute Counts:We also report the absolute counts of TPs and OSEs. This
can be particularly useful for object detection, where increased task performance
does not necessarily indicate increased TPs and detectors predict variable num-
bers of OSEs. We stress thatabsolute counts should not be considered alone as
a performance indicator, e.g. an increased number of OSEs can be acceptable if
all OSEs have very high uncertainty.

4.3 VLM Baselines

We test with six VLM classifiers – CLIP [34], ALIGN [19], CoCa [47], Image-
Bind [14], SigLIP [53] and LanguageBind [59] – and seven open vocabulary object
detectors – OVR-CNN [51], ViLD [15], OV-DETR [49], RegionCLIP [55], De-
tic [56], VLDet [22] and CORA [45] (see Supplementary Sec. S2 for implementa-
tion details). These VLMs cover a range of architectures and training paradigms
and have publicly-available implementations. Our goal with testing this range
of VLMs is not to find the “best” VLM for open-set recognition, but instead
to exhibit the general vulnerability of VLMs to open-set conditions. For each
VLM, we extract the predictionSoftmaxscore,Cosinesimilarity, andEntropy
of the softmax distribution as baseline uncertainty measures. Some VLMs use
Sigmoid activation rather than Softmax [22,45,49,53,56] – for these methods, we
only report theSigmoidscore as its results are identical to the cosine similarity
and the entropy is unsuitable as the scores are not normalised to a distribution.
We do not impose a minimum score threshold on any predictions, as the AuPR
metric captures performance over all possible thresholds.

4.4 Datasets

We focus on general object recognition and test with the large-scale and prevail-
ing datasets used to benchmark image classification and object detection: For
image classification, we test with the ImageNet1k validation dataset [9, 35]. It
contains 50 , 000 images from 1000 classes and is publicly available for research.
For object detection, we test with the MS COCO 2017 validation dataset [23].
It contains 4 , 952 images with 36 , 781 annotated objects from 80 classes and is
publicly available for research. We show results for domain-specific classification
datasets in the Supplementary Sec. S4.

### 5 Experiments and Results

5.1 State-of-the-art VLMs Perform Poorly in Open-set Conditions.

Classification:
Tab. 1 shows that all six VLM classifiers perform poorly for open-set recogni-
tion. For scenarios that require high recall of true positive predictions (i.e.95%


8 D. Miller et al.

Table 1:The open-set performance of state-of-the-art VLM classifiers when tested on
ImageNet1k. Arrows indicate direction of optimal performance and - indicates operat-
ing point could not be achieved.

AuROC AuPR P@95R R@95P TP OSE Accuracy
Classifier Uncertainty ↑↑↑↑↑↓↑

CLIP [34]

```
Softmax 79.7 71.5 46.2 3.
72.2 61.3 42.5 0.1 31,023 50,000 62.1Cosine
Entropy 80.2 72.3 45.4 2.
```
ALIGN [19]

```
Softmax 81.0 74.7 48.1 8.
72.4 62.1 44.8 - 32,618 50,000 65.2Cosine
Entropy 80.9 75.0 46.8 8.
```
CoCa [47]

```
Softmax 79.0 71.7 46.0 3.
73.2 63.0 43.3 0.1 31,725 50,000 63.4Cosine
Entropy 80.5 73.0 46.8 3.
```
ImageBind [14]

```
Softmax 82.8 79.1 52.8 5.
79.2 73.9 50.7 0.2 38,405 50,000 76.8Cosine
Entropy 84.3 80.1 56.3 5.
```
SigLIP [53] Sigmoid 81.0 76.2 52.1 2.2 37,851 50,000 75.

LanguageBind [59]

```
Softmax 83.9 80.7 54.7 10.
81.6 77.4 52.8 0.8 39,243 50,000 78.5Cosine
Entropy 85.4 81.7 58.4 11.
```
recall), approximately every second prediction returned is an open-set error (i.e.
precision between 46 .2%and 58 .4%). For scenarios that require high precision
of predictions (i.e.95%precision), less than 12% of the true positive predictions
are retained (i.e. recall between 3.4% and 11.1%). Comparing across uncertainty
types, the VLM classifiers show better uncertainty performance when using the
predicted Softmax score or Entropy rather than the predicted Cosine similarity.
This suggests raw similarity in VLM feature space is not the best indicator of
open-set error, with measures of relative similarity comparative to other classes
offering better performance.

Object Detection:All seven of the tested VLM object detectors are vulnerable
to open-set errors – see Tab. 2. Unlike VLM classifiers, most VLM detectors
already contain a negative query to capture objects not in the query set (often
referred to as thebackgroundclass). Despite this, the VLM detectors produce
open-set error counts in the range of 100 , 000 to 1 , 500 , 000 when tested on only
4,952 images, i.e. between 20 and 300 open-set error per image on average.

High open-set error counts are not inherently a problem if they are produced
with high uncertainty. Yet for all tested detectors, the baseline uncertainty meth-
ods cannot adequately distinguish true positive detections from open-set errors.
When thresholding for high recall of true positives,at bestonly every fifth de-
tection is a true positive (i.e. precision ranges between 5 .9%and 21 .9%). When


```
Open-Set Recognition in the Age of Vision-Language Models 9
```
Table 2:The open-set performance of state-of-the-art open-vocabulary object detec-
tors (with ResNet backbone indicated), tested on COCO. Arrows indicate the direction
of optimal performance.

Negative AuPR P@95R R@95P TP OSE mAP
Detector Uncertainty Embedding ↑↑↑↑↓↑

OVR-CNN (R50) [51]

```
Softmax Zero-Emb 75.4 13.0 42.
Zero-Emb 77.2 15.3 43.6 13,544 190,146 34.7Cosine
Entropy Zero-Emb 63.2 7.5 43.
```
ViLD (R152) [15]

```
Softmax “background” 60.1 7.5 15.
0.1 16,011 1,485,600 46.4Cosine “background” 33.1 6.
Entropy “background” 62.2 8.4 16.
```
OV-DETR (R50) [49] Sigmoid None 75.8 5.9 47.3 15,818 1,485,600 43.

RegionCLIP (R50) [55]

```
Softmax Zero-Emb 73.1 9.9 39.
Zero-Emb 74.0 7.1 40.9 15,741 493,940 44.6Cosine
Entropy Zero-Emb 35.4 3.9 17.
```
Detic (R50) [56] Sigmoid Zero-Emb 72.6 8.1 42.2 14,670 495,200 39.

VLDet (R50) [22] Sigmoid Zero-Emb 79.8 21.9 48.5 14,751 112,923 40.

CORA (R50) [45] Sigmoid Zero-Emb 67.2 8.1 24.7 13,763 495,200 31.

thresholding for high precision,at bestless than half of true positives are re-
called (i.e. recall ranges between 16.3% and 48.5%). We include some qualitative
examples of highly confident open-set errors in the Supplementary Sec. S6.
Notably, the precision-recall characteristics for VLM classification and object
detection differ significantly – VLM classifiers achieve better performance at high
recall, whereas VLM detectors achieve better performance at high precision.
This suggest that while the VLM classifiers struggle with over-confident open-
set errors, the VLM detectors suffer from under-confident true positives. We
include histograms that visualise this trend in the Supplementary Sec. S7.
We also consider how the performance of uncertainty differs for identifying
open-set error versus closed-set misclassifications, with results and a detailed dis-
cussion in the Supplementary Sec. S5. We find that the performance of softmax
uncertainty is lower for open-set error than for closed-set error in both classifica-
tion and detection tasks. In classification, open-set performance is particularly
decremented when thresholding for high precision – likely due to a greater pro-
portion of overconfident open-set error – whereas detection primarily degrades
with increased numbers of errors, with ViLD for example producing 11. 7 ×more
open-set than closed-set misclassifications.

5.2 Can Negative Queries Improve Open-set Performance?

Classification: Negative class queries provide the VLM classifiers with the
ability to select “unknown” when testing an input. When used in larger quan-
tities, negative queries can capture and remove significant portions of open-set
error, yet this happens in trade-off with a reduction in closed-set performance –


10 D. Miller et al.

Table 3:The open-set performance of state-of-the-art VLM classifiers on ImageNet1k
when used with negative random words (R-W) or random embeddings (R-E).

```
Negative AuPR P@95R R@95P TP OSE Accuracy
Classifier Embedding ↑↑↑↑↓↑
None 71.5 46.2 3.4 31,023 50,000 62.
CLIP [34] 500 R-W 73.4 48.1 3.3 29,96141,428 59.
(Softmax) 2500 R-W 72.8 49.8 1.6 28,65735,092 57.
500 R-E 71.9 47.2 2.5 30,160 44,673 60.
2500 R-E 71.7 47.1 0.9 29,291 40,497 58.
None 74.7 48.1 8.3 32,618 50,000 65.
ALIGN [19] 500 R-W 75.3 48.9 10.4 31,73143,198 63.
(Softmax) 2500 R-W 74.6 49.5 8.9 30,58837,738 61.
500 R-E 74.9 48.7 8.6 32,342 47,228 64.
2500 R-E 75.1 49.7 7.2 31,462 41,210 62.
None 71.7 46.0 3.4 31,725 50,000 63.
CoCa [47] 500 R-W 75.4 49.6 0.1 30,27639,528 60.
(Softmax) 2500 R-W 75.9 50.9 6.0 29,10734,360 58.
500 R-E 71.7 46.0 3.4 31,723 49,966 63.
2500 R-E 71.7 46.1 3.4 31,719 49,866 63.
None 79.1 52.8 5.0 38,405 50,000 76.
ImageBind [14] 500 R-W 83.8 63.1 7.7 35,85929,172 71.
(Softmax) 2500 R-W 84.6 66.1 8.3 34,17922,872 68.
500 R-E 81.0 57.3 9.0 36,966 38,636 73.
2500 R-E 81.7 59.0 11.2 35,954 33,349 71.
None 76.2 52.1 5.2 37,851 50,000 75.
SigLIP [53] 500 R-W 77.6 56.7 2.3 36,11133,287 72.
(Sigmoid) 2500 R-W 78.8 60.2 2.4 34,50726,110 69.
500 R-E 84.7 73.0 3.6 22,772 9,570 45.
2500 R-E 90.1 83.9 7.9 10,561 2,258 21.
None 80.7 54.7 10.5 39,243 50,000 78.
LanguageBind [59] 500 R-W 84.7 64.9 13.1 36,59628,337 73.
(Softmax) 2500 R-W 85.4 68.2 11.7 34,54021,230 69.
500 R-E 80.8 54.9 10.8 39,226 49,646 78.
2500 R-E 80.9 55.1 10.9 39,210 49,032 78.
```
see Tab. 3. For example, ImageBing using 2 , 500 random word negative queries
effectively halves the open-set error, but this occurs in trade-off with a closed-set
accuracy reduction of 8 .4%(approximately 4 , 000 of the closed-set images are
incorrectly classified as one of the random words).
While negative queries noticeably reduce the raw open-set error count, this is
not always mirrored with increased performance in the open-set uncertainty met-
rics. We present the change in closed-set and open-set performance against the
number of negative query points in Fig. 2. Generally the use of negative random
words appears the most effective, with CoCa, ImageBind, SigLIP and Language-
Bind showing improved open-set performance with minimal loss of closed-set
performance. For some of the classifiers, there is negligible change in open-set
performance despite the reduction in open-set error. We explore this further in
the Supplementary Sec. S8, where we show that in some VLMs the negative


```
Open-Set Recognition in the Age of Vision-Language Models 11
```
```
70
```
```
80
```
```
90
```
```
100
```
```
AuPR (%)
```
```
CLIP ALIGN CoCa ImageBind SigLIP LanguageBind
```
```
0 50
Acc (%)
```
```
70
```
```
80
```
```
90
```
```
100
```
```
AuPR (%)
0 50
Acc (%)
```
```
0 50
Acc (%)
```
```
0 50
Acc (%)
```
```
0 50
Acc (%)
```
```
0 50
Acc (%)
```
```
0 10 50 100 250 500 1000 2500 5000
Number of negative queries
```
```
Words
Embeddings
```
Fig. 2:The trade-off between closed-set and open-set performance with increasing
number of negative queries for different VLM classifiers.

```
60
```
```
80
```
```
100
```
```
AuPR (%)
```
```
OVR-CNN Vild OV-DETR RegionCLIP Detic VLDet CORA
```
```
0 20 40
mAP (%)
```
```
60
```
```
80
```
```
100
```
```
AuPR (%)
0 20 40
mAP (%)
0 20 40
mAP (%)
0 20 40
mAP (%)
0 20 40
mAP (%)
0 20 40
mAP (%)
0 20 40
mAP (%)
```
```
0 1 3 5 10 25 50 100 250 500
Number of negative queries
```
```
Words
Embeddings
```
Fig. 3:The trade-off between closed-set and open-set performance with increasing
number of negative queries for different VLM object detectors.

queries only capture and remove “easy” open-set errors, i.e. open-set errors that
already have high uncertainty.

Object Detection: The VLM detectors do not show a consistent response
to the use of negative queries. Different detectors show different sensitivities to
negative query counts, different trade-off rates between closed-set and open-set
performance, and differences in effectiveness of embedding versus word nega-
tive query types – see Tab. 4 and Fig. 3. Looking at Fig. 3, we can consider
the sensitivity, trade-off rate and response of the detectors to negative queries:
greater performance changes with increasing queries indicates sensitivity; large
decreases in mAP with more queries indicates increased trade-off rates between
open-set and closed-set performance; and a positive response to the negative
queries presents as an increase in AuPR when increasing queries (negative re-
sponse decreases AuPR).


12 D. Miller et al.

Table 4:The open-set performance of state-of-the-art open vocabulary object de-
tectors (with ResNet backbone indicated) when additional negative embeddings are
included, tested on COCO. Arrows indicate the direction of optimal performance. Neg-
ative embeddings introduced as words are higlighted gray.

```
Negative AuPR P@95R R@95P TP OSE mAP
Detector Embedding ↑↑↑↑↓↑
```
```
OVR-CNN (R50) [51]
```
```
Zero-Emb 75.4 13.0 42.9 13,544 190,146 34.
+5 Rand-Embs 75.6 13.1 43.1 13,543 186,120 34.
+100 Rand-Embs 78.3 18.2 45.6 13,129 109,267 34.
```
```
ViLD (R152) [15]
```
```
“background” 60.1 7.5 15.4 16,0111,485,600 46.
+5 Rand-Words 57.5 6.0 13.4 15,926 1,400,61246.
+100 Rand-Words 55.0 2.7 10.3 15,7721,088,725 47.
+5 Rand-Embs 60.1 7.5 15.2 16,011 1,485,231 46.
+100 Rand-Embs 60.2 7.6 15.2 16,019 1,478,806 46.
```
```
OV-DETR (R50) [49]
```
```
None 75.8 5.9 47.3 15,818 1,485,600 43.
+5 Rand-Words 77.2 8.1 48.3 15,510 923,398 43.
+100 Rand-Words 84.7 27.0 55.7 13,433 239,118 40.
+5 Rand-Embs 76.6 6.6 47.8 15,652 1,178,827 43.
+100 Rand-Embs 81.0 18.2 50.9 14,699 356,380 42.
```
```
RegionCLIP (R50) [55]
```
```
Zero-Emb 73.1 9.9 39.3 15,752 493,949 44.
+5 Rand-Words 73.8 11.1 39.7 15,662 322,596 44.
+100 Rand-Words 77.0 17.5 43.1 15,178 154,262 44.
+5 Rand-Embs 91.8 77.2 44.6 8,828 5,062 27.
+100 Rand-Embs 99.9 99.3 83.6 287 2 1.
```
```
Detic (R50) [56]
```
```
Zero-Emb 72.6 8.1 42.2 14,670 495,200 39.
+5 Rand-Words 72.8 8.5 42.4 14,637 451,633 39.
+100 Rand-Words 74.5 11.5 43.5 14,287 265,373 38.
+5 Rand-Embs 93.4 74.5 61.8 10,070 6,346 30.
+100 Rand-Embs 100.0 100.0 100.0 371 0 0.
```
```
VLDet (R50) [22]
```
```
Zero-Emb 79.8 21.9 48.4 14,751 112,954 40.
+5 Rand-Words 80.1 22.3 48.7 14,672 111,893 40.
+100 Rand-Words 86.5 36.7 58.4 11,132 50,122 33.
+5 Rand-Embs 97.3 91.3 84.9 8,464 1,382 25.
+100 Rand-Embs 99.0 99.3 3.7 2,531 22 5.
```
```
CORA (R50) [45]
```
```
Zero-Emb 67.2 8.1 24.9 13,745 495,200 31.
+5 Rand-Words 66.3 7.6 27.0 13,700 413,636 31.
+100 Rand-Words 69.1 10.1 28.6 13,722 329,393 30.
+5 Rand-Embs 63.1 6.2 21.8 13,689 495,200 31.
+100 Rand-Embs 59.9 6.0 16.7 13,680 495,174 30.
```
OV-DETR is the only detector to show a clear positive response, low trade-off
rate, and high sensitivity to the negative query approach, particularly with neg-
ative embeddings. While OVR-CNN (embeddings), RegionCLIP (words), Detic
(words) also show a positive response and low trade-off rate, this is with very
little sensitivity to increasing numbers of negative queries. In contrast, the open-
set performance of Detic, RegionCLIP, and VLDet are very sensitive to negative


```
Open-Set Recognition in the Age of Vision-Language Models 13
```
```
65 70 75
Accuracy (closed-set performance)
```
```
7272
```
```
73
```
```
74
```
```
75
```
```
76
```
```
77
```
```
78
```
```
79
```
```
80
```
```
81
```
```
AuPR (open-set performance)
```
```
CLIP
ALIGN
CoCa
ImageBind
SigLIP
LanguageBind
```
```
(a)Classification task.
```
```
32.535.037.540.042.545.
mAP (%) (closed-set performance)
```
```
60.
```
```
62.
```
```
65.
```
```
67.
```
```
70.
```
```
72.
```
```
75.
```
```
77.
```
```
80.
```
```
AuPR (%) (open-set performance)
```
```
OVR-CNN
OV-DETR
Vild
RegionCLIP
Detic
VLDet
CORA
```
```
(b)Object Detection task.
```
```
Fig. 4:The relationship between closed-set and open-set performance in VLMs.
```
embeddings, but at a rapid trade-off with closed-set performance. Both ViLD
and CORA show a negative response or no sensitivity to negative queries.
It is tempting to relate these trends to the architecture or training paradigm
of the VLM detectors, however we could not identify many common character-
istics. RegionCLIP, Detic and VLDet all learn from auxiliary datasets that have
been pseudo-labelled with regions [22,55,56], although this does not clearly in-
dicate a reason for high sensitivity and trade-off rate for negative embeddings.
Vild [15] and CORA [45] both heavily rely on CLIP’s visual feature encoding
(either by using CLIP directly [45] or knowledge distillation [15]) – their poor
response is consistent with CLIP in the classification experiments on negative
queries. Yet OV-DETR also uses CLIP’s visual features during training to condi-
tion the object proposals [49], but exhibits the best response to negative queries.

5.3 Is a good classifier all you need for VLM Open-set Recognition?

In traditional open-set recognition, Vaze et. al [41] identified a strong positive
correlation between closed-set and open-set performance. Testing this theory for
VLM open-set recognition, we compare classifier and object detector closed-set
and open-set performance in Fig. 4.
Classification:The correlation observed by [41] appears to hold for VLM clas-
sifiers, with greater accuracy (closed-set performance) relating to greater AuPR
scores (open-set performance). It is worth noting that AuPR is a summary met-
ric and this behaviour may not hold when examining performance at different
operating points on the precision-recall curves.
Object Detection:There is not a clear correlation between a VLM object de-
tector’s closed-set and open-set performance. ViLD in particular challenges this
correlation, with the greatest mAP yet worst AuPR of all the tested detectors.

5.4 How Does Query Set Size Impact Performance?

In Fig. 5, we test performance of the VLM classifiers on ImageNet while pro-
gressively increasing the query set size. We select the additional classes from the


14 D. Miller et al.

(^5010001500) Query Set Size 2000 2500 3000
55
60
65
70
75
Accuracy (closed-set performance) 1000 1500 2000 2500 3000
Query Set Size
68
70
72
74
76
78
80
82
AuPR (open-set performance)
CLIPALIGN
CoCa
SigLIPImageBind
LanguageBind
Fig. 5:Influence of the query set size on closed-set and open-set performance on Ima-
geNet1k classification, showing mean and standard deviation from 10 random seeds.
WordNet hierarchy by choosing leaf nodes that are related, but not too closely
related, to existing ImageNet classes (i.e. share a grandparent in the hierarchy
but do not share a direct parent). Increasing the number of target classes and
thus the query set size decrements both the closed-set and open-set performance
of a VLM classifier. This indicates that testing VLMs with an entire vocabulary
of nouns may not be a feasible approach to avoid open-set recognition.

### 6 Discussion

Open Challenges:The tested baseline uncertainty measures were ineffective
at identifying open-set errors, indicating a need to research better alternatives.
While uncertainty representations for VLMs have been proposed [18, 40], their
utility for open-set recognition remains unexplored.
While the use of negative random embeddings was able to reduce open-set er-
ror, large numbers were typically required to have any effect and this was often at
the cost of falsely captured true positive predictions. Interestingly, some negative
embeddings were much more effective at capturing open-set errors than others
(e.g. the random string “awy” tested with ImageBind, see the Supplementary Sec.
S9). Identifying the characteristics of these embeddings, learning them through
prompt-tuning [8,11,12,21,58] or introducing auto-labelling techniques [50] are
promising directions.
Conclusions:VLMs are increasingly applied in diverse real-world applications,
from robotics to medical image analysis. We hope that by highlighting their
weaknesses, we encourage further research to improve these models and enhance
their positive societal impact. We have clearly demonstrated that VLMs for
open-vocabulary image classification and object detection introduce closed-set
assumptions by relying on a finite query set. We thus refuted the assumption
that vision-language models (VLMs) are inherently open-set models due to their
training on extensive internet-scale datasets. While the investigated baseline ap-
proaches to identify open-set errors showed promise, we found none of them suf-
ficiently robust and reliable for safety-critical applications. Our new benchmark
and evaluation protocol is designed to foster and support this research. Under-
standing VLMs’ limitations ultimately reduces risks, improves transparency, and
ensures safer deployment in critical applications.


```
Open-Set Recognition in the Age of Vision-Language Models 15
```
### References

1. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Men-
    sch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for
    few-shot learning. Advances in Neural Information Processing Systems 35 , 23716–
    23736 (2022)
2. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., Mané, D.: Con-
    crete problems in ai safety. arXiv preprint arXiv:1606.06565 (2016)
3. Bao, W., Yu, Q., Kong, Y.: Evidential deep learning for open set action recognition.
    In: Proceedings of the IEEE/CVF International Conference on Computer Vision.
    pp. 13349–13358 (2021)
4. Bendale, A., Boult, T.E.: Towards open set deep networks. In: Proceedings of the
    IEEE conference on computer vision and pattern recognition. pp. 1563–1572 (2016)
5. Bolya, D., Foley, S., Hays, J., Hoffman, J.: Tide: A general toolbox for identifying
    object detection errors. In: Computer Vision–ECCV 2020: 16th European Confer-
    ence, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. pp. 558–573.
    Springer (2020)
6. Cen, J., Yun, P., Cai, J., Wang, M.Y., Liu, M.: Deep metric learning for open world
    semantic segmentation. In: Proceedings of the IEEE/CVF International Confer-
    ence on Computer Vision. pp. 15333–15342 (2021)
7. Chen, G., Peng, P., Wang, X., Tian, Y.: Adversarial reciprocal points learning
    for open set recognition. IEEE Transactions on Pattern Analysis and Machine
    Intelligence 44 (11), 8065–8081 (2021)
8. Cho, E., Kim, J., Kim, H.J.: Distribution-aware prompt tuning for vision-language
    models. In: Proceedings of the IEEE/CVF International Conference on Computer
    Vision. pp. 22004–22013 (2023)
9. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-
    scale hierarchical image database. In: 2009 IEEE conference on computer vision
    and pattern recognition. pp. 248–255. Ieee (2009)
10. Dhamija, A., Gunther, M., Ventura, J., Boult, T.: The overlooked elephant of
object detection: Open set. In: The IEEE Winter Conference on Applications of
Computer Vision. pp. 1021–1030 (2020)
11. Du, Y., Wei, F., Zhang, Z., Shi, M., Gao, Y., Li, G.: Learning to prompt for open-
vocabulary object detection with vision-language model. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14084–
14093 (2022)
12. Feng, C., Zhong, Y., Jie, Z., Chu, X., Ren, H., Wei, X., Xie, W., Ma, L.: Prompt-
det: Towards open-vocabulary detection using uncurated images. In: European
Conference on Computer Vision. pp. 701–717. Springer (2022)
13. Geng, C., Huang, S.j., Chen, S.: Recent advances in open set recognition: A survey.
IEEE transactions on pattern analysis and machine intelligence 43 (10), 3614–
(2020)
14. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra,
I.: Imagebind: One embedding space to bind them all. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15180–
15190 (2023)
15. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vi-
sion and language knowledge distillation. In: International Conference on Learning
Representations (2022)


16 D. Miller et al.

16. Han, J., Ren, Y., Ding, J., Pan, X., Yan, K., Xia, G.S.: Expanding low-density
    latent regions for open-set object detection. In: Proceedings of the IEEE/CVF
    Conference on Computer Vision and Pattern Recognition. pp. 9591–9600 (2022)
17. Hwang, J., Oh, S.W., Lee, J.Y., Han, B.: Exemplar-based open-set panoptic seg-
    mentation network. In: Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition. pp. 1175–1184 (2021)
18. Ji, Y., Wang, J., Gong, Y., Zhang, L., Zhu, Y., Wang, H., Zhang, J., Sakai, T.,
    Yang, Y.: Map: Multimodal uncertainty-aware vision-language pre-training model.
    In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition. pp. 23262–23271 (2023)
19. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung,
    Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation
    learning with noisy text supervision. In: Meila, M., Zhang, T. (eds.) Proceed-
    ings of the 38th International Conference on Machine Learning. Proceedings of
    Machine Learning Research, vol. 139, pp. 4904–4916. PMLR (18–24 Jul 2021),
    https://proceedings.mlr.press/v139/jia21b.html
20. Kuo, W., Cui, Y., Gu, X., Piergiovanni, A., Angelova, A.: F-vlm: Open-vocabulary
    object detection upon frozen vision and language models. In: International Con-
    ference on Learning Representations (2023)
21. Li, H., Zhang, R., Yao, H., Song, X., Hao, Y., Zhao, Y., Li, L., Chen, Y.: Learning
    domain-aware detection head with prompt tuning. Advances in Neural Information
    Processing Systems 36 (2024)
22. Lin, C., Sun, P., Jiang, Y., Luo, P., Qu, L., Haffari, G., Yuan, Z., Cai, J.: Learning
    object-language alignments for open-vocabulary object detection. In: International
    Conference on Learning Representations (2023)
23. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P.,
    Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision–
    ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12,
    2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P.,
    Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision–
    ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12,
    2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)
25. Liu, Y.C., Ma, C.Y., Dai, X., Tian, J., Vajda, P., He, Z., Kira, Z.: Open-set semi-
    supervised object detection. In: European Conference on Computer Vision. pp.
    143–159. Springer (2022)
26. Maalouf, A., Jadhav, N., Jatavallabhula, K.M., Chahine, M., Vogt, D.M., Wood,
    R.J., Torralba, A., Rus, D.: Follow anything: Open-set detection, tracking, and
    following in real-time. IEEE Robotics and Automation Letters 9 (4), 3283–
    (2024)
27. Miller, D., Nicholson, L., Dayoub, F., Sünderhauf, N.: Dropout sampling for robust
    object detection in open-set conditions. In: 2018 IEEE International Conference
    on Robotics and Automation (ICRA). pp. 3243–3249. IEEE (2018)
28. Miller, D., Sünderhauf, N., Milford, M., Dayoub, F.: Uncertainty for identifying
    open-set errors in visual object detection. IEEE Robotics and Automation Letters
    7 (1), 215–222 (2022).https://doi.org/10.1109/LRA.2021.
29. Oza, P., Patel, V.M.: C2ae: Class conditioned auto-encoder for open-set recog-
    nition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition. pp. 2307–2316 (2019)
30. Panareda Busto, P., Gall, J.: Open set domain adaptation. In: Proceedings of the
    IEEE international conference on computer vision. pp. 754–763 (2017)


```
Open-Set Recognition in the Age of Vision-Language Models 17
```
31. Perera, P., Morariu, V.I., Jain, R., Manjunatha, V., Wigington, C., Ordonez, V.,
    Patel, V.M.: Generative-discriminative feature representations for open-set recog-
    nition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition. pp. 11814–11823 (2020)
32. Pham, C., Vu, T., Nguyen, K.: Lp-ovod: Open-vocabulary object detection by lin-
    ear probing. In: Proceedings of the IEEE/CVF Winter Conference on Applications
    of Computer Vision. pp. 779–788 (2024)
33. Pham, T., Do, T.T., Carneiro, G., Reid, I., et al.: Bayesian semantic instance
    segmentation in open set world. In: Proceedings of the European Conference on
    Computer Vision (ECCV). pp. 3–18 (2018)
34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
    Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable
    visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.)
    Proceedings of the 38th International Conference on Machine Learning. Proceed-
    ings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul
    2021),https://proceedings.mlr.press/v139/radford21a.html
35. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z.,
    Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recog-
    nition challenge. International journal of computer vision 115 , 211–252 (2015)
36. Saito, K., Yamamoto, S., Ushiku, Y., Harada, T.: Open set domain adaptation by
    backpropagation. In: Proceedings of the European conference on computer vision
    (ECCV). pp. 153–168 (2018)
37. Saito, T., Rehmsmeier, M.: The precision-recall plot is more informative than the
    roc plot when evaluating binary classifiers on imbalanced datasets. PloS one 10 (3),
    e0118432 (2015)
38. Scheirer, W.J., de Rezende Rocha, A., Sapkota, A., Boult, T.E.: Toward open
    set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence
    35 (7), 1757–1772 (2013).https://doi.org/10.1109/TPAMI.2012.
39. Sünderhauf, N., Brock, O., Scheirer, W., Hadsell, R., Fox, D., Leitner, J., Upcroft,
    B., Abbeel, P., Burgard, W., Milford, M., et al.: The limits and potentials of deep
    learning for robotics. The International Journal of Robotics Research 37 (4-5), 405–
    420 (2018)
40. Upadhyay, U., Karthik, S., Mancini, M., Akata, Z.: Probvlm: Probabilistic adapter
    for frozen vison-language models. In: Proceedings of the IEEE/CVF International
    Conference on Computer Vision. pp. 1899–1910 (2023)
41. Vaze, S., Han, K., Vedaldi, A., Zisserman, A.: Open-set recognition: A good closed-
    set classifier is all you need? In: International Conference on Learning Representa-
    tions (ICLR) (2022)
42. Wang, J., Zhang, H., Hong, H., Jin, X., He, Y., Xue, H., Zhao, Z.: Open-vocabulary
    object detection with an open corpus. In: Proceedings of the IEEE/CVF Interna-
    tional Conference on Computer Vision. pp. 6759–6769 (2023)
43. Wu, J., Li, X., Xu, S., Yuan, H., Ding, H., Yang, Y., Li, X., Zhang, J., Tong, Y.,
    Jiang, X., Ghanem, B., Tao, D.: Towards Open Vocabulary Learning: A Survey.
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2024)
44. Wu, S., Zhang, W., Jin, S., Liu, W., Loy, C.C.: Aligning bag of regions for open-
    vocabulary object detection. In: Proceedings of the IEEE/CVF Conference on
    Computer Vision and Pattern Recognition. pp. 15254–15264 (2023)
45. Wu, X., Zhu, F., Zhao, R., Li, H.: Cora: Adapting clip for open-vocabulary de-
    tection with region prompting and anchor pre-matching. In: Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7031–
    7040 (2023)


18 D. Miller et al.

46. Yoshihashi, R., Shao, W., Kawakami, R., You, S., Iida, M., Naemura, T.:
    Classification-reconstruction learning for open-set recognition. In: Proceedings of
    the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp.
    4016–4025 (2019)
47. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca:
    Contrastive captioners are image-text foundation models. Transactions on Machine
    Learning Research (2022),https://openreview.net/forum?id=Ee277P3AYC
48. Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X.,
    Li, B., Li, C., et al.: Florence: A new foundation model for computer vision. arXiv
    preprint arXiv:2111.11432 (2021)
49. Zang, Y., Li, W., Zhou, K., Huang, C., Loy, C.C.: Open-vocabulary detr with
    conditional matching. In: European Conference on Computer Vision. pp. 106–122.
    Springer (2022)
50. Zara, G., Roy, S., Rota, P., Ricci, E.: Autolabel: Clip-based framework for open-
    set video domain adaptation. In: Proceedings of the IEEE/CVF Conference on
    Computer Vision and Pattern Recognition. pp. 11504–11513 (2023)
51. Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection
    using captions. In: Proceedings of the IEEE/CVF Conference on Computer Vision
    and Pattern Recognition. pp. 14393–14402 (2021)
52. Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., Beyer,
    L.: Lit: Zero-shot transfer with locked-image text tuning. In: Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18123–
    18133 (2022)
53. Zhai, X., et al.: Sigmoid loss for language image pre-training. In: Intl. Conf. on
    Computer Vision. pp. 11975–11986 (2023)
54. Zhang, H., Li, A., Guo, J., Guo, Y.: Hybrid models for open set recognition. In:
    Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August
    23–28, 2020, Proceedings, Part III 16. pp. 102–117. Springer (2020)
55. Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L.H., Zhou, L., Dai, X.,
    Yuan, L., Li, Y., et al.: Regionclip: Region-based language-image pretraining. In:
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
    nition. pp. 16793–16803 (2022)
56. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twenty-
    thousand classes using image-level supervision. In: European Conference on Com-
    puter Vision. pp. 350–368. Springer (2022)
57. Zhou, Z., Yang, Y., Wang, Y., Xiong, R.: Open-set object detection using
    classification-free object proposal and instance-level contrastive learning. IEEE
    Robotics and Automation Letters 8 (3), 1691–1698 (2023)
58. Zhu, B., Niu, Y., Han, Y., Wu, Y., Zhang, H.: Prompt-aligned gradient for prompt
    tuning. In: Proceedings of the IEEE/CVF International Conference on Computer
    Vision. pp. 15659–15669 (2023)
59. Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., HongFa, W., Pang, Y., Jiang, W.,
    Zhang, J., Li, Z., et al.: Languagebind: Extending video-language pretraining to
    n-modality by language-based semantic alignment. In: Intl. Conf. on Learning Rep-
    resentations (2023)


## Supplementary Material

### S1 Additional Evaluation Details

We have made our testing and evaluation repository publicly available atgithub.
com/dimitymiller/openset_vlms. We use the COCO API to identify True Pos-
itive (TP) predictions in object detection (see Sec. 4.1) – this is the established
process for identifying TPs used during mAP calculation [24].

### S2 Model Implementation Details

We list the testing configuration used for the VLM classifiers in the Table below.
This can also be seen in the evaluation repository included in the supplementary
folder, where we include the python scripts used to test and collect results from
each of the VLM classifiers.

Classifier Repository Extra Details

CLIP [34] OpenAI CLIP ViT-B/32 model

ALIGN [19] HuggingFace Transformers Kakao Brain implementation
trained on COYO dataset [61].

CoCa [47] OpenCLIP [62]
ViT-B/32 model
trained on LAION-2B [63].

ImageBind [14] Meta AI ImageBind imagebind_huge

SigLIP [53] HuggingFace Transformers google/siglip-base-patch16-

LanguageBind [59] LanguageBind Repository LanguageBind_Image

We list the testing configuration used for each VLM detectors in the Table
below. We use the identified config files or testing scripts without changes unless
specified otherwise. For all detectors, we do not impose any minimum confidence
threshold for predictions – low confidence predictions are naturally handled by
our AuPR metric which considers confidence thresholds.


20

```
Detector Repository Config file or testing script
OVR-CNN [51] Official Paper Repo zeroshot_v06.yaml
```
```
ViLD [15] Official Paper Repo ViLD_demo.ipynb
FLAGS.use_softmaxset toTrue
```
```
OV-DETR [49] Official Paper Repo Default evaluation setup
shown inrun_scripts.md
```
```
RegionCLIP [55] Official Paper Repo RN50, COCO Configuration
(Generalized: Novel + Base)
intest_transfer_learning.sh
```
```
Detic [56] Official Paper Repo Detic_OVCOCO_CLIP_
R50_1x_max-size_caption.yaml
```
```
VLDet [22] Official Paper Repo VLDet_OVCOCO_CLIP_
R50_1x_caption_custom.yaml
```
```
CORA [45] Official Paper Repo R50_dab_ovd_3enc_
apm128_splcls0.2_relabel_noinit.sh
```
### S3 ROC Curve Metrics

The area under the ROC curve (AuROC) is a popular metric in the traditional
open-set recognition literature [7,29,31,41,54]. However a key limitation of ROC
curves is that they are not affected by the size of the negative class [37] – this
can make them less informative and even misleading when used on datasets
where there is a large imbalance between the positive and negative class. We
highlight this in Tab. S1 for VLM object detection – VLM detectors withlarger
numbers of OSE achieve greater AuROC performance than detectors withlower
OSE counts. Due to these shortcomings of AuROC, we prefer AuPR for the
discussion of experiments in the main paper.


```
Supplementary Material 21
```
Table S1:Comparison between AuPR and AuROC metrics for the state-of-the-art
open-vocabulary object detectors (with ResNet backbone indicated), tested on COCO.
Arrows indicate the direction of optimal performance.

Negative AuPR AuROC TP OSE
Detector Uncertainty Embedding ↑↑↑↓

OVR-CNN (R50) [51]

```
Softmax Zero-Emb 75.4 93.0
Zero-Emb 77.2 94.1 13,544 190,146Cosine
Entropy Zero-Emb 63.2 81.2
```
ViLD (R152) [15]

```
Softmax “background” 60.1 97.0
16,011 1,485,600Cosine “background” 33.1 96.7
Entropy “background” 62.2 97.3
```
OV-DETR (R50) [49] Sigmoid None 75.8 97.4 15,818 1,485,600

RegionCLIP (R50) [55]

```
Softmax Zero-Emb 73.1 95.4
Zero-Emb 74.0 94.3 15,741 493,940Cosine
Entropy Zero-Emb 35.4 75.9
```
Detic (R50) [56] Sigmoid Zero-Emb 72.6 95.0 14,670 495,200

VLDet (R50) [22] Sigmoid Zero-Emb 79.8 92.8 14,751 112,923

CORA (R50) [45] Sigmoid Zero-Emb 67.2 95.0 13,763 495,200

### S4 Domain-specific Image Classification Results

We test three VLM classifiers – CLIP [34], ALIGN [19] and ImageBind [14] – on
an additional three popular domain-specific classification datasets:

1. German Traffic Sign Recognition Benchmark (GTSRB) [64]: containing im-
    ages of 34 different types of traffic signs. We use the same text prompts to
    describe each traffic sign as CLIP [34], see here. We test on the test split of
    the dataset, which contains 12 , 630 images.
2. Places365-Standard (Places365) [65]: containing images of 365 different scene
    categories. We use the category labels directly as text prompts (e.g. “amuse-
    ment park”, “aquarium”, etc.). We test on the validation split of the dataset,
    which contains 36 , 500 images.
3. Food101 [60]: containing images of 101 different food categories. We use the
    category labels directly as text prompts (e.g. “apple pie”, “bibimbap”, etc.).
    We test on the test split of the dataset, which contains 25 , 250 images.

In Fig. S1, we reproduce our experiments on the relationship between closed-
set and open-set performance from Sec. 5.3.


22

```
30 40 50 60 70 80 90
Accuracy (closed-set performance)
```
```
40
```
```
50
```
```
60
```
```
70
```
```
80
```
```
90
```
```
AuPR (open-set uncertainty performance)
```
```
CLIP - ImageNet
CLIP - Food101
CLIP - Places365
CLIP - GTSRB
ALIGN - ImageNet
ALIGN - Food101
ALIGN - Places365
ALIGN - GTSRB
ImageBind - ImageNet
ImageBind - Food101
ImageBind - Places365
ImageBind - GTSRB
```
Fig. S1:Results for the domain-specific datasets when investigating the relationship
between closed-set and open-set performance.

### S5 Uncertainty for Open-set and Closed-set Errors

We compare uncertainty performance for identifying errors in open-set recogni-
tion (i.e. open-set errors) versus standard closed-set recognition (i.e. closed-set
misclassifications). We report the softmax and sigmoid uncertainty baselines for
the VLM classifiers and object detectors.
Tab. S2 shows the results for VLM classification, where the performance of
uncertainty clearly degrades for open-set error compared to closed-set error for
all six classifiers. In particular, thresholding for high precision of predictions (i.e.
low error rates) shows substantial performance differences, indicating there may
be a greater portion of overconfident open-set errors than closed-set errors. In-
terestingly, SigLIP with the sigmoid activation and uncertainty appears to show
the least discrepancy between closed-set error and open-set error performance;
In fact, the AuROC performance for open-set errors is better than for closed-set
error. This suggests that sigmoid uncertainty may not be a powerful indicator
for closed-set error identification.
When performing the closed-set test with the VLM object detectors, false
positives can be categorised as either closed-set error or open-set error. False
positives that are caused by “background” detections – detections that do not
overlap a labelled object from the dataset class list – are actually open-set errors
under our proposed definition. False positives that are caused by misclassification

- detections that overlap a labelled object from the dataset class list but classify
incorrectly – are closed-set errors. For this experiment, we identify closed-set
errors as false positives from the closed-set test that overlap a labelled object
with an IoU greater than 0. 1 but misclassify the object class.
    Tab. S3 shows the results for VLM detection. Similar to the classification
results, the performance of uncertainty clearly degrades for open-set error com-
pared to closed-set error for all seven object detectors. For some detectors, the


```
Supplementary Material 23
```
Table S2:Difference in uncertainty (Softmax or Sigmoid) performance between open-
set error and closed-set error identification for the VLM classifiers tested on Ima-
geNet1k.

```
AuROC AuPR P@95R R@95P TP Error Count
Classifier Error ↑↑↑↑↑↓
CLIP [34] Open-set 79.7 71.5 46.2 3.4 31,023 50,000
(Softmax) Closed-set 80.7 87.4 68.5 30.0 31,023 18,977
ALIGN [19] Open-set 81.0 74.7 48.1 8.3 32,618 50,000
(Softmax) Closed-set 81.0 89.2 70.9 35.7 32,618 17,382
CoCa [47] Open-set 79.0 71.7 46.0 3.4 31,725 50,000
(Softmax) Closed-set 81.8 88.8 70.3 33.1 31,725 18,275
ImageBind [14] Open-set 82.8 79.1 52.8 5.0 38,405 50,000
(Softmax) Closed-set 83.4 94.1 82.3 57.1 38,405 11,595
SigLIP [53] Open-set 81.0 76.2 52.1 2.2 37,851 50,000
(Sigmoid) Closed-set 70.2 87.1 78.2 5.1 37,851 12,149
LanguageBind [59] Open-set 83.9 80.7 54.7 10.5 39,243 50,000
(Softmax) Closed-set 84.2 94.8 84.1 63.3 39,243 10,757
```
performance difference is substantial (e.g. the AuPR of CORA decrements by
10.2% between closed-set and open-set performance), whereas others are less no-
table (e.g. OV-DETR). For all detectors, one of the primary degradations from
closed-set to open-set is the number of errors, with between 1. 8 ×to 11. 7 ×more
open-set than closed-set misclassifications depending on the detector.

Table S3:Difference in uncertainty performance between open-set error and closed-set
error identification for the VLM detectors tested on COCO.

```
AuPR P@95R R@95P TP Error Count
Detector Error ↑↑↑↑↓
OVR-CNN (R50) [51] Open-set 75.4 13.0 42.9 13,544 190,146
(Softmax) Closed-set 81.8 25.7 50.9 13,544 68,708
ViLD (R152) [15] Open-set 60.1 7.5 15.4 16,011 1,485,600
(Softmax) Closed-set 75.6 28.0 29.7 16,011 126,774
OV-DETR (R50) [49] Open-set 75.8 5.9 47.3 15,818 1,485,600
(Sigmoid) Closed-set 78.1 9.0 48.2 15,818 461,950
RegionCLIP (R50) [55] Open-set 73.1 9.9 39.4 15,741 493,940
(Softmax) Closed-set 81.6 24.1 53.4 15,741 120,483
Detic (R50) [56] Open-set 72.6 8.1 42.2 14,670 495,200
(Sigmoid) Closed-set 80.4 20.1 52.4 14,670 154,824
VLDet (R50) [22] Open-set 79.8 21.9 48.5 14,751 112,923
(Sigmoid) Closed-set 82.8 29.2 52.0 14,751 64,247
CORA (R50) [45] Open-set 67.2 8.1 24.7 13,763 495,200
(Sigmoid) Closed-set 77.4 20.0 45.8 13,763 101,015
```

24

### S6 Qualitative Examples

Below we show a selection of images containing open-set errors from each of
the VLM object detectors. We threshold predictions from each detector at its
individual95%precision confidence threshold to show the most confident and
pervasive open-set errors.

```
OVR-CNN [51].
```
```
ViLD [15].
```
```
OV-DETR [49].
```

```
Supplementary Material 25
```
```
RegionCLIP [55].
```
```
Detic [56]
```
```
VLDet [22].
```
```
CORA [45].
```
Fig. S3:Qualitative examples of open-set errors from the VLM object detectors.


26

### S7 Uncertainty Histograms

In Fig. S4 and Fig. S5, we show histograms visualising prediction uncertainty for
the VLM classifiers and detectors respectively. We compare the uncertainty from
all correct closed-set predictions with the uncertainty from all open-set errors. In
particular, the histograms highlight the different uncertainty behaviours of VLM
classifiers and detectors – while the VLM classifiers are more prone to overcon-
fident open-set errors (i.e. open-set errors with very high softmax score), the
VLM detectors are more prone to underconfident correct closed-set predictions
(i.e. correct predictions with low softmax score).

```
0.000.0 0.2 0.4 0.6 0.8 1.0
```
```
0.05
```
```
0.10
```
```
0.15
```
```
0.20
```
```
0.25
```
```
Normalised Count
```
```
CLIP
Correct closed-set
Open-set error
```
```
0.000.0 0.2 0.4 0.6 0.8 1.0
```
```
0.05
```
```
0.10
```
```
0.15
```
```
0.20
```
```
0.25 Correct closed-setALIGN
Open-set error
```
```
0.00.0 0.2 0.4 0.6 0.8 1.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
CoCa
Correct closed-set
Open-set error
```
```
0.0 0.2 0.4 0.6 0.8 1.0
Confidence score
```
```
0.0
```
```
0.2
```
```
0.4
```
```
0.6
```
```
Normalised Count
```
```
ImageBind
Correct closed-set
Open-set error
```
```
0.0 0.2 0.4 0.6 0.8 1.0
Confidence score
```
```
0.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
0.5
```
```
0.6
```
```
SigLIP
Correct closed-set
Open-set error
```
```
0.0 0.2 0.4 0.6 0.8 1.0
Confidence score
```
```
0.0
```
```
0.2
```
```
0.4
```
```
0.6
```
```
LanguageBind
Correct closed-set
Open-set error
```
Fig. S4:The VLM classifiers suffer from overconfident open-set errors, i.e. open-set
errors with very high softmax scores.

```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
0.5
```
```
0.6
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
OVR-CNN
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
0.5
```
```
0.6
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
ViLD
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.2
```
```
0.4
```
```
0.6
```
```
0.8
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
OV-DETR
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
0.5
```
```
0.6
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
RegionCLIP
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.1
0.2
0.3
0.4
```
```
0.5
0.6
0.7
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
Detic
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.1
```
```
0.2
```
```
0.3
```
```
0.4
```
```
0.5
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
VLDet
```
```
0.00.0 0.2Softmax score0.4 0.6 0.8 1.0
```
```
0.2
```
```
0.4
```
```
0.6
```
```
0.8
```
```
Normalised Count
```
```
Correct closed-setOpen-set error
```
```
CORA
```
Fig. S5:The VLM object detectors are prone to underconfident correct closed-set
predictions, i.e. correct predictions with low softmax score. The histograms show nor-
malised counts, as there is large imbalance in the magnitude of the correct closed-set
and open-set error sets.


```
Supplementary Material 27
```
### S8 Negative Embeddings and Uncertainty Interactions

For some VLM classifiers, the negative queries only capture and remove “easy”
open-set errors, i.e. open-set errors that already have high uncertainty. See
Fig. S6, which shows the predictions that are removed when introducing 500
random words as negative queries. In contrast to CLIP and ALIGN, ImageBind
is able to remove previously overconfident open-set errors with these negative
queries, resulting in up to 5 .5%increase in AuPR.

```
0.0 0.2 0.4 0.6 0.8 1.0
Softmax score
```
```
0
```
```
2000
```
```
4000
```
```
6000
```
```
8000
```
```
10000
```
```
Count
```
```
CLIP
```
```
0.0 0.2 0.4 0.6 0.8 1.0
Softmax score
```
```
0
```
```
2000
```
```
4000
```
```
6000
```
```
8000
```
```
10000
```
```
ALIGN
```
```
True positive
True positive removed by negative query
```
```
Open-set error
Open-set error removed by negative query
```
```
0.0 0.2 0.4 0.6 0.8 1.0
Softmax score
```
```
0
```
```
5000
```
```
10000
```
```
15000
```
```
20000
```
```
25000
```
```
30000 ImageBind
```
Fig. S6:Using negative queries (random 500 words) removes open-set error with dif-
ferent uncertainties depending on the VLM classifier.

### S9 Investigating Individual Random Embedding Efficacy

Some negative embeddings are more effective for capturing open-set errors. Be-
low, we plot the closed-set versus open-set trade-off for 2500 random embeddings
when used with the VLM classifiers. Ideally, an embedding should capture many
open-set images without capturing any closed-set images. Green shading shows
embeddings that capture more open-set than closed-set images, and red shading
shows the opposite.

(^0020) Closed-set images captured 40 60 80 100120140
50
100
150
200
250
300
Open-set images captured
CLIP Random Embeddings
(^0020) Closed-set images captured 40 60 80 100 120 140
50
100
150
200
250
Open-set images captured
CLIP Random Words


28

(^00) Closed-set images captured 200 400 600 800
500
1000
1500
2000
Open-set images captured
ALIGN Random Embeddings
(^0010) Closed-set images captured 20 30 40 50 60 70
20
40
60
80
100
120
140
160
Open-set images captured
ALIGN Random Words
0 1 2 3 4
Closed-set images captured
0
2
4
6
8
10
12
Open-set images captured
CoCa Random Embeddings
0 25 50 75 100 125 150
Closed-set images captured
0
50
100
150
200
250
300
350
400
Open-set images captured
CoCa Random Words
(^00100) Closed-set images captured 200 300 400
200
400
600
800
1000
1200
1400
Open-set images captured
ImageBind Random Embeddings
(^0020) Closed-set images captured 40 60 80 100
50
100
150
200
250
Open-set images captured
ImageBind Random Words
(^00) Closed-set images captured 1000 2000 3000 4000
1000
2000
3000
4000
5000
Open-set images captured
SigLIP Random Embeddings
(^0010) Closed-set images captured 20 30 40 50 60 70
50
100
150
200
250
Open-set images captured
SigLIP Random Words
(^002) Closed-set images captured 4 6 8 10 12
10
20
30
40
50
60
70
Open-set images captured
LanguageBind Random Embeddings
(^0020) Closed-set images captured 40 60 80 100120140
50
100
150
200
250
300
350
Open-set images captured
LanguageBind Random Words
Fig. S8:Trade-off of individual negative embeddings for capturing closed-set images
incorrectly versus capturing open-set images correctly.


```
Supplementary Material 29
```
### Supplementary References

60. Bossard, L., Guillaumin, M., Van Gool, L.: Food-101 – mining discriminative com-
    ponents with random forests. In: European Conference on Computer Vision (2014)
61. Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., Kim, S.: Coyo-700m: Image-text
    pair dataset.https://github.com/kakaobrain/coyo-dataset(2022)
62. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave,
    A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.:
    Openclip (Jul 2021).https://doi.org/10.5281/zenodo.5143773,https://doi.
    org/10.5281/zenodo.5143773
63. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C.W., Wightman, R., Cherti,
    M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kun-
    durthy, S.R., Crowson, K., Schmidt, L., Kaczmarczyk, R., Jitsev, J.: LAION-5b:
    An open large-scale dataset for training next generation image-text models. In:
    Thirty-sixth Conference on Neural Information Processing Systems Datasets and
    Benchmarks Track (2022),https://openreview.net/forum?id=M3Y74vmsMcY
64. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C.: Man vs. computer: Bench-
    marking machine learning algorithms for traffic sign recognition. Neural Net-
    works (0), – (2012).https://doi.org/10.1016/j.neunet.2012.02.016,http:
    //www.sciencedirect.com/science/article/pii/S0893608012000457
65. Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., Torralba, A.: Places: A 10 million
    image database for scene recognition. IEEE transactions on pattern analysis and
    machine intelligence 40 (6), 1452–1464 (2017)


