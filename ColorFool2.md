# Recap:

Tecniche di difesa:

1) Adversarial Traning, costoso
2) Gradient Masking
   * Ridurre la senzibilita' a piccoli cambiamenti nell'input 
   * Aggiunta di un Gradient penality term, calcolato come sommatoria layer per layer della norma di Frobenius calcolata sulla matrice Jacobiana ![Frobenius_norma](https://valentinagaudio1.altervista.org/wp-content/uploads/2019/11/image-2-960x145.png)
3) Input trasformation, such as Principal Component Analysis (PCA)...

Riconoscimento di Adversarial Examples:

1) Sample statistics
2) Traning a Detector:
   * ~  Adversarial Traning, costoso
   * Addestrare un CNN/DNN a riconoscere sempi legittimi
3) Prediction Incosistency:
   * Si basa sull'incoerenza tra le previsioni.
   * Se si allenano piu' modelli, si nota che il distacco tra questi modelli su campioni legittimi e' basso.

# Feature Squeezing:

L'idea dello Feature Squeezing e' quella di utilizzare lo stesso modello per classificare piu' campioni e confrontare queste classificazione al fine di determinare se tale campione e' o non e' legittimo.
Dato un campione X, vi viene application un filtro che "riduce le caratteristiche",  la nuova immagine Y poi classificata dal modello, tale classificazione viene confrontata con quella dell'immagine iniziale.
Se la differenza e' superiore a una certa soglia allora il campione e' un campione "Aversario" altrimenti e' legittimo.

![Feature Squeezing](./imgs/Feature_Squeezing_Diagram.png)

Ovviamente piu' filtri "Squeezing" possono essere usati al fine di migliorare la classificazione dei campioni avversari.

Il Feature Squeezing si basa sulle riduzione delle caratteristich, tale operazione viene effettuata in due passi:

1) Riduzione della profondita' del colore (desaturazione)
    * Loro ipotizzano infatti che riducendo il color depth, si riduca la possibilita' di un attacco, senza inficiare la qualita' della classificazione.
    Su CIFAR-10 e ImageNet hanno ridotto il color depth da 8 a 4 (per ogni canale).

2) Uso di filtri di smoothing locale e globale 
   * L'idea e' di ridurre la variazione tra i pixel e il rumore,
   * Come smoothing locale e' stato usato il Gaussian blur (kernel 2x2)
   * Come smoothing globale e' stato usato il "denoising" implementato in OpenCV, che converte l'immagine da RGB a CIELAB space, applicata un senoiser alla componente L e AB, e poi la riconverte in RGB 

Questo metodo ha il beneficio di essere abbastanza accurato e poco costoso.



***********************************************************************************************************************************************************************************************************************************************************************************

# ColorFool: Semantic Adversarial Colorization

Authors: Ali Shahin Shamsabadi, Ricardo Sanchez-Matilla, Andrea Cavallaro

Link to paper: https://arxiv.org/pdf/1911.10891.pdf


# General info about the paper:

The authors propose ColorFool : a content-based black-box adversarial attack that generates unrestricted (low-frequency) perturbations by exploiting image semantics to selectively modify colors within chosen ranges that are perceived as natural by humans. ColorFool operates only on the de-correlated a and b channels of the Lab color space without changing the lightness L. Also, introduces perturbations only within a chosen natural-color range for specific semantic categories. ColorFool is appliable on images of any size.
The evaluation and validation is based on: 

- success rate, robustness to defense frameworks and transferability; 

- compared with 5 state-of-the-art adversarial attacks(Basic Iterative Method-BIM; Translation-Invariant BIM;DeepFool;SparseFool;SemanticAdv; BigAdv was excluded since no code was available) on 2 different tasks(scene and object classification);

- 3 state-of-the-art DNN classifiers under attack (ResNet18, ResNet50, AlexNet) with 3 standard datasets (Private-Place365-scene classification db; CIFAR-10-object classification db; ImageNet-object classification db)

Note: the perturbations can be restricted (by controlling the <img src="/tex/17ed5737af9a2158e95dcfaa3e149851.svg?invert_in_darkmode&sanitize=true" align=middle width=79.01779214999999pt height=22.465723500000017pt/> , such as <img src="/tex/2c2d5be52a564c33e9bd18c490d5c668.svg?invert_in_darkmode&sanitize=true" align=middle width=31.664032949999992pt height=22.465723500000017pt/>, <img src="/tex/cc96eb8a40f81e8514147d06c9e8ad92.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/>,<img src="/tex/929ed909014029a206f344a28aa47d15.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/>,<img src="/tex/4327ea69d9c5edcc8ddaf24f1d5b47e4.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=22.465723500000017pt/>) or unrestricted which span a wider range as determined by different colorization approaches.

Defenses used:

- requantization (feature squeezing paper), median filter (feature squeezing paper) and JPEG compression: to remove adversarial perturbation prior to classification
-  improve robustness of the classifier through adversarial training or by changing the loss functions

An attack should be: robust (able to mislead a classifier in presence of defense frameworks), tranferable (able to mislead an unseen classifier), unnoticeable/undetectable (the shape and spatial arrangement of objects in the adversarial image should be perceived as in the clean image and the colors should look natural).

## How does ColorFool work

Uses image segmentation to identify image regions. These image regions are classified in 2 categories: 	
-sensitive regions: the colors/appereance of these regions are typically within a specific range and unusual colors would attract the attention of a human observer. Here we have 4 categories of sensitive regions: person, sky, vegetation(grass, trees, etc), water(rivers, sea, lake, swimming pool, etc).
-non-sensitive regions: may have their colors modified withing an arbitrary range and still look natural.<img src="C:\Users\Ionuy\Desktop\object_segmentation.PNG" style="zoom:75%;" />

![ColorFool- semantic segmentation](./imgs/semantic_segmentation.PNG)

Firstly, an image <img src="/tex/783e9a00f6408d8417af7858904f235a.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=22.55708729999998pt/> is decomposed in <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> semantic regions:
<p align="center"><img src="/tex/3d89fd7aa2b55ad24059a1e9d7eec72d.svg?invert_in_darkmode&sanitize=true" align=middle width=201.7021743pt height=21.0913428pt/></p>

where <img src="/tex/ab583fc1469101108937225bb0f4d059.svg?invert_in_darkmode&sanitize=true" align=middle width=108.31648904999999pt height=27.91243950000002pt/> is a binary mask that specifies the location of pixels to regions <img src="/tex/67f338190db57bac70d43e66e745cbfb.svg?invert_in_darkmode&sanitize=true" align=middle width=17.345954999999993pt height=22.465723500000017pt/> and "<img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/> " denotes a pixel-wise multiplication.

Binary masks are outputted by a pyramid Pooling R50-Dilated architecture of Cascade Segmentation Module segmentation trained on MIT ADE20k dataset on 150 semantic regions types.



Then, the regions are divided into sensitive (<img src="/tex/7b5263c6bc295e4ecde58fead6baf509.svg?invert_in_darkmode&sanitize=true" align=middle width=89.56631639999999pt height=27.6567522pt/>) and non-sensitive regions (<img src="/tex/3cb794d243681bb946c1ae141a39c1d6.svg?invert_in_darkmode&sanitize=true" align=middle width=90.51375465pt height=32.127734099999984pt/>) such that <img src="/tex/e46cfbadd60fa813138e7ff1d18e5ea7.svg?invert_in_darkmode&sanitize=true" align=middle width=69.63438569999998pt height=27.90838379999999pt/>,where <img src="/tex/655ba4ff82d4b3329109a26a2295efe1.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=18.264896099999987pt/> is the union operator.

After identifying these two sets of regions , the colors of the regions are modified in the Lab color space, which separates color information from brightness: 

 - a ranges from green(-128) to red (+127)
 - b ranges from blue(-128) to yellow (+127)
 - L ranges from black(0) to white (100)

The sensitive regions are modified and the adversarial set is generated according to the following formula :
<p align="center"><img src="/tex/08ff5a457ad899027876e2047dfe5f18.svg?invert_in_darkmode&sanitize=true" align=middle width=313.86046395pt height=19.3346868pt/></p>

where <img src="/tex/8d9685dcc15b9ba5ca3a402c117a8008.svg?invert_in_darkmode&sanitize=true" align=middle width=26.775518549999994pt height=24.65753399999998pt/> converts the intensities of an image from RGB to the Lab color space, <img src="/tex/9c93ef2cdb88b4d39489e795d126594d.svg?invert_in_darkmode&sanitize=true" align=middle width=78.04495214999999pt height=24.246581700000014pt/> and <img src="/tex/48f6909b19fe7618f72f20205783754f.svg?invert_in_darkmode&sanitize=true" align=middle width=75.34584419999999pt height=30.3196509pt/> are the adversarial perturbations in the channels <img src="/tex/e9bd88c7f374ee1f4efeb29273104c65.svg?invert_in_darkmode&sanitize=true" align=middle width=9.66282734999999pt height=14.15524440000002pt/> and <img src="/tex/d01c261fd795e601575529caaf627248.svg?invert_in_darkmode&sanitize=true" align=middle width=8.59924724999999pt height=22.831056599999986pt/> that are chosen randomly from the set of natural-color ranges, <img src="/tex/b66984539066f0300cf284005588be43.svg?invert_in_darkmode&sanitize=true" align=middle width=28.70582549999999pt height=24.246581700000014pt/> and   <img src="/tex/af362a8e50c61919fcca6141ae74153c.svg?invert_in_darkmode&sanitize=true" align=middle width=27.356280599999987pt height=30.3196509pt/> in the <img src="/tex/e9bd88c7f374ee1f4efeb29273104c65.svg?invert_in_darkmode&sanitize=true" align=middle width=9.66282734999999pt height=14.15524440000002pt/> and <img src="/tex/d01c261fd795e601575529caaf627248.svg?invert_in_darkmode&sanitize=true" align=middle width=8.59924724999999pt height=22.831056599999986pt/> channels. 
These ranges are defined based on the actual colors, region semantics and prior knowledge about color perception in that region type as shown in the following table:
![ColorFool- semantic segmentation](./imgs/semantic_color_range.PNG)

Note that no color changes are applied to image regions classified as person.

Multiple trials are allowed until a perturbation misleads the classifier. To avoid large color changes in the first trials the perturbations is scaled down by a factor alpha = n/N, where n is the index of the trial and N is the max number of trials.

The non-sensitive regions are also modified as follows:
<p align="center"><img src="/tex/8ccae28aed0b1324dd9b6d2da8b61327.svg?invert_in_darkmode&sanitize=true" align=middle width=304.1113515pt height=22.62218145pt/></p>

where <img src="/tex/811b96af754d179103a80000b6503d14.svg?invert_in_darkmode&sanitize=true" align=middle width=149.89293pt height=30.821574299999984pt/> and <img src="/tex/55cfe68c206776dde5e029fff1accf92.svg?invert_in_darkmode&sanitize=true" align=middle width=148.54336859999998pt height=36.89464019999998pt/> are  chosen randomly inside the whole range of <img src="/tex/e9bd88c7f374ee1f4efeb29273104c65.svg?invert_in_darkmode&sanitize=true" align=middle width=9.66282734999999pt height=14.15524440000002pt/> and <img src="/tex/d01c261fd795e601575529caaf627248.svg?invert_in_darkmode&sanitize=true" align=middle width=8.59924724999999pt height=22.831056599999986pt/>, since the regions can undergo larger intensity changes.

Finally, the adversarial image generated by ColorFool combines the modified sensitive and non-sensitive regions as:



<p align="center"><img src="/tex/e55f7ee46207d57297518517945fb5d3.svg?invert_in_darkmode&sanitize=true" align=middle width=249.9647568pt height=59.1786591pt/></p>


where <img src="/tex/a949f14faf88fa4f6b407ebd900f4e78.svg?invert_in_darkmode&sanitize=true" align=middle width=30.347082149999988pt height=24.65753399999998pt/> is the quantization function which ensures that the generated adversarial image is in the dynamic range of pixel values and <img src="/tex/395000f9e960d9c30bd167c86f2ed1cc.svg?invert_in_darkmode&sanitize=true" align=middle width=26.250429149999988pt height=26.76175259999998pt/> is the inverse function that converts the intensities of an image from Lab to the RGB color space.

## Validation

Algorithms under comparison: BIM, TI-BIM, DeepFool, SparseFool, SemanticAdv (BigAdv excluded because no code was available)

Datasets: Private-Places365, CIFAR-10, ImageNet

Classifiers under attack: ResNet18, ResNet50, AlexNet

Performance measures: 

 - success rate in misleading a classifier
 - robustness to defenses (requantization, median filter, lossy jpeg compression and report SR and undetectability; evaluate SR when attacking a seen classifier trained with Prototype Conformity Loss and adversarial training)
 - image quality of the adversarial images (using NIMA: neural image assessment, trained on AVA dataset; NIMA estimates the perceived image quality and was shown to predict human preferences)

## Results

SR:

 - all adversarial attacks achieve higher SR in a seen classifier

 - unrestricted attacks achieve higher SR than restricted attacks (in unseen classifiers)
 - ColorFool achieves high SR on both seen and unseen classifiers
 - unrestrcited attacks obtain high transferability rates

Robustness:

 - the most robust attacks are the unrestricted ones where Colorfool, Colorfool-r (no priors are considered for the sematic regions) and SemanticAdv consistently obtain a SR above 60% across datasets and classifiers
 - restricted attacks are more detectable than unrestricted ones when considering all image filters across all classifiers and datasets.
 - restricted attacks generate high-frequency adversarial perturbations whereas unrestricted attacks generate low-frequency perturbations (these ones are more robust to requantization, median filter and jpeg compression). NOTE: Jpeg compression is the most effective detection framework.
 - the robustness of the adversarial images is proportional to the accuracy of the classifier use for their generation.
 - ColorFool is robust when misleading ResNet110 equipped with both PCL and adversarial training defenses (adversarial images generated by BIM)

Quality:

 - restricted attacks such as TI-BIM or SparseFool generate adversarial images with minimal perturbations but they are noticeable
 - SemanticAdv and ColorFool-r generate unrealistic colors
 - ColorFool generate image that look natural even though they are largely different from the clean images.
 - unrestricted attacks obtain the highest NIMA scores across all attacks, classifiers and datasets
 - ColorFool obtains equal or higher NIMA scores than the clean images considering all datasets and classifiers

