# Quantum Convolutional Neural Networks for Phase Recognition
## Project Description 
Quantum Convolutional Neural Networks (QCNNs) are a special class of variational quantum circuits which take inspiration from the convolution and pooling operations of classical CNNs. In this project we explore QCNNs for recognition of quantum phases of matter. This project follows the original paper[1] presented by Iris Cong and company ([original paper](https://arxiv.org/pdf/1810.03787.pdf)). The project deliverables are as follows: 

**Quantum Machine Learning Module**: We constructed a custom QML module which allows users to create their own QCNNs with access to the original convolution and pooling layers mentioned in the paper (legacy layers) as well as a variety of customizable convolution and pooling layers. The core class and functions mirror those of popular ML frameworks (TF or pytorch) which make using the module easy and intuitive. 

**Reproducing Results for Phase Recognition**: We were able to recreate the exact architecture used in the original paper[1] and show similar results for the phase prediction diagram; thus we were able to independantly validate the results of the study. 

**Improving Unitary Parameterization**: We realized that the Gel Mann matrices based parameterization of the unitaries in the model used significantly more trainable parameters than the theoretical lower bound (total gm parameters = <img src="https://render.githubusercontent.com/render/math?math=2^{2n}-1"> v.s theoretical lower bound = <img src="https://render.githubusercontent.com/render/math?math=2^{n %2B 1}-2">). We found a paper[2] which presents an ansatz for quantum state preparation using uniformally controlled rotations. We used this as the basis of an alternate parameterization scheme with <img src="https://render.githubusercontent.com/render/math?math=2^{n %2B 2}-5"> total parameters. We then showed that this new scheme was able to achieve the same performance standards as the original model, now with fewer total parameters (thus faster training). In our particular case we realized a 8x reduction in total trainable parameters per 3 qubit convolutional layer

## Background 
### QCNNs 
#### Unitary Parameterization
CNNs are typically used when working with image data, although any form of data which can be represented in an array of values can be used. CNNs perform convolutions and poolings by sliding a smaller "image" (called a kernel) along the input image and computing the sum of pixel by pixel products. The model learns by tuning the values in the kernel to extract the features of interest from the input data. 

In order to translate this concept to the quantum equivalent we have to consider parameterized quantum circuits. In this case, our *tunable knobs* are the parameters we use to generate unitary operators which will act on the qubits in our circuit. This paper[1] uses the following parameterization: <img src="https://render.githubusercontent.com/render/math?math=\hat{U}(\vec{\theta})=e^{\sum_{i}^{N}(\theta_{i}\Lambda_{i})}">

Where <img src="https://render.githubusercontent.com/render/math?math=\vec{\theta}"> is the set of trainable parameters and <img src="https://render.githubusercontent.com/render/math?math=\Lambda_{i}"> is the i'th Gell Mann matrix. In this case we are summing over all of the Gell Mann matricies of a particular order. 

If we want an operator <img src="https://render.githubusercontent.com/render/math?math=\hat{U}"> which acts on k qubits, then we use the set of <img src="https://render.githubusercontent.com/render/math?math=2^{k}">'th order Gell Mann matricies. In general there are <img src="https://render.githubusercontent.com/render/math?math=(2^{k})^{2}-1"> such matricies. Since each matrix has an associated tuneable scale parameter, we have <img src="https://render.githubusercontent.com/render/math?math=(2^{k})^{2}-1"> trainable parameters per parameterized unitary operator. An alternate parameterization was found by adapting uniformly controlled rotations (ref [2]). This parameterization requires only <img src="https://render.githubusercontent.com/render/math?math=2^{k %2B 2}-5"> parameters per unitary operator. This new parameterization was also explored in our project. 

#### Convolution Layer


#### Pooling Layer


### Distinguish Quantum Phases of Matter 


## Results 


## Code and Installation
### code sections 
The code for the project is split into 3 sections (data_gen, qml_src, and qml_for_phase_recog): 

_**data_gen**_: This folder contains code which generates our ground state wavefunctions for vary values of the coupling constants h1 and h2. It also makes the distinction between our training dataset and the testing dataset. The resultant text files that are produced here are moved into the qml_for_phase_recog folder to be processed. 

_**qml_src**_: This folder contains our custom module which implements the abstract QCNN class and all of the varying layers used in the class! For additional information on using the module refer to the readme in this folder.

_**qml_for_phase_recog**_: This folder imports the qml module and uses the data produced in data_gen to build the model and use the QCNN for the prescribed application.

### qml install
Before running any of the scripts, users must first install the qml module itself. This can be done as follows: 
* Clone this repo and change directory in the terminal to the `/qml_src` directory.
* Run the following command: `pip install -e ./` 
* Check that it is installed by running `pip list` and looking for the module named `qml`

## References
* [1]: I. Cong, S. Choi, and M. D. Lukin, "Quantum Convolutional Neural Networks", arXiv.org, May 2019 
* [2]: M. Mottonen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa, arXiv.org, July 2004
* [3]: M. Plesch, and C. Brukner, Physical Review A83, 2011 

## Acknowledgements
We would like to give a huge thank you to _**Iris Cong**_ from Harvard for all of her insightful discussions and her guidance throughout the project! We would like to thank _**Dr.Ronagh Pooya**_ for giving us the opportunity to pursue this ambitious (and slightly un-orthodox) direction for as our final course project. We would also like to thank him for giving us access to cloud compute resources which allowed us to easily train and validate all of the models we developed. 

## Meet the Team
**Jay Soni** *(Project Lead + QML Lead)*: A Mathematical Physics major with a passion for quantum computing. I enjoy coding almost as much as I enjoy drinking Ice Capps! 

**Ivan Sharankov** *(Lead Software Dev)*: Astrophysics major who's looking to continue his masters in computational physics. I love programming and will jump on any hackathon or project I can find. Interested in handling big data, machine learning applications in physics, cryptography/system security, and network engineering.

**Anthony Allega** *(Computational Software Dev)*: A Physics & Astronomy major with an interest in experimental particle physics. My goal is to contribute a large body of work geared to how we measure neutrinos in detectors around the world. The acronyms in the field of particle physics are pretty bad, so I'll probably come up with some good ones as well. I intend to prioritize the acronyms over the other stuff.
