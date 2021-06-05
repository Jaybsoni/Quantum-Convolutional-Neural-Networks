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

#### Layers and Structure
The first layer we look at is the convolutional layer. In the classical CNN, the kernel's size determines how many neighboring pixels will be involved in the convolution. In this case, we use a unitary operator which acts on a subset of our qubits (we call this the convolutional operator). In a CNN the same kernel is swept across the entire input image; to mirror this, we apply the same convolutional operator to each set of qubits in the layer. This is described in figure 1. 

<img src="/images/Conv_layer_diagram.PNG">

**Figure 1**: A convolutional layer with a 3 qubit convolutional operator. Note, the same operator (and parameter vector theta) is used on each set of qubits within the layer. Adding a second convolutional layer would introduce an independent convolutional operator. 
*Technical Remark: If the total number of qubits in the circuit is not divisible by the number of qubits per convolutional operator then the bottom few qubits in the circuit are left unchanged*. 

The next layer is the pooling layer. Classically, convolutional layers also did some inherent amount of pooling as the size of the kernel will affect the size of the convolved image. Since quantum computation involves unitary operators, we can't "remove qubits"; we can however measure certain qubits and then choose to not operate on the for the remainder of the circuit. Thus we define the quantum pooling layer by again grouping qubits into sets. We then measure a subset of the qubits in a given set and use the result of the measurement to perform controlled unitary operations on the remaining qubits. If multiple operators are used, then they are repeated exactly for each set of qubits in the layer. This is depicted in figure 2. 

<img src="/images/pooling_layer_diagram.PNG">

**Figure 2**: A Pooling layer which groups qubits in sets of 3s, measures the first and third qubits and uses those results to apply the controlled unitray operations to the second. 

With an understanding of the types of layers we can now present the architecture that was used in the paper[1]. The QCNN has 4 convolutional layers followed by a single pooling layer followed by a fully-connected layer leading to the final measurement. The first convolutional layer applies a special type of 4 qubit convolution (details found in qml_src/qml/layers.py). The next 3 convolutional layers are the standard 3 qubit convolutions as described above. They are shifted by 1 qubit from each other to allow for entanglement between all of the qubits as opposed to just the adjacent set of 3 qubits. The pooling layer reduces the total number of "active" qubits in the circuit by a factor of 3. The pooling unitaries are controlled by a measurement in the X-basis. The final fully connected layer can be thought of as a single convolution operator acting on all of the remaining qubits in the circuit. 
  
<img src="/images/QCNN_arc.PNG">

**Figure 3**: The QCNN architecture used in the original paper[1]. 

### Distinguish Quantum Phases of Matter 
The authors present us with the following hamiltonian[1]: 

<img src="https://render.githubusercontent.com/render/math?math=\hat{H}=-J\sum_{i=1}^{n-2}(\hat{Z}_{i}\hat{X}_{i%2B1}\hat{Z}_{i%2B2})-h_{1}\sum_{i=1}^{n}(\hat{X}_{i})-h_{2}\sum_{i=1}^{n-1}(\hat{X}_{i}\hat{X}_{i%2B1})">

It turns out that the ground state wave functions of this hamiltonian can exist in multiple states of matter depending on the coupling constants (J, h1, h2). Specifically, for certain values of the coupling constants the states can exist in a symmetry protected topological phase (SPT phase) with Z2 x Z2 symmetry. Other states of matter include the paramagnetic state and the anti-ferromagnetic state. The question is then, can a QCNN determine which state of matter a ground state wave function belongs to? 

To answer this, the authors first generated a training set of 40 ground state wavefunctions corresponding to coupling constant values of J=1, h2=0 and h1 sampled 40 times between 0 and 1.6. The state of matter is analytically solvable for this choice of parameters and the solution is used to label the training dataset (0 corresponds to paramagnetic or anti-ferromagnetic phase while 1 corresponds to SPT phase). They then found the ground truth phase boundaries using DMRG simulations. These phase boundaries are a set of values for (h1/J,h2/J) which separate the two phases. Finally, they generated ground state wave functions corresponding to each combination of (h1/J, h2/J) in the range (0, 1.6) and (-1.6, 1.6) respectively. These wavefunctions would be fed into the QCNN and the final measurement would determine the predicted label/phase of matter that the wavefunction is in. Below is a diagram of the results the authors have presented in the paper. 

<img src="/images/paper_results.PNG">

**Figure 4**: A diagram of the predictions of the QCNN. They grey dotted line represents the training dataset. The red and blue lines are the true phase boundaries between the SPT phase and the other phases. The green / yellow heat map represents the output of the QCNN for those coupling constants (h1/J, h2/J). 

These are the results we aimed to reproduce in our project.

## Results 
In this section we present the results of our project. In the first figure below we present the phase prediction diagram generated by our implementation of the QCNN. We were successfully able to reproduce the architecture used in the paper and generated an almost identical phase prediction diagram, thus showing that the QCNN is capable of differentiating between the SPT phase and the paramagnetic and anti-ferromagnetic phase. In the original paper, the authors used a 12 qubit QCNN, but due to computational restrictions we were only able to simulate the QCNN with 9 qubits. Nevertheless, our results match theirs almost perfectly and we believe that this trend would carry forward as we increase the number of qubits.

<img src="/images/results_and_loss.PNG">

**Figure 5**: Phase prediction diagram and loss function over iterations. Note that we converge fairly quickly (~300 iterations)

We also used a different parameterization for our unitary operators. This uniformly controlled rotation based parameterization uses significantly fewer parameters to generate the convolutional operators. In the case of a 3 qubit convolution operator we observe an 8x reduction in total parameters, resulting in much faster training. Below is the phase prediction diagram using the new convolutional operator. As you can see, we get almost identical performance using the new parameterization! Thus we gain a speed up without compromising performance.

<img src="/images/results_and_loss_new_conv.PNG">

**Figure 6**: Phase prediction diagram and loss function over iterations for the new model using uniformly controlled rotations to parameterize convolutional operators.


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

**Anthony Allega** *(Software Dev)*: A Physics & Astronomy major with an interest in experimental particle physics. My goal is to contribute a large body of work geared to how we measure neutrinos in detectors around the world. The acronyms in the field of particle physics are pretty bad, so I'll probably come up with some good ones as well. I intend to prioritize the acronyms over the other stuff.
