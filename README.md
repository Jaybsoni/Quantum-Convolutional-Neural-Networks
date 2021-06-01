# Quantum Convolutional Neural Networks for Phase Recognition
## Project Description 


## Background 


## Results 


## Code and Installation
### code sections 
The code for the project is split into 3 sections (data_gen, qml_src, and qml_for_phase_recog): 

_**data_gen**_: This folder contains code which generates our ground state wavefunctions for vary values of the coupling constants h1 and h2. It also makes the distinction between our training dataset and the testing dataset. The resultant text files that are produced here are moved into the qml_for_phase_recog folder to be processed. 

_**qml_src**_: This folder contains our custom module which implements the abstract QCNN class and all of the varying layers used in the class! 

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
