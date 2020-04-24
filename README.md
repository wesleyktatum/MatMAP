# m2py: Materials Morphology Python Package

An open-sourced library for automated phase and domain labeling in scanning probe microscopy (SPM) data


### Usage

Pre and post-processing functions are found in the *utils* folder, along with a modules for label handling
and other utility functions. Instance and Semantic segmenter modules are found in the *segmentation* directory.

Example scripts for generating and analyzing m2py labels for organic photovoltaic (OPV) and organic field-effect
transistor (OFET) device active-layers are found in the *scripts* directory. Finally, neural networks
implemented using PyTorch are located in the *networks* folders. The shown networks are tuned to the OPV
and OFET datasets, but include base classes and protocol for networks training on tabular data, m2py
labels, or both.

Examples of features, usage, and data exploration are available in the *ipynb* folder.


### Background

SPM techniques have been pivotal in understanding surfaces, morphologies, and intermolecular interactions
of matter from the atomic to millimeter scales. Probes interacting with the material surface produce
2-dimensional images of the topography with intensity scales representing any number of material properties
(_e.g._ modulus, conductivity, or capacitance). In many experiments, multiple properties can be imaged
simultaneously, producing a 3-dimensional stack of these images.

By utilizing different combinations of computer vision and machine learning techniques, m2py leverages
differences in imaged  material properties to recognize different material phases, as well as 
different domains and topographical features. First, outlier pixels or signals can be removed. Next,
signals are compressed to only the most informative features via PCA. These signals can be deconvoluted
via GMM, or some other semantic segmenter for phase labeling. Finally, an instance segmenter, such as
Persistence Watershed Segmentation, clusters the phase-labeled pixels into domains, which receive
another label.

Once labeled by m2py segmenters, quantitative descriptions of each domain and the total morphology
are extracted and can be used to train supervised models to predict material properties or performance.
Such supervised training has not been accessible to most SPM researchers, due to the labor-intensive
nature of manual-labeling. 


#### Results from the MATDAT 2018 Hackathon

Wesley Tatum, Diego Torrejon, Patrick O’Neil

Thin films of semiconducting materials will enable stretchable and flexible electronic
devices, but these thin films are currently stochastic and inconsistent in their properties and
morphologies because processing and chemical conditions influence the mixing and domain size
of the different components. By using atomic force microscopy (AFM), a cheap and quick
technique, it is possible to spatially resolve and quantify these different domains based on
differences in their mechanical properties, which are strongly correlated to their electronic
performance. For this project, a library of AFM images has been curated, which includes poly(3-
hexylthiophene) that has been processed in different ways (e.g. annealing time and temperature,
thin film vs nanowire), as well as thin film mixtures of PTB7-th and PC 71 BM. To analyze these
samples, several semantic segmentation methods from the fields of machine learning and
topological data analysis are employed. Among these, a Gaussian mixture model utilizing
machine learned local geometric features proved effective. From the segmentation, probability
distributions describing the mechanical properties of each semantic segment can be obtained,
allowing the accurate classification of the various phase domains present in each sample.
