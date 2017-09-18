# OGN

This is the repo containing the code for our CS468 course project, Spring 2017. I implement the Caffe layers to discriminate Octree reprensented shapes, including sparse convolution and pooling to do auto-encoder and GAN experiments using our discriminator (encoder) and OGN paper method as generator (decoder).

## Original OGN paper

Check `ORI_README.md` or [this github repo](https://github.com/lmb-freiburg/ogn) for the original [OGN paper](https://arxiv.org/abs/1703.09438) implementation.

<p align="center"> 
<img src="https://github.com/mtatarchenko/ogn/blob/master/thumbnail.png">
</p>

## This repo

This github repo contains more GPU supported Caffe layers implementation to augment the original OGN paper experiments with auto-encoder and GAN experiments.
I implement the Caffe layers to discriminate Octree reprensented shapes, including sparse convolution and pooling.

<p align="center"> 
<img src="https://github.com/daerduoCarey/ogn/blob/master/teaser.png">
</p>

In the above figures, the first rows are OGN generator (decoder) layers implemented by [this github repo](https://github.com/lmb-freiburg/ogn). The rest rows are discriminator (encoder) layers implemented in this repo.

This repo contains the implementation for `OGN Select Level Layer`, `OGN Level Prediction Layer`, `OGN Down Convolution Layer`, `OGN Concatenate Layer` and `OGN sparse-to-dense Layer` to form the discriminator (encoder) of a Octree-represented shape. Refer to our report for more details.

We successfully make it work for the auto-encoder experiment. But due to time limit and some techinical issue, we believe the current framework is hard to make it work for GAN experiments. More training supervision signals, like the idea of Stacked GAN, should be helpful!

Here is the [report](http://www.cs.stanford.edu/~kaichun/resume/cs468_project_report.pdf).

# Disclaimer

Use of our your own risk. No guarantee on code correctness and effectiveness.
This project is done with collaboration with Karen Yang and Te-lin Wu.

## Contact
Webpage: http://www.cs.stanford.edu/~kaichun/ 

E-mail: kaichun [at] cs.stanford.edu

## License
MIT

