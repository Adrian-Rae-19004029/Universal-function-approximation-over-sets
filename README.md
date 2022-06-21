# Universal function approximation over sets

## Introduction

---
This repository contains the code needed to realize the research objectives established by Adrian Rae in the dissertation entitled "Universal function approximation over sets".

Many machine learning practices make use of fixed-length vectors to represent objects. While this representation is sensible for many real-world tasks, it is ill-suited to tasks where the number of elements in an input collection is allowed to vary. Deep Learning, the subset of machine learning devoted to solving tasks by drawing inspiration from the behaviour of the human brain, evokes a particular need for mechanisms which operate on objects with less rigid structure than vectors, such as sets and sequences. Sets are a natural choice for modelling certain real-world objects as they provide a means of expressing a collection of arbitrarily many elements where no intrinsic ordering exists between them. Processes which act upon sets as inputs can effectively be modelled as functions which act on an arbitrary number of input elements, and exhibit a property known as permutation invariance. This research considers the challenges of approximating these functions of sets and motivates the need for deriving models which allow for such approximation. We investigate certain state-of-the-art mechanisms and architectures which employ similar approximations and draw inspiration from the behaviour of other permutation-invariant processes in order to prompt the creation and evaluation of models for approximating arbitrary functions of sets.
