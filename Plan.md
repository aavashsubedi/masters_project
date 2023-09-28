
### Combinatorial optimization on graphs for science operations in radio astronomy

*Modern radio telescopes are often built as arrays of antennas that operate interferometrically to synthesize apertures much larger than that provided by a single element. 
Due to improvements in digitisation, these arrays are able to be reconfigured in a flexible manner for different scientific applications, including division into 
heterogeneous sub-arrays, bandwidth partitioning and antenna/baseline reweighting, often for multiple sub-arrays operating in parallel. Choosing an optimal set of 
configuration parameters for such partitions is a combinatorial optimization problem, subject to complexities that are often specific to every individual scientific 
use-case. In this project the student will explore the potential of deep learning, including reinforcement learning, to learn the heuristics of array configuration 
for the science operations of the Square Kilometre Array Observatory (SKAO) and its associated facilities. In doing so, this work is intended to provide insight into 
how machine learning more generally can be employed to solve combinatorial optimization problems on graphs.*

---

### Work Plan:

* Reproduce one of the experiments from the [Differentiation of Combinatorial Solvers paper](https://arxiv.org/pdf/1912.02175.pdf), building your code in a modular
  way that allows it to be re-used for other problems;
* Consider changing the deep-learning approach to use a reinforcement learning or graph neural network architecture;  
* Replace the combinatorial problem with one that is more similar to the telescope array configuration problem, perhaps a node colouring problem?
* Implement a model that solves the telescope array configuration problem. Think carefully about what it is you want to optimise. 
