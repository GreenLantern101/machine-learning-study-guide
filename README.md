# Self-Study Guide for ML and Deep Learning

A no-bullshit personal guide that assumes competence in multivariable calculus, linear algebra, and machine-learning basics.
[Advice from the author of Keras](https://www.quora.com/What-advice-would-you-give-to-people-studying-ML-DL-from-MOOCs-Udacity-Coursera-edx-MIT-Opencourseware-or-from-books-in-their-own-time/answer/Fran%C3%A7ois-Chollet)

## Libraries
ML
  - scikit-learn
  - AirBnB's [aerosolve](https://github.com/airbnb/aerosolve)

Neural networks/deep-learning
  - [Keras](https://github.com/fchollet/keras) (high level)
    - [Theano vs. TensorFlow backend](https://www.quora.com/Do-you-recommend-using-Theano-or-Tensor-Flow-as-Keras-backend)
  - [TFLearn](https://github.com/tflearn/tflearn) - high-level API for deep-learning w/ TensorFlow
  - Google's [Sonnet](https://github.com/deepmind/sonnet) - Tensorflow-based neural network library
  - UC Berkeley's [Caffe](https://github.com/BVLC/caffe)
  - Facebook's [Torch](http://torch.ch/) - losing popularity?
  - [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano
  - [nolearn](https://github.com/dnouri/nolearn) - scikit-learn compatible neural network library, wrapper around Lasagne

Low-level math libraries:
  - Google's [TensorFlow](https://github.com/tensorflow/tensorflow) (lower level)
  - [Theano](https://github.com/Theano/Theano) (very low level)

## Cool Examples
- [Part 3: Deep Learning and Convolutional Neural Networks](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.44rhxy637)
- [Part 5: Language Translation with Deep Learning and the Magic of Sequences](https://medium.com/@ageitgey/machine-learning-is-fun-part-5-language-translation-with-deep-learning-and-the-magic-of-sequences-2ace0acca0aa#.wyfthap4c)
- [Part 6: Speech Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a)
- [Part 7: Using GANs to make art](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7)
- [Deep Learning the stock market](https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02)

## Short video series
- [Neural Networks, Demystified](http://lumiverse.io/series/neural-networks-demystified)
- [Nuts and Bolts of Applying Deep Learning - Andrew Ng](https://www.youtube.com/watch?v=F1ka6a13S9I)
- [Imbalanced Data sets](https://www.youtube.com/watch?v=X9MZtvvQDR4)

## Free Ebooks
- [Probabilistic Programming & Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- [Reinforcement Learning: An Introduction (2nd Edition)](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)
    - [GitHub repository](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)

## Classic Books
- Hastie, Tibshirani, and Friedman's [The Elements of Statistical Learning](https://www.goodreads.com/book/show/148009.The_Elements_of_Statistical_Learning)
- Bishop's [Pattern Recognition and Machine Learning](https://www.goodreads.com/book/show/55881.Pattern_Recognition_and_Machine_Learning)
- Kevin Murphy's [Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)
- David Barber's [Bayesian Reasoning and Machine Learning free pdf](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/020217.pdf)
- Mitchell's [Machine Learning](https://www.goodreads.com/book/show/213030.Machine_Learning)

^^^ those five are recommended by [Xavier Amatriain](https://www.quora.com/How-do-I-learn-machine-learning-1/answer/Xavier-Amatriain) and others and are highly rated.

Others:
  - [Python Machine Learning](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130) and [Github](https://github.com/rasbt/python-machine-learning-book)

## Classes
- [Udacity: Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120)
- [Udacity: Supervised, Unsupervised & Reinforcement](https://www.udacity.com/course/machine-learning--ud262)
- [Andrew Ng's Coursera’s Machine Learning](https://www.youtube.com/playlist?list=PLZ9qNFMHZ-A4rycgrgOYma6zxF4BZGGPW)
    - [Coursera: Machine Learning Roadmap](https://metacademy.org/roadmaps/cjrd/coursera_ml_supplement)
- [mathematicalmonk's Machine Learning tutorials](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA)
- [Paul G. Allen School](https://www.youtube.com/user/UWCSE/playlists?shelf_id=16&sort=dd&view=50)

- University courses
    - [CMU machine learning course - intro level](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml)
    - [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/materials.html)
    - [Oxford - Machine Learning & Deep Learning](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)

## Deep Learning
Blogs:
- Christopher Olah's easy-to-understand [blog](http://colah.github.io/)
- Andrej Karpathy's interesting [blog](http://karpathy.github.io/)
  - post on [RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - post on [Deep Reinforcement Learning](http://karpathy.github.io/2016/05/31/rl/)

Notes:
- types of ANNs(articifial neural networks): RBF networks, convolutional networks, RNNs (recurrent neural networks)

Free Classic Ebooks:
- Nielson's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - easier
- Yoshua Bengio's [Deep Learning](http://www.deeplearningbook.org/) - more advanced

Classes:
- [Neural networks class - Université de Sherbrooke](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
- [Udacity: Deep Learning](https://classroom.udacity.com/courses/ud730)
- [6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/schedule.html)
- [6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/)
- [Stanford's Unsupervised Feature Learning and Deep Learning](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial)
- [Stanford CS224D - Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)
- [Stanford CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

## List of resource lists & Misc.
- Todo: to get a deep understanding of algorithms, implement them from scratch (start w/ logistic regression & k-means, work to more complicated & interesting algorithms)
- eg. [A collection of minimal and clean implementations of machine learning algorithms](https://github.com/rushter/MLAlgorithms)


## Interview Questions
- [How To Prepare For A Machine Learning Interview](http://blog.udacity.com/2016/05/prepare-machine-learning-interview.html)
- [40 Interview Questions asked at Startups in Machine Learning / Data Science](https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science)
- [21 Must-Know Data Science Interview Questions and Answers](http://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html)
- [Top 50 Machine learning Interview questions & Answers](http://career.guru99.com/top-50-interview-questions-on-machine-learning/)
- [Machine Learning Engineer interview questions](https://resources.workable.com/machine-learning-engineer-interview-questions)
- [Popular Machine Learning Interview Questions](http://www.learn4master.com/machine-learning/popular-machine-learning-interview-questions)
- [What are some common Machine Learning interview questions?](https://www.quora.com/What-are-some-common-Machine-Learning-interview-questions)
- [What are the best interview questions to evaluate a machine learning researcher?](https://www.quora.com/What-are-the-best-interview-questions-to-evaluate-a-machine-learning-researcher)
- [Collection of Machine Learning Interview Questions](http://analyticscosm.com/machine-learning-interview-questions-for-data-scientist-interview/)
- [121 Essential Machine Learning Questions & Answers](https://learn.elitedatascience.com/mlqa-welcome)
