NaiveBayes
==========

**Naive Bayes text classification**

What is Naive bayes
-------------------

See also.

+ http://en.wikipedia.org/wiki/Naive_bayes


Tutorial
--------

The Bernoulli model.

+ http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

``` html
require 'naivebayes'
classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
classifier.train("positive", {"aaa" => 0, "bbb" => 1})
classifier.train("negative", {"ccc" => 2, "ddd" => 3})
result = classifier.classify({"aaa" => 1, "bbb" => 1})
p result # => {"positive" => 0.8767123287671234,"negative" => 0.12328767123287669}
```

Relation to multinomial unigram language model.

+ http://nlp.stanford.edu/IR-book/html/htmledition/relation-to-multinomial-unigram-language-model-1.html

``` html
require 'naivebayes'
classifier = NaiveBayes::Classifier.new(:model => "multinomial")
classifier.train("positive", {"aaa" => 0, "bbb" => 1})
classifier.train("negative", {"ccc" => 2, "ddd" => 3})
result = classifier.classify({"aaa" => 1, "bbb" => 1})
p result # => {"positive" => 0.9411764705882353,"negative" => 0.05882352941176469}
```

Complement Naive Bayes.

+ http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.13.8572

``` html
require 'naivebayes'
classifier = NaiveBayes::Classifier.new(:model => "complement", :smoothing_parameter => 1)
classifier.train("positive", {"aaa" => 3, "bbb" => 1, "ccc" => 2})
classifier.train("negative", {"aaa" => 1, "bbb" => 4, "ccc" => 2})
classifier.train("neutral",  {"aaa" => 2, "bbb" => 3, "ccc" => 5})
result = classifier.classify({"aaa" => 4, "bbb" => 3, "ccc" => 3})
p result #=> {"neutral"=>9.985931139006835, "negative"=>10.112101263742268, "positive"=>10.836883752313222}
```


ChangeLog
---------

See doc/ChangeLog.


Developers
----------

See doc/AUTHORS.


Author
------

**774**

+ http://id774.net
+ http://github.com/id774


Copyright and license
---------------------

See the file doc/LICENSE.


