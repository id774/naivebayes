#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

$:.unshift File.join(File.dirname(__FILE__))

require 'lib/naivebayes'

puts "--- The Bernoulli model ---"
classifier = NaiveBayes::Classifier.new(:model => "berounoulli")

classifier.train("positive", {"aaa" => 0, "bbb" => 1})
classifier.train("negative", {"ccc" => 2, "ddd" => 3})
result = classifier.classify({"aaa" => 1, "bbb" => 1})

p result # => {"positive" => 0.8767123287671234,"negative" => 0.12328767123287669}

puts "--- Relation to multinomial unigram language model ---"

classifier = NaiveBayes::Classifier.new(:model => "multinomial")

classifier.train("positive", {"aaa" => 0, "bbb" => 1})
classifier.train("negative", {"ccc" => 2, "ddd" => 3})
result = classifier.classify({"aaa" => 1, "bbb" => 1})

p result # => {"positive" => 0.9411764705882353,"negative" => 0.05882352941176469}

classifier = NaiveBayes::Classifier.new(:model => "complement", :smoothing_parameter => 1)

classifier.train("positive", {"aaa" => 3, "bbb" => 1, "ccc" => 2})
classifier.train("negative", {"aaa" => 1, "bbb" => 4, "ccc" => 2})
classifier.train("neutral",  {"aaa" => 2, "bbb" => 3, "ccc" => 5})
result = classifier.classify({"aaa" => 4, "bbb" => 3, "ccc" => 3})

p result #=> {"neutral"=>9.985931139006835, "negative"=>10.112101263742268, "positive"=>10.836883752313222}

