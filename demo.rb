#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require 'naivebayes'

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
