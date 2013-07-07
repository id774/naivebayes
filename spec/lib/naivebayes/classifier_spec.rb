#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require File.dirname(__FILE__) + '/../../spec_helper'

def train_by_2
  @classifier.train("positive", {"aaa" => 0, "bbb" => 1})
  @classifier.train("negative", {"ccc" => 2, "ddd" => 3})
end

def train_by_3
  @classifier.train("positive", {"aaa" => 2, "bbb" => 1})
  @classifier.train("negative", {"ccc" => 2, "ddd" => 2})
  @classifier.train("neutral",  {"eee" => 3, "fff" => 3})
end

describe NaiveBayes::Classifier, 'ナイーブベイズ' do
  context '多変数ベルヌーイモデルにおいて' do
    describe '2 つの教師データで positive が期待される値を与えると' do
      it 'positive が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
        train_by_2
        expect = {
          "positive" => 0.8767123287671234,
          "negative" => 0.12328767123287669
        }
        result = @classifier.classify({"aaa" => 1, "bbb" => 1})
        result.should == expect
      end
    end
    describe '2 つの教師データで negative が期待される値を与えると' do
      it 'negative が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
        train_by_2
        expect = {
          "positive" => 0.12328767123287668,
          "negative" => 0.8767123287671234
        }
        result = @classifier.classify({"ccc" => 3, "ddd" => 3})
        result.should == expect
      end
    end
  end
end

describe NaiveBayes::Classifier, 'ナイーブベイズ' do
  context '多項分布モデルにおいて' do
    describe '2 つの教師データで positive が期待される値を与えると' do
      it 'positive が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "multinomial")
        train_by_2
        expect = {
          "positive" => 0.9411764705882353,
          "negative" => 0.05882352941176469
        }
        result = @classifier.classify({"aaa" => 1, "bbb" => 1})
        result.should == expect
      end
    end
    describe '2 つの教師データで negative が期待される値を与えると' do
      it 'negative が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "multinomial")
        train_by_2
        expect = {
          "positive" => 0.0588235294117647,
          "negative" => 0.9411764705882353
        }
        result = @classifier.classify({"ccc" => 3, "ddd" => 3})
        result.should == expect
      end
    end
  end
end

describe NaiveBayes::Classifier, 'ナイーブベイズ' do
  context '多変数ベルヌーイモデルにおいて' do
    describe '3 つの教師データで positive が期待される値を与えると' do
      it 'positive が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
        train_by_3
        expect = {
          "positive" => 0.7422680412371133,
          "negative" => 0.12886597938144329,
          "neutral"  => 0.12886597938144329
        }
        result = @classifier.classify({"aaa" => 1, "bbb" => 1})
        result.should == expect
      end
    end
    describe '3 つの教師データで negative が期待される値を与えると' do
      it 'negative が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
        train_by_3
        expect = {
          "positive" => 0.12886597938144329,
          "negative" => 0.7422680412371133,
          "neutral"  => 0.12886597938144329
        }
        result = @classifier.classify({"ccc" => 3, "ddd" => 2})
        result.should == expect
      end
    end
    describe '3 つの教師データで neutral が期待される値を与えると' do
      it 'neutral が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "berounoulli")
        train_by_3
        expect = {
          "positive" => 0.2272727272727273,
          "negative" => 0.22727272727272724,
          "neutral"  => 0.5454545454545455
        }
        result = @classifier.classify({"aaa" => 1, "ddd" => 2, "eee" => 3, "fff" => 1})
        result.should == expect
      end
    end
  end
end

describe NaiveBayes::Classifier, 'ナイーブベイズ' do
  context '多項分布モデルにおいて' do
    describe '3 つの教師データで positive が期待される値を与えると' do
      it 'positive が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "multinomial")
        train_by_3
        expect = {
          "positive" => 0.896265560165975,
          "negative" => 0.06639004149377592,
          "neutral"  => 0.03734439834024896
        }
        result = @classifier.classify({"aaa" => 1, "bbb" => 1})
        result.should == expect
      end
    end
    describe '3 つの教師データで negative が期待される値を与えると' do
      it 'negative が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "multinomial")
        train_by_3
        expect = {
          "positive" => 0.05665722379603399,
          "negative" => 0.9178470254957508,
          "neutral"  => 0.0254957507082153
        }
        result = @classifier.classify({"ccc" => 3, "ddd" => 2})
        result.should == expect
      end
    end
    describe '3 つの教師データで neutral が期待される値を与えると' do
      it 'neutral が返る' do
        @classifier = NaiveBayes::Classifier.new(:model => "multinomial")
        train_by_3
        expect = {
          "positive" => 0.12195121951219513,
          "negative" => 0.09756097560975606,
          "neutral"  => 0.7804878048780488
        }
        result = @classifier.classify({"aaa" => 1, "ddd" => 2, "eee" => 3, "fff" => 1})
        result.should == expect
      end
    end
  end
end
