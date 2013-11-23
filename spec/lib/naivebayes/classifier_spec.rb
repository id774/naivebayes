#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require File.dirname(__FILE__) + '/../../spec_helper'

describe NaiveBayes::Classifier do
  describe 'The berounoulli model' do
    context 'with train data of two expecting positive' do

      subject { classifier.classify({"aaa" => 1, "bbb" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return positive' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})

        expected = {
          "positive" => 0.8767123287671234,
          "negative" => 0.12328767123287669
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of two expecting negative' do

      subject { classifier.classify({"ccc" => 3, "ddd" => 3}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return negative' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})

        expected = {
          "positive" => 0.12328767123287668,
          "negative" => 0.8767123287671234
        }

        expect(subject).to eq expected
      end
    end
  end
end

describe NaiveBayes::Classifier do
  describe 'The multinomial model' do
    context 'with train data of two expecting positive' do

      subject { classifier.classify({"aaa" => 1, "bbb" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return positive' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})

        expected = {
          "positive" => 0.9411764705882353,
          "negative" => 0.05882352941176469
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of two expecting negative' do

      subject { classifier.classify({"ccc" => 3, "ddd" => 3}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return negative' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})

        expected = {
          "positive" => 0.0588235294117647,
          "negative" => 0.9411764705882353
        }

        expect(subject).to eq expected
      end
    end
  end
end

describe NaiveBayes::Classifier do
  describe 'The berounoulli model' do
    context 'with train data of three expecting positive' do

      subject { classifier.classify({"aaa" => 1, "bbb" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return positive' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive" => 0.7422680412371133,
          "negative" => 0.12886597938144329,
          "neutral"  => 0.12886597938144329
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting negative' do

      subject { classifier.classify({"ccc" => 3, "ddd" => 2}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return negative' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive" => 0.12886597938144329,
          "negative" => 0.7422680412371133,
          "neutral"  => 0.12886597938144329
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting neutral' do

      subject { classifier.classify({"aaa" => 1, "ddd" => 2, "eee" => 3, "fff" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return neutral' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive"=>0.2272727272727273,
          "negative"=>0.22727272727272724,
          "neutral"=>0.5454545454545455
        }

        expect(subject).to eq expected
      end
    end
  end
end

describe NaiveBayes::Classifier do
  describe 'The multinomial model' do
    context 'with train data of three expecting positive' do

      subject { classifier.classify({"aaa" => 1, "bbb" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return positive' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive" => 0.896265560165975,
          "negative" => 0.06639004149377592,
          "neutral"  => 0.03734439834024896
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting negative' do

      subject { classifier.classify({"ccc" => 3, "ddd" => 2}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return negative' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive" => 0.05665722379603399,
          "negative" => 0.9178470254957508,
          "neutral"  => 0.0254957507082153
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting neutral' do

      subject { classifier.classify({"aaa" => 1, "ddd" => 2, "eee" => 3, "fff" => 1}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return neutral' do
        classifier.train("positive", {"aaa" => 2, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 2})
        classifier.train("neutral",  {"eee" => 3, "fff" => 3})

        expected = {
          "positive" => 0.12195121951219513,
          "negative" => 0.09756097560975606,
          "neutral"  => 0.7804878048780488
        }

        expect(subject).to eq expected
      end
    end
  end
end
