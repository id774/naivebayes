#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require File.dirname(__FILE__) + '/../../spec_helper'

describe NaiveBayes::Classifier do
  describe '#initialize' do
    context '@frequency_table with berounoulli model' do
      subject { classifier.frequency_table }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return frequency table' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "positive" => {"aaa" => 2, "bbb" => 2},
          "negative" => {"ccc" => 2, "ddd" => 2}
        }

        expect(subject).to eq expected
      end
    end

    context '@frequency_table with multinomial model' do
      subject { classifier.frequency_table }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return frequency table' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "positive" => {"aaa" => 1, "bbb" => 3},
          "negative" => {"ccc" => 5, "ddd" => 7}
        }

        expect(subject).to eq expected
      end
    end

    context '@word_table with berounoulli model' do
      subject { classifier.word_table }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return word table' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "aaa" => 1, "bbb" => 1,
          "ccc" => 1, "ddd" => 1
        }

        expect(subject).to eq expected
      end
    end

    context '@word_table with multinomial model' do
      subject { classifier.word_table }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return word table' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "aaa" => 1, "bbb" => 1,
          "ccc" => 1, "ddd" => 1
        }

        expect(subject).to eq expected
      end
    end

    context '@instance_count_of with berounoulli model' do
      subject { classifier.instance_count_of }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return instance_count_of' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "positive" => 2,
          "negative" => 2
        }

        expect(subject).to eq expected
      end
    end

    context '@instance_count_of with multinomial model' do
      subject { classifier.instance_count_of }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return instance_count_of' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = {
          "positive" => 2,
          "negative" => 2
        }

        expect(subject).to eq expected
      end
    end

    context '@total_count with berounoulli model' do
      subject { classifier.total_count }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return total count' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = 4

        expect(subject).to eq expected
      end
    end

    context '@total_count with multinomial model' do
      subject { classifier.total_count }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return total count' do
        classifier.train("positive", {"aaa" => 0, "bbb" => 1})
        classifier.train("negative", {"ccc" => 2, "ddd" => 3})
        classifier.train("positive", {"aaa" => 1, "bbb" => 2})
        classifier.train("negative", {"ccc" => 3, "ddd" => 4})

        expected = 4

        expect(subject).to eq expected
      end
    end

    context '@model with berounoulli model' do
      subject { classifier.model }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "berounoulli") }

      it 'should return model name' do
        expected = "berounoulli"
        expect(subject).to eq expected
      end
    end

    context '@model with multinomial model' do
      subject { classifier.model }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "multinomial") }

      it 'should return model name' do
        expected = "multinomial"
        expect(subject).to eq expected
      end
    end
  end
end

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
          "positive" => 0.2272727272727273,
          "negative" => 0.22727272727272724,
          "neutral"  => 0.5454545454545455
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

describe NaiveBayes::Classifier do
  describe 'The complement model' do
    context 'with train data of three expecting positive' do

      subject { classifier.classify({"aaa" => 4, "bbb" => 3, "ccc" => 3}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "complement") }

      it 'should return positive' do
        classifier.train("positive", {"aaa" => 3, "bbb" => 1, "ccc" => 2})
        classifier.train("negative", {"aaa" => 1, "bbb" => 4, "ccc" => 2})
        classifier.train("neutral",  {"aaa" => 2, "bbb" => 3, "ccc" => 5})

        expected = {
          "neutral"  => 9.985931139006835,
          "negative" => 10.112101263742268,
          "positive" => 10.836883752313222
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting negative' do

      subject { classifier.classify({"aaa" => 3, "bbb" => 4, "ccc" => 3}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "complement") }

      it 'should return negative' do
        classifier.train("positive", {"aaa" => 3, "bbb" => 1, "ccc" => 2})
        classifier.train("negative", {"aaa" => 1, "bbb" => 4, "ccc" => 2})
        classifier.train("neutral",  {"aaa" => 2, "bbb" => 3, "ccc" => 5})

        expected = {
          "neutral"  => 9.80360958221288,
          "positive" => 10.143736571753276,
          "negative" => 10.294422820536223
        }

        expect(subject).to eq expected
      end
    end

    context 'with train data of three expecting neutral' do

      subject { classifier.classify({"aaa" => 3, "bbb" => 3, "ccc" => 5}) }

      let(:classifier) { NaiveBayes::Classifier.new(:model => "complement") }

      it 'should return neutral' do
        classifier.train("positive", {"aaa" => 3, "bbb" => 1, "ccc" => 2})
        classifier.train("negative", {"aaa" => 1, "bbb" => 4, "ccc" => 2})
        classifier.train("neutral",  {"aaa" => 2, "bbb" => 3, "ccc" => 5})

        expected = {
          "negative" => 10.68941662877709,
          "positive" => 11.06002730362743,
          "neutral"  => 11.149081948812517
        }

        expect(subject).to eq expected
      end
    end
  end
end
