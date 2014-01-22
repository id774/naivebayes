#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

module NaiveBayes
  class Classifier
    attr_accessor :frequency_table, :word_table, :instance_count_of, :total_count, :model

    def initialize(params = {})
      @frequency_table = Hash.new
      @word_table = Hash.new
      @instance_count_of = Hash.new(0)
      @total_count = 0
      @model = params[:model]
      @smoothing_parameter = params[:smoothing_parameter] || 1
    end

    def train(label, feature)
      unless @frequency_table.has_key?(label)
        @frequency_table[label] = Hash.new(0)
      end
      feature.each {|word, frequency|
        if @model == "berounoulli"
          @frequency_table[label][word] += 1
        else
          @frequency_table[label][word] += frequency
        end
        @word_table[word] = 1
      }
      @instance_count_of[label] += 1
      @total_count += 1
    end

    def classify(feature)
      class_prior_of = Hash.new(1)
      likelihood_of = Hash.new(1)
      class_posterior_of = Hash.new(1)
      evidence = 0
      @instance_count_of.each {|label, freq|
        class_prior_of[label] = freq.to_f / @total_count.to_f
      }
      @frequency_table.each_key {|label|
        likelihood_of[label] = 1
        @word_table.each_key {|word|
          laplace_word_likelihood = (@frequency_table[label][word] + 1).to_f /
            (@instance_count_of[label] + @word_table.size()).to_f
          if feature.has_key?(word)
            likelihood_of[label] *= laplace_word_likelihood
          else
            likelihood_of[label] *= (1 - laplace_word_likelihood)
          end
        }
        class_posterior_of[label] = class_prior_of[label] * likelihood_of[label]
        evidence += class_posterior_of[label]
      }
      class_posterior_of.each {|label, posterior|
        class_posterior_of[label] = posterior / evidence
      }
      return class_posterior_of
    end
  end
end
