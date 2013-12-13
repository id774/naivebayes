# -*- coding: utf-8 -*-

require File.dirname(__FILE__) + '/../spec_helper'

describe NaiveBayes do
  context "VERSION" do
    subject { NaiveBayes::VERSION }

    it { expect(subject).to eq "0.0.3" }
  end
end
