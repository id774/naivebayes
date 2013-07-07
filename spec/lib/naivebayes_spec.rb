# -*- coding: utf-8 -*-

require File.dirname(__FILE__) + '/../spec_helper'

describe NaiveBayes do
  context 'のバージョンを参照した場合' do
    it "バージョンが正しく表示される" do
      expect = '0.0.1'
      NaiveBayes.const_get(:VERSION).should be_true
      NaiveBayes.const_get(:VERSION).should == expect
    end
  end
end
