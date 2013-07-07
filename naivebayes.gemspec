# Generated by jeweler
# DO NOT EDIT THIS FILE DIRECTLY
# Instead, edit Jeweler::Tasks in Rakefile, and run 'rake gemspec'
# -*- encoding: utf-8 -*-

Gem::Specification.new do |s|
  s.name = "naivebayes"
  s.version = "0.0.1"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.authors = ["id774"]
  s.date = "2013-07-07"
  s.description = "Naive Bayes classifier"
  s.email = "idnanashi@gmail.com"
  s.extra_rdoc_files = [
    "README.md"
  ]
  s.files = [
    "Gemfile",
    "README.md",
    "Rakefile",
    "VERSION",
    "doc/AUTHORS",
    "doc/COPYING",
    "doc/COPYING.LESSER",
    "doc/ChangeLog",
    "doc/LICENSE",
    "doc/README",
    "lib/naivebayes.rb",
    "lib/naivebayes/classifier.rb",
    "naivebayes.gemspec",
    "script/build",
    "spec/lib/naivebayes/classifier_spec.rb",
    "spec/lib/naivebayes_spec.rb",
    "spec/spec_helper.rb",
    "vendor/.gitkeep"
  ]
  s.homepage = "http://github.com/id774/naivebayes"
  s.licenses = ["GPL"]
  s.require_paths = ["lib"]
  s.rubygems_version = "2.0.3"
  s.summary = "naivebayes"

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_development_dependency(%q<cucumber>, [">= 0"])
      s.add_development_dependency(%q<bundler>, [">= 0"])
      s.add_development_dependency(%q<jeweler>, [">= 0"])
    else
      s.add_dependency(%q<cucumber>, [">= 0"])
      s.add_dependency(%q<bundler>, [">= 0"])
      s.add_dependency(%q<jeweler>, [">= 0"])
    end
  else
    s.add_dependency(%q<cucumber>, [">= 0"])
    s.add_dependency(%q<bundler>, [">= 0"])
    s.add_dependency(%q<jeweler>, [">= 0"])
  end
end

