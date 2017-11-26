package dcs

object LearnTypes {
  type Feature = String
  type MyLearner = Learner[Example,Feature]
  type MyPoint = Point[Example,Feature]
}
